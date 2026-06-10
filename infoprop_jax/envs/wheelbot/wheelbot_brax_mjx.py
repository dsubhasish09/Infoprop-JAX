"""
Wheelbot environment using MuJoCo/MJX physics — the example `InfopropWrappable`.

`WheelbotEnv` plays two roles with one observation/reward/state layout:
  - the ground-truth env (its real `step`) for collecting training data and final evaluation, and
  - the env wrapped by `InfopropEnv` for imagined model rollouts, via the Infoprop hooks
    (`preprocess` / `augment_prediction` / `postprocess` / `reset_from_buffer`).

See infoprop_jax/envs/README.md for the contract and infoprop_jax/envs/wheelbot/README.md for the
Wheelbot state/observation/reward details.

Robot dynamics summary:
  - 2-wheeled differential robot with a reaction wheel for pitch balance.
  - Action: [driving_torque, reaction_wheel_torque] (2D).
  - Total torque = RL action + linear balancing prior (tau_bal).
  - Fixed-frequency control (multiple MJX steps per control step).

Variant physics state (11D, learned by the dynamics model):
  [roll, pitch, yaw_rate, roll_rate, pitch_rate,
   drive_angle_rate, balance_angle_rate,
   body_vx, body_vy, body_vz, z]

Invariant state (5D, global odometry):
  [yaw, drive_angle, balance_angle, x, y]
"""
from jax import numpy as jp
from jax.scipy.spatial.transform import Rotation
from etils import epath
import jax
from jax import lax
from mujoco import mjx
from typing import Optional
import mujoco
from mujoco import mjx
from brax.io import mjcf
from brax.envs.base import PipelineEnv, State
from brax.training.types import Transition
from brax import envs
from infoprop_jax.envs.infoprop_wrappable_env import InfopropWrappable
import mediapy as media
from omegaconf import dictconfig, OmegaConf
import xml.etree.ElementTree as ET
from .utils import compute_line_element
from .assets.track.generator import create_track, load_track_by_seed
from .trajectory import Trajectory, pad_line_segments_to_size, points_to_line_segemnt


def Rx(theta):
    """Rotation matrix around x-axis."""
    return jp.array([[1, 0, 0],
                     [0, jp.cos(theta), -jp.sin(theta)],
                     [0, jp.sin(theta), jp.cos(theta)]])


def Ry(theta):
    """Rotation matrix around y-axis."""
    return jp.array([[jp.cos(theta), 0, jp.sin(theta)],
                     [0, 1, 0],
                     [-jp.sin(theta), 0, jp.cos(theta)]])


def Rz(theta):
    """Rotation matrix around z-axis."""
    return jp.array([[jp.cos(theta), -jp.sin(theta), 0],
                     [jp.sin(theta), jp.cos(theta), 0],
                     [0, 0, 1]])


def YRP(yaw, roll, pitch):
    return Rz(yaw) @ Rx(roll) @ Ry(pitch)


def skew_symmetric(w):
    """Skew symmetric matrix."""
    v = w.flatten()
    return jp.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def mat2Quat(mat):
    """Converts a rotation matrix to a quaternion."""
    r = Rotation.from_matrix(mat)
    q = r.as_quat(scalar_first=True)
    return q


def quat2Mat(q):
    """Converts a quaternion to a rotation matrix."""
    r = Rotation.from_quat(q[jp.array([1, 2, 3, 0])])
    return r.as_matrix()


def jacobian_w2euler(q1, q2):
    return jp.array([[jp.cos(q2), 0, jp.sin(q2)],
                     [jp.sin(q2) * jp.tan(q1), 1, -jp.cos(q2) * jp.tan(q1)],
                     [-jp.sin(q2) / jp.cos(q1), 0, jp.cos(q2) / jp.cos(q1)]])

del_jacobian_w2euler_del_q1 = (jax.jacobian(jacobian_w2euler, 0))
del_jacobian_w2euler_del_q2 = (jax.jacobian(jacobian_w2euler, 1))


def jacobian_dot_w2euler(q1, q2, dq1, dq2):
    jac_dot = (
        del_jacobian_w2euler_del_q1(q1, q2) * jp.array([dq1])
        + del_jacobian_w2euler_del_q2(q1, q2) * jp.array([dq2])
    )
    return jac_dot


def robot_state_to_qpos_qvel(state, O_r_OC=None):
    """Convert the 10D full robot state [yaw, roll, pitch, rates..., drive_angle, balance_angle, rates] to MuJoCo qpos/qvel arrays."""
    C_R_W = Ry(state[2])
    O_R_C = Rz(state[0]) @ Rx(state[1])
    C_r_CW = jp.array([[0, 0, 0.032]]).T
    W_r_WB = jp.array([[0, 0, 0.0325]]).T
    O_r_dot_OC = (
        0.032 * (state[5] + state[7])
        * jp.array([[jp.cos(state[0]), jp.sin(state[0]), 0]]).T
    )
    O_w_C = jp.array([[0, 0, state[3]]]).T + Rz(state[0]) @ jp.array([[state[4], 0, 0]]).T
    C_w_W = jp.array([[0, state[5], 0]]).T
    S_O_w_C = skew_symmetric(O_w_C)
    S_C_w_W = skew_symmetric(C_w_W)
    if O_r_OC is None:
        O_r_OC = jp.zeros((3, 1))
    O_r_OB = O_r_OC + O_R_C @ C_r_CW + O_R_C @ C_R_W @ W_r_WB
    R = O_R_C @ C_R_W
    quat = mat2Quat(R)
    qpos = jp.array((*(O_r_OB.flatten()), *(quat.flatten()), state[6], state[8]))
    O_w_B = R.T @ (O_w_C + O_R_C @ C_R_W @ C_w_W)
    O_r_dot_OB = (
        O_r_dot_OC
        + S_O_w_C @ O_R_C @ (C_r_CW + C_R_W @ W_r_WB)
        + O_R_C @ S_C_w_W @ C_R_W @ W_r_WB
    )
    qvel = jp.array((*(O_r_dot_OB.flatten()), *(O_w_B.flatten()), state[7], state[9]))
    return qpos, qvel


def qpos_qvel_to_robot_state(qpos, qvel):
    """Convert MuJoCo qpos/qvel arrays to the 10D full robot state [yaw, roll, pitch, rates..., drive_angle, balance_angle, rates]."""
    quat = qpos[3:7]
    R = quat2Mat(quat)
    R = R.reshape((3, 3))
    phi = jp.arctan2(-R[0, 1], R[1, 1])
    theta = jp.arcsin(R[2, 1])
    psi = jp.arctan2(-R[2, 0], R[2, 2])

    alpha = qpos[7]
    beta = qpos[8]

    omega = qvel[3:6]
    euler = jacobian_w2euler(theta, psi) @ omega

    euler = euler[jp.array([2, 0, 1])]
    alpha_dot = qvel[6]
    beta_dot = qvel[7]

    state = jp.array([phi, theta, psi, *euler, alpha, alpha_dot, beta, beta_dot])
    return state


lookahead = 10


def get_trajectory_by_seed(track_seed: int, lookahead: int = lookahead) -> Trajectory:
    """Load the pre-generated track trajectory for the given integer seed."""
    traj_flattened = trajectories_flattened[track_seed]
    size = trajectory_lengths[track_seed]
    return Trajectory(traj_flattened, size, lookahead)


def get_projected_velocity(State):
    """Extract the longitudinal (forward) velocity component in the robot body frame."""
    physics_state = State.info['physics_state']
    long_vel_body_frame = (
        Ry(physics_state[0]) @ Rx(physics_state[1]) @ physics_state[-4:-1][:, None]
    )[0, 0]
    proj_vel_body_frame = long_vel_body_frame * jp.cos(-State.obs[1])
    return proj_vel_body_frame


# Static index arrays for fancy indexing — use arrays to avoid deprecated list indexing
_VARIANT_INDICES = jp.array([1, 2, 3, 4, 5, 7, 9])
_PITCH_CTRL_INDICES = jp.array([2, 5, 6, 7])
_ROLL_CTRL_INDICES = jp.array([1, 4, 8, 9])
# Physics-state column indices used in the model's odometry integration
_EULER_RATE_IDX = jp.array([2, 5, 6])   # roll_dot, pitch_dot, yaw_dot
_BODY_VEL_IDX = jp.array([7, 8, 9])     # body vx, vy, vz

WHEELBOT_ASSET_PATH = (
    epath.Path(epath.resource_path('infoprop_jax')) / 'envs' / 'wheelbot' / 'assets'
)

# Create trajectories statically to be used in reset method.
tracks = [load_track_by_seed(i) for i in range(200)]
track_width = tracks[0]['width']
centerlines = [track['centerline'] for track in tracks]
trajectories = [points_to_line_segemnt(centerline) for centerline in centerlines]
trajectory_lengths = jp.array([t.shape[0] for t in trajectories])
max_length = max(trajectory_lengths)
trajectories_flattened = jp.array([pad_line_segments_to_size(t, max_length) for t in trajectories])


class WheelbotEnv(InfopropWrappable):
    """Wheelbot MJX env: real MuJoCo-MJX physics + the Infoprop hooks.

    As an ``InfopropWrappable`` it is both the ground-truth data-collection / eval env
    (via its real ``step``) and the env that ``InfopropEnv`` wraps for learned-dynamics
    rollouts (via ``preprocess`` / ``augment_prediction`` / ``postprocess``). The
    ``preprocess`` hook also applies the balancing prior. The two roles share one
    observation, reward and state layout.

    Attributes (beyond base PipelineEnv):
        K_roll / K_pitch: Linear balancing controller gains (4D each).
        meas_noise_std / process_noise_std: Per-state noise standard deviations.
        obs_history / act_history: Number of past steps included in the observation.
    """

    def __init__(
        self,
        cfg: dictconfig.DictConfig = dictconfig.DictConfig({}),
        visualize: bool = False,
        track_seed: Optional[int] = None,
        eval_mode: bool = False,
        **kwargs,
    ):
        """Initialise the MJX environment: load the MuJoCo XML, configure the solver,
        and pre-load all training tracks from disk.
        """
        mjcf_path = (WHEELBOT_ASSET_PATH / 'mjcf').as_posix()
        xml_path = mjcf_path + '/wheelbot_alpha.xml'
        xml_str = epath.Path(xml_path).read_text()

        root = ET.fromstring(xml_str)
        # Set the mesh file paths.
        asset = root.find('asset')
        for mesh in asset.findall('mesh'):
            if mesh.attrib.get('name') == 'body' or mesh.attrib.get('name') == 'wheel':
                relative_file = mesh.get('file')
                mesh.set('file', mjcf_path + '/' + relative_file)

        if track_seed is not None:
            self.track_seed = track_seed
            track = tracks[self.track_seed]

            if visualize:
                inner_cones = track['inner_cones']
                outer_cones = track['outer_cones']

                track_body = root.find(".//body[@name='track']")
                outer_cone_line_elements = [
                    compute_line_element(outer_cones[i], outer_cones[i + 1])
                    for i in range(len(outer_cones) - 1)
                ]
                inner_cone_line_elements = [
                    compute_line_element(inner_cones[i], inner_cones[i + 1])
                    for i in range(len(inner_cones) - 1)
                ]

                track_line_elements = outer_cone_line_elements + inner_cone_line_elements

                for elem in track_line_elements:
                    track_body.append(elem)

        xml_str = ET.tostring(root, encoding='unicode')

        # Create the mujoco model from the xml file.
        mj_model = mujoco.MjModel.from_xml_string(xml_str)
        # Set the solver.
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        # Set the number of iterations for the solver.
        mj_model.opt.iterations = 6
        # Set the number of line search iterations inside each solver iteration.
        mj_model.opt.ls_iterations = 6
        # Timestep of the simulation.
        mj_model.opt.timestep = 0.001

        # Load the model into the mjx system.
        sys = mjcf.load_model(mj_model)

        kwargs['n_frames'] = kwargs.get('n_frames', cfg.get('control_interval', 6))
        kwargs['backend'] = 'mjx'

        super().__init__(sys, **kwargs)
        self.K_roll = jp.array(cfg.get('K_roll', [-1.3e0, -1.6e-1, -0.8e-04, -4e-04]))
        self.K_pitch = jp.array(cfg.get('K_pitch', [-400e-3, -40e-3, -4e-3, -3e-3]))
        self.max_state = jp.array([*(cfg.get(
            'max_state',
            [2 * jp.pi, jp.pi / 4, jp.pi / 4, 40, 40, 40, jp.inf, 800, jp.inf, 800],
        ))])
        self.action_lim = cfg.get('action_lim', 0.5)
        # Zeros the robot_state values that are not used in trajectory driving.
        self.robot_state_mask = jp.array([0, 1, 1, 1, 1, 1, 0, 1, 0, 1])
        # Reward parameters.
        self.action_scale = cfg.get('action_scale', 0.1)
        self.ca_weight = cfg.get('ca_weight', 1.0)
        self.ct_weight = cfg.get('ct_weight', 1.0)
        self.driving_weight = cfg.get('driving_weight', 1.0)
        self.crash_penalty = cfg.get('crash_penalty', -1000)
        self.rew_scale = cfg.get('rew_scale', 1e-3)
        self.eval_mode = eval_mode
        # Randomize env parameters (real-env data-collection reset).
        self.init_xy_std = cfg.get('init_xy_std', track_width / 16)
        self.init_angle_std = cfg.get('init_angle_std', jp.pi / 16)
        # Wider spread used when re-seeding model rollouts from the replay buffer
        # (reset_with_init_robot_state). Kept separate from the data-collection spread
        # above so model rollouts branch from a diverse initial-state distribution.
        self.resample_init_xy_std = cfg.get('resample_init_xy_std', track_width / 3)
        self.resample_init_angle_std = cfg.get('resample_init_angle_std', jp.pi / 3)
        # History parameters for model learning.
        self.obs_history = cfg.get('obs_history', 1)
        self.act_history = cfg.get('act_history', 0)
        # Env-owned fast-rollout flag: skip building the MJX pipeline_state during
        # model rollouts. The framework is agnostic to this; see InfopropWrappable.
        self.fast_model_rollout = cfg.get('fast_model_rollout', True)
        self.lookahead = cfg.get('lookahead', 10)
        self.sin_cos_encoding = cfg.get('sin_cos_encoding', False)
        meas_std = cfg.get('meas_noise_std', None)
        self.meas_noise_std = jp.array(meas_std) if meas_std is not None else None
        process_std = cfg.get('process_noise_std', None)
        self.process_noise_std = jp.array(process_std) if process_std is not None else None
        _enc = self.sin_cos_encoding
        self._obs_slice_start = (3 if _enc else 2) * self.lookahead + (3 if _enc else 2)
        # Dimensions of the learned model state (see infoprop_env.py for the split).
        self.model_state_size = 11
        self.context_size = 5
        self.full_state_size = self.model_state_size + self.context_size

    # ---------------------------------------------------- physics-buffer data contract
    @property
    def dummy_physics_transition(self) -> Transition:
        """Zero-filled transition that sizes the physics replay buffer and declares the
        per-transition context fields (``state_extras``) carried for model training and
        consumed by the model env's ``reset_from_buffer``."""
        ms, oh, ah = self.model_state_size, self.obs_history, self.act_history
        return Transition(
            observation=jp.zeros(ms * oh + self.action_size * ah),
            action=jp.zeros(self.action_size),
            reward=0.0,
            discount=0.0,
            next_observation=jp.zeros(ms),
            extras={'state_extras': {'truncation': 0.0, 'track_seed': 0,
                                     'invariant_physics_state': jp.zeros(self.context_size)},
                    'policy_extras': {}},
        )

    def extract_physics_transition(self, prev_state: State, next_state: State, policy_extras) -> Transition:
        """Build the (model_state_history+action_history -> next_model_state) transition
        with its context extras, from this env's own `info` keys."""
        state_extras = {
            'truncation': next_state.info['truncation'],
            'track_seed': next_state.info['track_seed'],
            'invariant_physics_state': next_state.info['invariant_physics_state'],
        }
        return Transition(
            observation=jp.concatenate(
                [prev_state.info['phys_state_history'], prev_state.info['act_history']], axis=-1),
            action=next_state.info['applied_torque'],
            reward=jp.zeros(next_state.reward.shape, dtype=jp.float32),
            discount=1 - next_state.done,
            next_observation=next_state.info['physics_state'],
            extras={'policy_extras': policy_extras, 'state_extras': state_extras},
        )

    def context_from_transition(self, transition: Transition):
        """Extract Wheelbot's invariant state context for cutoff evaluation."""
        return transition.extras['state_extras']['invariant_physics_state']

    @property
    def reset_carry_keys(self):
        """Env-owned dynamic info keys the real-env auto-reset wrapper reverts on `done`."""
        return ['physics_state', 'invariant_physics_state', 'applied_torque']

    def _get_obs(self, data: mjx.Data, action: jp.ndarray, track_seed: int) -> jp.ndarray:
        """Construct the observation vector: trajectory features + masked robot state history."""
        robot_state = qpos_qvel_to_robot_state(data.qpos, data.qvel)
        pos_xy = data.qpos[0:2]
        yaw = robot_state[0]
        trajectory = get_trajectory_by_seed(track_seed, self.lookahead)
        traj_state = trajectory.get_state(pos_xy, yaw, self.sin_cos_encoding)
        robot_state_masked = robot_state * self.robot_state_mask
        body_vel = (
            Ry(-robot_state[2]) @ Rx(-robot_state[1]) @ Rz(-robot_state[0])
            @ data.qvel[0:3][:, None]
        ).flatten()
        extras = jp.array([data.qpos[2], *body_vel])
        return jp.array([*traj_state, *robot_state_masked, *extras])

    def _get_rew(self, State, action):
        """Compute the step reward.

        reward = rew_scale * [
            (1 - done) * ct_weight  * (track_width/2 - |cte|) / (track_width/2)
          + (1 - done) * ca_weight  * (pi/2 - |cae|) / (pi/2)
          + (1 - done) * driving_weight * longitudinal_velocity
          + done       * crash_penalty
        ]
        """
        state = State.obs
        traj_state = state[:self._obs_slice_start]
        cross_track_error = traj_state[0]
        cross_angle_error = (
            traj_state[1] if not self.sin_cos_encoding
            else jp.arctan2(traj_state[1], traj_state[2])
        )
        cross_track_rew = (
            self.ct_weight * (track_width / 2 - jp.abs(cross_track_error)) / (track_width / 2)
        )
        cross_angle_rew = self.ca_weight * (jp.pi / 2 - jp.abs(cross_angle_error)) / (jp.pi / 2)

        robot_state = state[self._obs_slice_start:-4]
        done = jp.where((jp.abs(robot_state) > self.max_state).any(), 1.0, 0.0)
        track_violated = jp.where(jp.abs(cross_track_error) > track_width / 2, 1.0, 0.0)

        done = jp.where(track_violated, 1.0, done)
        projected_velocity = get_projected_velocity(State)
        driving_reward = self.driving_weight * projected_velocity
        crash_penalty = jp.float32(self.crash_penalty)

        reward = self.rew_scale * (
            (1 - done) * cross_track_rew
            + (1 - done) * cross_angle_rew
            + (1 - done) * driving_reward
            + done * crash_penalty
        )
        reward_metrics = {
            'cross_track_rew': cross_track_rew,
            'cross_angle_rew': cross_angle_rew,
            'driving_reward': driving_reward,
            'crash_penalty': crash_penalty,
        }
        return reward, done, reward_metrics

    def reset(self, rng: jp.ndarray) -> State:
        """Reset to a random track and a random starting position with a short warm-up."""
        # Get random track.
        rng, track_key = jax.random.split(rng)
        if self.eval_mode:
            track_seed = jax.random.randint(track_key, shape=(), minval=0, maxval=200)
        else:
            track_seed = jp.array(21)
        trajectory = get_trajectory_by_seed(track_seed)
        state = trajectory.get_state(jp.array([1, 1]), 0)
        # Get position on a random point on the trajectory.
        rng, pos_key = jax.random.split(rng)
        init_xy, init_angle = trajectory.get_rand_init_pos(pos_key)
        # Get random initial location.
        rng, xy_key = jax.random.split(rng)
        d = jp.clip(
            jax.random.normal(xy_key, shape=()) * self.init_xy_std,
            -track_width / 2,
            track_width / 2,
        )
        perp_dir = jp.array([-jp.sin(init_angle), jp.cos(init_angle)])
        offset_xy = d * perp_dir
        init_xy = init_xy + offset_xy
        # Get random initial angle.
        rng, angle_key = jax.random.split(rng)
        offset_angle = jp.clip(
            jax.random.normal(angle_key, shape=init_angle.shape) * self.init_angle_std,
            -jp.pi,
            jp.pi,
        )
        init_angle = init_angle + offset_angle
        # Create random state.
        init_robot_state = jp.array([init_angle, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        qpos, qvel = robot_state_to_qpos_qvel(init_robot_state)
        qpos = qpos.at[:2].set(init_xy)

        data = self.pipeline_init(qpos, qvel)

        # Get initial observation and physics state.
        action = jp.zeros(2)
        obs = self._get_obs(data, action, track_seed)

        # Initialize history arrays with zeros.
        act_history_array = jp.zeros(self.action_size * self.act_history) if self.act_history > 0 else jp.array([])
        phys_state_history_array = (
            jp.zeros(self.model_state_size * self.obs_history) if self.obs_history > 0 else jp.array([])
        )

        robot_state = qpos_qvel_to_robot_state(data.qpos, data.qvel)
        variant_physics_state = robot_state[_VARIANT_INDICES]
        body_vel = (
            Ry(-robot_state[2]) @ Rx(-robot_state[1]) @ Rz(-robot_state[0])
            @ data.qvel[0:3][:, None]
        ).flatten()

        metrics = {}

        reward_metrics = {
            'cross_track_rew': 0.0,
            'cross_angle_rew': 0.0,
            'driving_reward': 0.0,
            'crash_penalty': 0.0,
        }

        info = {
            'track_seed': track_seed,
            'applied_torque': action,
            'physics_state': jp.concatenate([variant_physics_state, body_vel, data.qpos[2:3]]),
            # Invariant physics state is the robot state without the velocity.
            'invariant_physics_state': jp.array(
                [robot_state[0], *(data.qpos[-2:]), *(data.qpos[:2])]
            ),
            'act_history': act_history_array,
            'phys_state_history': phys_state_history_array,
            'reward_metrics': reward_metrics,
            'rng': rng,
        }

        state = State(data, obs, 0.0, 0.0, metrics, info)
        warmup_steps = max(self.act_history, self.obs_history)

        def func(carry, _):
            state, action = carry
            next_state = self.step(state, action)
            return (next_state, action), None

        (state, action), _ = jax.lax.scan(func, (state, action), None, warmup_steps)

        return state

    def reset_to_start(self, rng) -> State:
        """Reset to the beginning of the track (segment 0) without randomisation."""
        trajectory = get_trajectory_by_seed(self.track_seed)

        rng, pos_key = jax.random.split(rng)
        init_xy, init_angle = trajectory.get_init_pos(0)
        # get random initial location
        # rng, xy_key = jax.random.split(rng)
        # offset_xy = jp.clip(jax.random.normal(xy_key, shape=init_xy.shape) * self.init_xy_std,
        #                     -0.9*track_width/2, 0.9*track_width/2)
        # init_xy = init_xy + offset_xy
        # get random initial angle
        # rng, angle_key = jax.random.split(rng)
        # offset_angle = jp.clip(jax.random.normal(angle_key, shape=init_angle.shape)
        #                        * self.init_angle_std, -jp.pi/4, jp.pi/4)
        # init_angle = init_angle + offset_angle
        # Create state at segment 0.
        init_robot_state = jp.array([init_angle, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        qpos, qvel = robot_state_to_qpos_qvel(init_robot_state)
        qpos = qpos.at[:2].set(init_xy)

        data = self.pipeline_init(qpos, qvel)

        # Get initial observation and physics state.
        action = jp.zeros(2)
        obs = self._get_obs(data, action, self.track_seed)

        # Initialize history arrays with zeros.
        act_history_array = jp.zeros(self.action_size * self.act_history) if self.act_history > 0 else jp.array([])
        phys_state_history_array = (
            jp.zeros(self.model_state_size * self.obs_history) if self.obs_history > 0 else jp.array([])
        )

        robot_state = qpos_qvel_to_robot_state(data.qpos, data.qvel)
        variant_physics_state = robot_state[_VARIANT_INDICES]
        body_vel = (
            Ry(-robot_state[2]) @ Rx(-robot_state[1]) @ Rz(-robot_state[0])
            @ data.qvel[0:3][:, None]
        ).flatten()

        metrics = {}

        reward_metrics = {
            'cross_track_rew': 0.0,
            'cross_angle_rew': 0.0,
            'driving_reward': 0.0,
            'crash_penalty': 0.0,
        }

        info = {
            'track_seed': self.track_seed,
            'applied_torque': action,
            'physics_state': jp.concatenate([variant_physics_state, body_vel, data.qpos[2:3]]),
            # Invariant physics state is the robot state without the velocity.
            'invariant_physics_state': jp.array(
                [robot_state[0], *(data.qpos[-2:]), *(data.qpos[:2])]
            ),
            'act_history': act_history_array,
            'phys_state_history': phys_state_history_array,
            'reward_metrics': reward_metrics,
            'rng': rng,
        }

        state = State(data, obs, 0.0, 0.0, metrics, info)
        warmup_steps = max(self.act_history, self.obs_history)

        def func(carry, _):
            state, action = carry
            next_state = self.step(state, action)
            return (next_state, action), None

        (state, action), _ = jax.lax.scan(func, (state, action), None, warmup_steps)

        return state

    def add_noise(self, rng, noise_std, physics_state, invariant_physics_state):
        """Apply measurement and process noise to the physics state to simulate sensor noise."""
        noise = jax.random.normal(rng, physics_state.shape) * noise_std
        physics_state = physics_state + noise
        # Add noise to odom states.
        yaw = invariant_physics_state[0] + noise[2] * self.dt
        drive_angle = invariant_physics_state[1] + noise[5] * self.dt
        balance_angle = invariant_physics_state[2] + noise[6] * self.dt
        xy = invariant_physics_state[-2:]
        xy = xy + (
            YRP(yaw, physics_state[0], physics_state[1]) @ (noise[7:10] * self.dt)[:, None]
        ).squeeze(-1)[:-1]

        invariant_physics_state = jp.array([yaw, drive_angle, balance_angle, *xy])
        return physics_state, invariant_physics_state

    def step(self, state: State, action: jp.ndarray) -> State:
        """Advance the simulation by one control step (6 MJX physics steps).

        Applies the RL action combined with the balancing prior, runs MJX, and
        returns the updated Brax State with noise-corrupted observations.
        """
        # Get current data and observation.
        data0 = state.pipeline_state
        obs = state.obs
        track_seed = state.info['track_seed']

        # Compute torques.
        robot_state = obs[self._obs_slice_start:-4]
        action_clipped = jp.clip(action, -1, 1)
        driving_wheel_torque = -self.K_pitch @ robot_state[_PITCH_CTRL_INDICES]
        balancing_wheel_torque = -self.K_roll @ robot_state[_ROLL_CTRL_INDICES]
        LC_torque = jp.array([driving_wheel_torque, balancing_wheel_torque])
        applied_torque = jp.clip(
            self.action_scale * action_clipped + LC_torque, -self.action_lim, self.action_lim
        )

        # Step the physics.
        data = self.pipeline_step(data0, applied_torque)

        # Get the physics state.
        robot_state = qpos_qvel_to_robot_state(data.qpos, data.qvel)
        variant_physics_state = robot_state[_VARIANT_INDICES]
        body_vel = (
            Ry(-robot_state[2]) @ Rx(-robot_state[1]) @ Rz(-robot_state[0])
            @ data.qvel[0:3][:, None]
        ).flatten()
        physics_state = jp.array([*variant_physics_state, *body_vel, data.qpos[2]])
        invariant_physics_state = jp.array(
            [robot_state[0], *(data.qpos[-2:]), *(data.qpos[:2])] # yaw, drive_angle, balance_angle, x, y
        )

        # Add process noise.
        rng, process_rng = jax.random.split(state.info['rng'])
        if self.process_noise_std is not None:
            rng, _ = jax.random.split(rng)
            physics_state, invariant_physics_state = self.add_noise(
                process_rng, self.process_noise_std, physics_state, invariant_physics_state
            )

            robot_state = jp.array([
                invariant_physics_state[0], #yaw
                *physics_state[0:5], # roll, pitch, yaw_rate, roll_rate, pitch_rate
                invariant_physics_state[1], # drive_angle
                physics_state[5], # drive_angle_rate
                invariant_physics_state[2], # balance_angle
                physics_state[6], # balance_angle_rate
            ])
            qpos, qvel = robot_state_to_qpos_qvel(robot_state) 
            qpos = qpos.at[:3].set(jp.array([*invariant_physics_state[-2:], physics_state[-1]]))  # x, y, z
            qvel = qvel.at[:3].set(
                (YRP(invariant_physics_state[0], physics_state[0], physics_state[1])
                @physics_state[7:10][:, None]).squeeze(-1)
            )

            data = self.pipeline_init(qpos, qvel)

        measured_physics_state = physics_state
        measured_invariant_physics_state = invariant_physics_state
        measured_data = data
        rng, measure_rng = jax.random.split(state.info['rng'])
        if self.meas_noise_std is not None:
            measured_physics_state, measured_invariant_physics_state = self.add_noise(
                measure_rng, self.meas_noise_std, measured_physics_state, measured_invariant_physics_state
            )

            measured_robot_state = jp.array([
                measured_invariant_physics_state[0], #yaw
                *measured_physics_state[0:5], # roll, pitch, yaw_rate, roll_rate, pitch_rate
                measured_invariant_physics_state[1], # drive_angle
                measured_physics_state[5], # drive_angle_rate
                measured_invariant_physics_state[2], # balance_angle
                measured_physics_state[6], # balance_angle_rate
            ])
            measured_qpos, measured_qvel = robot_state_to_qpos_qvel(measured_robot_state) 
            measured_qpos = measured_qpos.at[:3].set(jp.array([*measured_invariant_physics_state[-2:], measured_physics_state[-1]]))  # x, y, z
            measured_qvel = measured_qvel.at[:3].set(
                (YRP(measured_invariant_physics_state[0], measured_physics_state[0], measured_physics_state[1])
                @ measured_physics_state[7:10][:, None]).squeeze(-1)
            )

            measured_data = self.pipeline_init(measured_qpos, measured_qvel)
        obs = self._get_obs(measured_data, action_clipped, track_seed)

        info = state.info
        updated_act_history = self.shift_action(info['act_history'], applied_torque)
        info['act_history'] = updated_act_history
        updated_phys_history = self.shift_phys(info['phys_state_history'], measured_physics_state)
        info['phys_state_history'] = updated_phys_history
        info['applied_torque'] = applied_torque
        info['physics_state'] = measured_physics_state
        # Invariant physics state is the robot state without the velocity.
        info['invariant_physics_state'] = jp.array(
            [measured_invariant_physics_state[0], *(measured_data.qpos[-2:]), *(measured_data.qpos[:2])]
        )
        info['rng'] = rng

        state = state.replace(pipeline_state=data, obs=obs, info=info)

        reward, done, reward_metrics = self._get_rew(state, action_clipped)
        info['reward_metrics'] = reward_metrics

        return state.replace(reward=reward, done=done, info=info)

    # ============================================================ Infoprop hooks
    def preprocess(self, state: State, action: jp.ndarray):
        """Build the NN inputs and the applied torque from the State + RL action.

        Returns ``(nn_input, curr_model_state, curr_context, applied_torque,
        action_clipped)``. The applied torque is the RL action scaled and added to the
        linear balancing prior (the Wheelbot's control prior); ``action_clipped`` is the
        RL action used for observation/reward.
        """
        nn_input = jp.concatenate(
            [state.info['phys_state_history'], state.info['act_history']], axis=-1)
        robot_state = state.obs[self._obs_slice_start:]
        action_clipped = jp.clip(action, -1, 1)
        driving_wheel_torque = -self.K_pitch @ robot_state[_PITCH_CTRL_INDICES]
        balancing_wheel_torque = -self.K_roll @ robot_state[_ROLL_CTRL_INDICES]
        LC_torque = jp.array([driving_wheel_torque, balancing_wheel_torque])
        applied_torque = jp.clip(
            self.action_scale * action_clipped + LC_torque, -self.action_lim, self.action_lim)
        return (nn_input, state.info['physics_state'], state.info['invariant_physics_state'],
                applied_torque, action_clipped)

    def augment_prediction(self, member_mean, member_var, curr_model_state, curr_context):
        """Append the 5 integrated odometry dims (yaw, drive/balance angle, x, y) and
        propagate their variance. Single rollout sample: ``[E, model_state_size]`` in."""
        dt = self.dt
        means_, vars_ = member_mean, member_var
        curr_physics_state, curr_odom_state = curr_model_state, curr_context
        R = jax.vmap(YRP)(curr_odom_state[None, 0], curr_physics_state[None, 0], curr_physics_state[None, 1])
        means__1 = curr_odom_state[None, 0:3] + dt * (curr_physics_state[None, _EULER_RATE_IDX] + means_[:, _EULER_RATE_IDX]) / 2
        # x,y integration only; z is directly predicted by the model at physics_state index 10
        means__2 = curr_odom_state[None, 3:5] + dt * (R @ curr_physics_state[None, _BODY_VEL_IDX, None] + R @ means_[:, _BODY_VEL_IDX, None]).squeeze(axis=-1)[:, :2] / 2
        means__ = jp.concatenate((means__1, means__2), axis=-1)
        full_mean = jp.concatenate((means_, means__), axis=-1)

        vars__1 = (dt / 2) ** 2 * vars_[:, _EULER_RATE_IDX]
        vars__2 = (dt / 2) ** 2 * (jp.square(R) @ vars_[:, _BODY_VEL_IDX, None]).squeeze(-1)[:, :2]
        vars__ = jp.concatenate((vars__1, vars__2), axis=-1)
        full_var = jp.concatenate((vars_, vars__), axis=-1)
        return full_mean, full_var

    def _get_obs_from_states(self, physics_state, invariant_physics_state, track_seed):
        """Construct observation directly from variant and invariant state (fast rollout)."""
        robot_state = jp.array([
            invariant_physics_state[0],
            *physics_state[:5],
            invariant_physics_state[1],
            physics_state[5],
            invariant_physics_state[2],
            physics_state[6],
        ])
        trajectory = get_trajectory_by_seed(track_seed, self.lookahead)
        traj_state = trajectory.get_state(
            invariant_physics_state[3:5], invariant_physics_state[0], self.sin_cos_encoding)
        robot_state_masked = robot_state * self.robot_state_mask
        extras = jp.array([physics_state[-1], *physics_state[7:10]])
        return jp.array([*traj_state, *robot_state_masked, *extras])

    def _get_model_data_and_obs(self, state, physics_state, invariant_physics_state,
                                action, track_seed, build_pipeline_state):
        """Build the next pipeline state (optional) and observation for a model rollout step."""
        if not build_pipeline_state:
            return None, self._get_obs_from_states(physics_state, invariant_physics_state, track_seed)
        robot_state = jp.array([
            invariant_physics_state[0],
            *physics_state[:5],
            invariant_physics_state[1],
            physics_state[5],
            invariant_physics_state[2],
            physics_state[6],
        ])
        qpos, qvel = robot_state_to_qpos_qvel(robot_state)
        qpos = qpos.at[0:3].set(jp.concatenate([invariant_physics_state[-2:], physics_state[-1:]]))
        qvel = qvel.at[0:3].set(
            (YRP(invariant_physics_state[0], physics_state[0], physics_state[1])
             @ physics_state[7:10][:, None]).squeeze(-1))
        data = self.pipeline_init(qpos, qvel)
        return data, self._get_obs(data, action, track_seed)

    def postprocess(self, state, applied_action, next_model_state, next_context,
                    processed_action):
        """Rebuild the MJX-shaped State, env-owned `info`, and reward from a prediction.

        The framework-owned rng + entropy accumulation are already set on ``state``.
        Skips building the MJX pipeline_state when ``self.fast_model_rollout``.
        """
        track_seed = state.info['track_seed']
        data, obs = self._get_model_data_and_obs(
            state, next_model_state, next_context, processed_action, track_seed,
            build_pipeline_state=not self.fast_model_rollout)
        info = state.info
        info['applied_torque'] = applied_action
        info['physics_state'] = next_model_state
        info['invariant_physics_state'] = next_context
        info['phys_state_history'] = self.shift_phys(info['phys_state_history'], next_model_state)
        info['act_history'] = self.shift_action(info['act_history'], applied_action)
        state = state.replace(info=info, pipeline_state=data, obs=obs)

        reward, done, reward_metrics = self._get_rew(state, processed_action)
        info['reward_metrics'] = reward_metrics
        return state.replace(reward=reward, done=done, info=info)

    def reset_from_buffer(self, rng, init_transition):
        """Reset a model rollout from a sampled real-data physics transition."""
        init_history = init_transition.observation
        track_seed = init_transition.extras['state_extras']['track_seed']
        invariant = init_transition.extras['state_extras']['invariant_physics_state']
        return self.reset_with_init_robot_state(
            rng, init_history, track_seed, invariant[3:5], invariant[0],
            build_pipeline_state=not self.fast_model_rollout)

    def reset_with_init_robot_state(self, rng, init_history, track_seed, init_xy, init_angle,
                                    build_pipeline_state):
        """Reset using a provided model-state history and global position.

        Used during episodic resampling: initial states are drawn from the real-data
        replay buffer so that model rollouts branch from states grounded in real experience.
        """
        rng, track_key = jax.random.split(rng)
        track_seed = jax.random.randint(track_key, shape=(), minval=0, maxval=200)
        trajectory = get_trajectory_by_seed(track_seed)
        rng, pos_key = jax.random.split(rng)
        init_xy, init_angle = trajectory.get_rand_init_pos(pos_key)
        rng, xy_key = jax.random.split(rng)
        d = jp.clip(jax.random.normal(xy_key, shape=()) * self.resample_init_xy_std, -track_width / 2, track_width / 2)
        perp_dir = jp.array([-jp.sin(init_angle), jp.cos(init_angle)])
        init_xy = init_xy + d * perp_dir
        rng, angle_key = jax.random.split(rng)
        offset_angle = jp.clip(jax.random.normal(angle_key, shape=init_angle.shape) * self.resample_init_angle_std, -jp.pi, jp.pi)
        init_angle = init_angle + offset_angle

        ms = self.model_state_size
        init_physics_state_history = init_history[:ms * self.obs_history]
        init_action_history = init_history[ms * self.obs_history:]
        init_physics_state = init_physics_state_history[-ms:]
        init_robot_state = jp.array([init_angle, *(init_physics_state[:5]), 0, init_physics_state[5], 0, init_physics_state[6]])

        qpos, qvel = robot_state_to_qpos_qvel(init_robot_state)
        qpos = qpos.at[:2].set(init_xy)
        qpos = qpos.at[2].set(init_physics_state[-1])
        qvel = qvel.at[:3].set((YRP(init_angle, init_physics_state[0], init_physics_state[1]) @ init_physics_state[7:10][:, None]).squeeze(-1))

        data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(data, jp.zeros(self.action_size), track_seed)
        reward, done = jp.zeros(2)
        reward_metrics = {'cross_track_rew': 0.0, 'cross_angle_rew': 0.0, 'driving_reward': 0.0, 'crash_penalty': 0.0}
        info = {
            'track_seed': track_seed,
            'applied_torque': jp.zeros(self.action_size),
            'physics_state': init_physics_state,
            'accumulated_conditional_entropy': jp.zeros((self.full_state_size,)),
            'current_conditional_entropy': jp.zeros((self.full_state_size,)),
            'reward_metrics': reward_metrics,
            'invariant_physics_state': jp.array([init_angle, 0, 0, *(data.qpos[:2])]),
            'act_history': init_action_history,
            'phys_state_history': init_physics_state_history,
        }
        pipeline_state = data if build_pipeline_state else None
        return State(pipeline_state, obs, reward, done, {}, info)

    def reset_with_init_robot_state_eval(self, rng, init_history, track_seed, init_xy, init_angle):
        """Deterministic reset for evaluation (always builds the pipeline_state for rendering)."""
        ms = self.model_state_size
        init_physics_state_history = init_history[:ms * self.obs_history]
        init_action_history = init_history[ms * self.obs_history:]
        init_physics_state = init_physics_state_history[-ms:]
        init_robot_state = jp.array([init_angle, *(init_physics_state[:5]), 0, init_physics_state[5], 0, init_physics_state[6]])

        qpos, qvel = robot_state_to_qpos_qvel(init_robot_state)
        qpos = qpos.at[:2].set(init_xy)
        qpos = qpos.at[2].set(init_physics_state[-1])
        qvel = qvel.at[:3].set((YRP(init_angle, init_physics_state[0], init_physics_state[1]) @ init_physics_state[7:10][:, None]).squeeze(-1))

        data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(data, jp.zeros(self.action_size), track_seed)
        reward, done = jp.zeros(2)
        reward_metrics = {'cross_track_rew': 0.0, 'cross_angle_rew': 0.0, 'driving_reward': 0.0, 'crash_penalty': 0.0}
        info = {
            'track_seed': track_seed,
            'applied_torque': jp.zeros(self.action_size),
            'physics_state': init_physics_state,
            'accumulated_conditional_entropy': jp.zeros((self.full_state_size,)),
            'current_conditional_entropy': jp.zeros((self.full_state_size,)),
            'reward_metrics': reward_metrics,
            'invariant_physics_state': jp.array([init_angle, 0, 0, *(data.qpos[:2])]),
            'act_history': init_action_history,
            'phys_state_history': init_physics_state_history,
        }
        return State(data, obs, reward, done, {}, info)
