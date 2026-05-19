"""
Wheelbot model-based environment for InfoProp Dyna rollouts.

This environment replaces MuJoCo physics with a learned probabilistic ensemble model.
Each step applies the InfoProp Dyna mechanism:
  1. Query all ensemble members for (mean, logvar) predictions.
  2. Compute precision-weighted fused posterior and epistemic variance.
  3. Apply a Kalman update to obtain the filtered next-state estimate.
  4. Compute per-step conditional entropy as the information-loss signal.
  5. Terminate rollouts when per-step or accumulated entropy exceeds thresholds.

Used exclusively during model-based training. Ground-truth MuJoCo evaluation
uses wheelbot_brax_mjx.py instead.
"""

import xml.etree.ElementTree as ET
from typing import Optional

import jax
from jax import lax, numpy as jp
from jax.scipy.spatial.transform import Rotation
import mujoco
from mujoco import mjx
from etils import epath
from brax import envs
from brax.io import mjcf
from brax.envs.base import PipelineEnv, State
import mediapy as media
from omegaconf import dictconfig, OmegaConf

from .utils import compute_line_element
from ..track.generator import create_track, load_track_by_seed
from .trajectory import Trajectory, pad_line_segments_to_size, points_to_line_segemnt
from wheelbot_sim_python.algorithms.util.model_learning.model_trainer import ModelTrainer


def Rx(theta):
    """
    Rotation matrix around x-axis.
    """
    return jp.array([[1, 0, 0],
                     [0, jp.cos(theta), -jp.sin(theta)],
                     [0, jp.sin(theta), jp.cos(theta)]])

def Ry(theta):
    """
    Rotation matrix around y-axis.
    """
    return jp.array([[jp.cos(theta), 0, jp.sin(theta)],
                     [0, 1, 0],
                     [-jp.sin(theta), 0, jp.cos(theta)]])

def Rz(theta):
    """
    Rotation matrix around z-axis.
    """
    return jp.array([[jp.cos(theta), -jp.sin(theta), 0],
                     [jp.sin(theta), jp.cos(theta), 0],
                     [0, 0, 1]])

def skew_symmetric(w):
    """
    Skew symmetric matrix.
    """
    v = w.flatten()
    return jp.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def mat2Quat(mat):
    """
    Converts a rotation matrix to a quaternion.
    """
    r = Rotation.from_matrix(mat)
    q = r.as_quat(scalar_first=True)
    return q

def quat2Mat(q):
    """
    Converts a quaternion to a rotation matrix.
    """
    r = Rotation.from_quat(q[jp.array([1, 2, 3, 0])])
    return r.as_matrix()

def jacobian_w2euler(q1, q2):
    return jp.array([[jp.cos(q2), 0, jp.sin(q2)], 
                     [jp.sin(q2) * jp.tan(q1), 1, -jp.cos(q2) * jp.tan(q1)], 
                     [-jp.sin(q2) / jp.cos(q1), 0, jp.cos(q2) / jp.cos(q1)]])

del_jacobian_w2euler_del_q1 = (jax.jacobian(jacobian_w2euler, 0))
del_jacobian_w2euler_del_q2 = (jax.jacobian(jacobian_w2euler, 1))

def jacobian_dot_w2euler(q1, q2, dq1, dq2):
  jac_dot = del_jacobian_w2euler_del_q1(q1, q2) * jp.array([dq1]) + del_jacobian_w2euler_del_q2(q1, q2) * jp.array([dq2])
  return jac_dot

def robot_state_to_qpos_qvel(state, O_r_OC=None):
    C_R_W = Ry(state[2])
    O_R_C = Rz(state[0]) @ Rx(state[1])
    C_r_CW = jp.array([[0, 0, 0.032]]).T
    W_r_WB = jp.array([[0, 0, 0.0325]]).T
    O_r_dot_OC = 0.032*(state[5]+state[7])*jp.array([[jp.cos(state[0]), jp.sin(state[0]), 0]]).T
    O_w_C = jp.array([[0, 0, state[3]]]).T + Rz(state[0]) @ jp.array([[state[4], 0, 0]]).T
    C_w_W = jp.array([[0, state[5], 0]]).T
    S_O_w_C = skew_symmetric(O_w_C)
    S_C_w_W = skew_symmetric(C_w_W)
    if O_r_OC is None:
        O_r_OC = jp.zeros((3, 1))
    O_r_OB = O_r_OC + O_R_C @ C_r_CW + O_R_C @ C_R_W @ W_r_WB
    R = O_R_C @ C_R_W
    quat=mat2Quat(R)
    qpos = jp.array((*(O_r_OB.flatten()), *(quat.flatten()),state[6], state[8]))
    O_w_B = R.T @ (O_w_C + O_R_C @ C_R_W @ C_w_W)
    O_r_dot_OB = (O_r_dot_OC + S_O_w_C @ O_R_C @ (C_r_CW + C_R_W @ W_r_WB) + O_R_C @ S_C_w_W @ C_R_W @ W_r_WB)
    qvel = jp.array((*(O_r_dot_OB.flatten()), *(O_w_B.flatten()), state[7], state[9]))
    return qpos, qvel#, O_r_dot_OC.flatten()

def qpos_qvel_to_robot_state(qpos, qvel):
    quat = qpos[3:7]
    R = quat2Mat(quat)
    R = R.reshape((3,3))
    phi = jp.arctan2(-R[0,1], R[1,1])
    theta = jp.arcsin(R[2,1])
    psi = jp.arctan2(-R[2,0], R[2,2])

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
   traj_flattened = trajectories_flattened[track_seed]
   size = trajectory_lengths[track_seed]
   return Trajectory(traj_flattened, size, lookahead)

def YRP(yaw, roll, pitch):
   return Rz(yaw) @ Rx(roll) @ Ry(pitch)

def YRP_inv(yaw, roll, pitch):
   return Ry(-pitch) @ Rx(-roll) @ Rz(-yaw)

def get_projected_velocity(State):
   physics_state = State.info['physics_state']
   long_vel_body_frame = (Ry(physics_state[0]) @ Rx(physics_state[1]) @ physics_state[-4:-1][:, None])[0,0]
   proj_vel_body_frame = long_vel_body_frame * jp.cos(-State.obs[1])
   return proj_vel_body_frame



# Static index arrays for fancy indexing — use arrays to avoid deprecated list indexing
_VARIANT_INDICES = jp.array([1, 2, 3, 4, 5, 7, 9])
_PITCH_CTRL_INDICES = jp.array([2, 5, 6, 7])
_ROLL_CTRL_INDICES = jp.array([1, 4, 8, 9])

WHEELBOT_ROOT_PATH = epath.Path(epath.resource_path('wheelbot_sim_python'))

#create trajectories statically to be used in reset method.
tracks = [load_track_by_seed(i) for i in range(200)]
track_width = tracks[0]['width']
centerlines = [track['centerline'] for track in tracks]
trajectories = [points_to_line_segemnt(centerline) for centerline in centerlines]
trajectory_lengths = jp.array([t.shape[0] for t in trajectories])
max_length = max(trajectory_lengths)
trajectories_flattened = jp.array([pad_line_segments_to_size(t, max_length) for t in trajectories])

class Wheelbot(PipelineEnv):
  """Brax environment backed by the learned InfoProp ensemble dynamics model.

  Extends the MJX environment interface so it can be used as a drop-in replacement
  during model-based rollouts. Carries InfoProp-specific state: Kalman gains,
  conditional variances, and accumulated entropy.

  Key attributes (beyond base Wheelbot):
      model_state: Flax TrainState for the probabilistic ensemble.
      per_step_cutoff: Lambda_1 — maximum allowed per-step information loss.
      accumulated_cutoff: Lambda_2 — maximum allowed cumulative information loss.
  """

  def __init__(
      self,
      cfg: dictconfig.DictConfig = dictconfig.DictConfig({}),
      visualize: bool = False,
      track_seed: Optional[int] = None,
      min_log_var: float = -4,
      max_log_var: float = -2,
      fast_model_rollout: bool = False,
      **kwargs,
  ):
    mjcf_path = (WHEELBOT_ROOT_PATH / 'mjcf').as_posix()
    xml_path = mjcf_path + '/wheelbot_alpha.xml'
    xml_str = epath.Path(xml_path).read_text()

    root = ET.fromstring(xml_str)
    #set the mesh file paths
    asset = root.find('asset')
    for mesh in asset.findall('mesh'):
        if mesh.attrib.get('name') == 'body' or  mesh.attrib.get('name') == 'wheel':
            relative_file = mesh.get('file')
            mesh.set('file', mjcf_path + '/' + relative_file)
    
    if track_seed != None:
        self.track_seed = track_seed
        track = tracks[self.track_seed]

        if visualize:
            inner_cones = track['inner_cones']
            outer_cones = track['outer_cones']
    
            track_body = root.find(".//body[@name='track']")
            outer_cone_line_elements = [compute_line_element(outer_cones[i], outer_cones[i+1]) for i in range(len(outer_cones) - 1)]
            inner_cone_line_elements = [compute_line_element(inner_cones[i], inner_cones[i+1]) for i in range(len(inner_cones) - 1)]
            
            track_line_elements = outer_cone_line_elements + inner_cone_line_elements

            for elem in track_line_elements:
                track_body.append(elem)

    xml_str = ET.tostring(root, encoding='unicode')

    #create the mujoco model from the xml file
    mj_model = mujoco.MjModel.from_xml_string(xml_str)
    # set the solver 
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    # set the number of iterations for the solver
    mj_model.opt.iterations = 6
    # set the number of line search iterations inside each solver iteration
    mj_model.opt.ls_iterations = 6
    # timestep of the simulation
    mj_model.opt.timestep = 0.001

    # load the model into the mjx system
    sys = mjcf.load_model(mj_model)

    kwargs['n_frames'] = kwargs.get(
        'n_frames', cfg.get('control_interval', 6))
    kwargs['backend'] = 'mjx'

    super().__init__(sys, **kwargs)
    self.K_roll = jp.array(cfg.get('K_roll', [-1.3e0, -1.6e-1, -0.8e-04, -4e-04]))
    self.K_pitch = jp.array(cfg.get('K_pitch', [-400e-3, -40e-3, -4e-3, -3e-3]))
    self.max_state = jp.array([*(cfg.get('max_state', [2 * jp.pi, jp.pi / 4, jp.pi / 4, 40, 40, 40, jp.inf, 800, jp.inf, 800]))])
    self.action_lim = cfg.get('action_lim', 0.5)
    #zeros the robot_state values that are not used in trajectory driving
    self.robot_state_mask = jp.array([0, 1, 1, 1, 1, 1, 0, 1, 0, 1]) 
    #reward parameters
    self.action_scale = cfg.get('action_scale', 0.1)
    self.ca_weight = cfg.get('ca_weight', 1.0)
    self.ct_weight = cfg.get('ct_weight', 1.0)
    self.driving_weight = cfg.get('driving_weight', 1.0)
    self.crash_penalty = cfg.get('crash_penalty', -1000)
    self.rew_scale = cfg.get('rew_scale', 1e-3)
    #randomize env parameters
    self.init_xy_std = cfg.get('init_xy_std', track_width/3)
    self.init_angle_std = cfg.get('init_angle_std', jp.pi/3)
    self.min_log_var = min_log_var
    self.max_log_var = max_log_var
    self.fast_model_rollout = fast_model_rollout
    # History parameters for model learning
    self.obs_history = cfg.get('obs_history', 1)
    self.act_history = cfg.get('act_history', 0)
    self.lookahead = cfg.get('lookahead', 10)
    self.sin_cos_encoding = cfg.get('sin_cos_encoding', True)
    _enc = self.sin_cos_encoding
    self._obs_slice_start = (3 if _enc else 2) * self.lookahead + (3 if _enc else 2)

  def _get_obs(self, data: mjx.Data, action: jp.ndarray, track_seed: int) -> jp.ndarray:
    """Gets the observation from the environment."""
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

  def _get_obs_from_states(
      self,
      physics_state: jp.ndarray,
      invariant_physics_state: jp.ndarray,
      track_seed: int,
  ) -> jp.ndarray:
    """Construct observation directly from variant and invariant state."""
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
        invariant_physics_state[3:5], invariant_physics_state[0], self.sin_cos_encoding
    )
    robot_state_masked = robot_state * self.robot_state_mask
    extras = jp.array([physics_state[-1], *physics_state[7:10]])
    return jp.array([*traj_state, *robot_state_masked, *extras])

  def _get_model_data_and_obs(
      self,
      state: State,
      physics_state: jp.ndarray,
      invariant_physics_state: jp.ndarray,
      action: jp.ndarray,
      track_seed: int,
  ):
    """Build the next pipeline state and observation for render or fast rollout."""
    if self.fast_model_rollout:
      return None, self._get_obs_from_states(
          physics_state, invariant_physics_state, track_seed
      )

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
        @ physics_state[7:10][:, None]).squeeze(-1)
    )
    data = self.pipeline_init(qpos, qvel)
    return data, self._get_obs(data, action, track_seed)

  def _get_rew(self, State, action):
    state = State.obs
    traj_state = state[:self._obs_slice_start]
    cross_track_error = traj_state[0]
    cross_angle_error = traj_state[1] if not self.sin_cos_encoding else jp.arctan2(traj_state[1], traj_state[2])
    cross_track_rew = self.ct_weight * (track_width/2 - jp.abs(cross_track_error))/(track_width/2)

    cross_angle_rew = self.ca_weight * (jp.pi/2 - jp.abs(cross_angle_error))/(jp.pi/2)

    robot_state = state[self._obs_slice_start:-4]
    done = jp.where((jp.abs(robot_state) > self.max_state).any()
                    , 1.0, 0.0)
    track_violated = jp.where(jp.abs(cross_track_error) > track_width / 2, 1.0, 0.0)
    
    done = jp.where(track_violated, 1.0, done)
    projected_velocity = get_projected_velocity(State)
    driving_reward = self.driving_weight * projected_velocity
    crash_penalty = jp.float32(self.crash_penalty)

    reward = self.rew_scale * (
      (1-done) * cross_track_rew +
      (1-done) * cross_angle_rew +
      (1-done) * driving_reward +
      done * crash_penalty
    )
    reward_metrics = {
       'cross_track_rew': cross_track_rew,
       'cross_angle_rew': cross_angle_rew,
       'driving_reward': driving_reward,
       'crash_penalty': crash_penalty
    }
    return reward, done, reward_metrics

  def put_in_NN_params_and_rng(
      self, model, model_obs_mean, model_obs_std, next_state_delta_mean,
      next_state_delta_std, per_step_cutoff, accumulated_cutoff, binning_entropy, rng, State
  ):
      """Inject model parameters and entropy cutoffs into the environment info dict.

      Must be called before rollouts to supply the ensemble weights and termination thresholds.
      """
      info = State.info
      info['model'] = model
      info['model_obs_mean'] = model_obs_mean
      info['model_obs_std'] = model_obs_std
      info['next_state_delta_mean'] = next_state_delta_mean
      info['next_state_delta_std'] = next_state_delta_std
      info['per_step_cutoff'] = per_step_cutoff
      info['accumulated_cutoff'] = accumulated_cutoff
      info['binning_entropy'] = binning_entropy
      info['rng'] = rng
      return State.replace(info=info)
     
  def reset(self, rng: jp.ndarray) -> State:
    """Reset to a random track and a random position along its centreline.

    Samples a track index uniformly from the training split (tracks 0–159),
    then places the robot at a random position with a short MJX warm-up.
    """
    #get random track
    rng, track_key = jax.random.split(rng)
    #use 80% of tracks for training
    track_seed = jax.random.randint(track_key, shape=(), minval=0, maxval=160)
    trajectory = get_trajectory_by_seed(track_seed)
    state = trajectory.get_state(jp.array([1,1]), 0)
    #get position on a random point on the trajectory
    rng, pos_key = jax.random.split(rng)
    init_xy, init_angle = trajectory.get_rand_init_pos(pos_key)
    #get random initial location
    rng, xy_key = jax.random.split(rng)
    offset_xy = jp.clip(jax.random.normal(xy_key, shape=init_xy.shape) * self.init_xy_std, -0.9*track_width/2, 0.9*track_width/2)
    init_xy = init_xy + offset_xy
    #get random initial angle
    rng, angle_key = jax.random.split(rng)
    offset_angle = jp.clip(jax.random.normal(angle_key, shape=init_angle.shape) * self.init_angle_std, -jp.pi/2, jp.pi/2)
    init_angle = init_angle + offset_angle
    #create random state
    init_robot_state = jp.array([init_angle, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    qpos, qvel = robot_state_to_qpos_qvel(init_robot_state)
    qpos = qpos.at[:2].set(init_xy)
    
    data = self.pipeline_init(qpos, qvel)
    action = jp.zeros(2)
    obs = self._get_obs(data, action, track_seed)
    
    # Initialize history arrays with zeros
    act_history_array = jp.zeros(2 * self.act_history) if self.act_history > 0 else jp.array([])
    phys_state_history_array = jp.zeros(11 * self.obs_history) if self.obs_history > 0 else jp.array([])
    
    robot_state = qpos_qvel_to_robot_state(data.qpos, data.qvel)
    variant_physics_state = robot_state[_VARIANT_INDICES]
    body_vel = (Ry(-robot_state[2]) @ Rx(-robot_state[1]) @ Rz(-robot_state[0]) @ data.qvel[0:3][:, None]).flatten()

    metrics = {
       
    }
    
    info = {
        'track_seed': track_seed,
        'applied_torque': action,
        'physics_state': jp.concatenate([variant_physics_state, body_vel, data.qpos[2:3]]),
        'accumulated_conditional_entropy': jp.zeros((16,)),
        'current_conditional_entropy': jp.zeros((16,)),
        'invariant_physics_state': jp.array([robot_state[0], *(data.qpos[-2:]), *(data.qpos[:2])]),
        'act_history': act_history_array,
        'phys_state_history': phys_state_history_array,

    }

    state = State(data, obs, 0.0, 0.0, metrics, info)
    warmup_steps = max(self.act_history, self.obs_history)
    def func(carry, _):
        state, action = carry
        next_state = self.step(state, action)
        return (next_state, action), None

    (state, action), _ = jax.lax.scan(func, (state, action), None, warmup_steps)

    return state

  def reset_with_init_robot_state(
      self, rng: jp.ndarray, init_history, track_seed, init_xy, init_angle
  ) -> State:
    """Reset using a provided physics-state history and global position.

    Used during episodic resampling: initial states are drawn from the real-data
    replay buffer so that model rollouts branch from states grounded in real experience.
    """
    rng, track_key = jax.random.split(rng)
    track_seed = jax.random.randint(track_key, shape=(), minval=0, maxval=200)
    trajectory = get_trajectory_by_seed(track_seed)
    # # state = trajectory.get_state(jp.array([1,1]), 0)
    rng, pos_key = jax.random.split(rng)
    init_xy, init_angle = trajectory.get_rand_init_pos(pos_key)
    # get random initial location
    rng, xy_key = jax.random.split(rng)
    d = jp.clip(jax.random.normal(xy_key, shape=()) * self.init_xy_std, -track_width/2, track_width/2)
    perp_dir = jp.array([-jp.sin(init_angle), jp.cos(init_angle)])
    offset_xy = d * perp_dir
    init_xy = init_xy + offset_xy
    #get random initial angle
    rng, angle_key = jax.random.split(rng)
    offset_angle = jp.clip(jax.random.normal(angle_key, shape=init_angle.shape) * self.init_angle_std, -jp.pi, jp.pi)
    init_angle = init_angle + offset_angle
    #create random state
    init_physics_state_history = init_history[:11*self.obs_history]
    init_action_history = init_history[11*self.obs_history:]
    init_physics_state = init_physics_state_history[-11:]
    init_robot_state = jp.array([init_angle, *(init_physics_state[:5]), 0, init_physics_state[5], 0, init_physics_state[6]])

    qpos, qvel = robot_state_to_qpos_qvel(init_robot_state)
    qpos = qpos.at[:2].set(init_xy)
    qpos = qpos.at[2].set(init_physics_state[-1])
    qvel = qvel.at[:3].set((YRP(init_angle, init_physics_state[0], init_physics_state[1]) @ init_physics_state[7:10][:, None]).squeeze(-1))

    data = self.pipeline_init(qpos, qvel)
    action = jp.zeros(2)
    obs = self._get_obs(data, action, track_seed)

    reward, done = jp.zeros(2)
    metrics = {

    }
    reward_metrics = {
       'cross_track_rew': 0.0,
       'cross_angle_rew': 0.0,
       'driving_reward': 0.0,
       'crash_penalty': 0.0
    }

    info = {
        'track_seed': track_seed,
        'applied_torque': jp.zeros(2),
        'physics_state': init_physics_state,
        'accumulated_conditional_entropy': jp.zeros((16,)),
        'current_conditional_entropy': jp.zeros((16,)),
        'reward_metrics': reward_metrics,
        'invariant_physics_state': jp.array([init_angle, 0, 0, *(data.qpos[:2])]),
        'act_history': init_action_history,
        'phys_state_history': init_physics_state_history,
    }
    pipeline_state = None if self.fast_model_rollout else data
    return State(pipeline_state, obs, reward, done, metrics, info)

  def reset_with_init_robot_state_eval(
      self, rng: jp.ndarray, init_history, track_seed, init_xy, init_angle
  ) -> State:
    """Deterministic reset for evaluation — places the robot at track index 0."""
    trajectory = get_trajectory_by_seed(track_seed)
    state = trajectory.get_state(jp.array([1,1]), 0)

    #create random state
    init_physics_state_history = init_history[:11*self.obs_history]
    init_action_history = init_history[11*self.obs_history:]
    init_physics_state = init_physics_state_history[-11:]
    init_robot_state = jp.array([init_angle, *(init_physics_state[:5]), 0, init_physics_state[5], 0, init_physics_state[6]])

    qpos, qvel = robot_state_to_qpos_qvel(init_robot_state)
    qpos = qpos.at[:2].set(init_xy)
    qpos = qpos.at[2].set(init_physics_state[-1])
    qvel = qvel.at[:3].set((YRP(init_angle, init_physics_state[0], init_physics_state[1]) @ init_physics_state[7:10][:, None]).squeeze(-1))

    data = self.pipeline_init(qpos, qvel)
    action = jp.zeros(2)
    obs = self._get_obs(data, action, track_seed)

    reward, done = jp.zeros(2)
    metrics = {

    }
    reward_metrics = {
       'cross_track_rew': 0.0,
       'cross_angle_rew': 0.0,
       'driving_reward': 0.0,
       'crash_penalty': 0.0
    }

    info = {
        'track_seed': track_seed,
        'applied_torque': jp.zeros(2),
        'physics_state': init_physics_state,
        'accumulated_conditional_entropy': jp.zeros((16,)),
        'current_conditional_entropy': jp.zeros((16,)),
        'reward_metrics': reward_metrics,
        'invariant_physics_state': jp.array([init_angle, 0, 0, *(data.qpos[:2])]),
        'act_history': init_action_history,
        'phys_state_history': init_physics_state_history,
    }
    return State(data, obs, reward, done, metrics, info)
  
  
  def first_half_of_step(self, state: State, action: jp.ndarray) -> State:
    """Compute the total torque applied to the robot: RL action + balancing prior."""
    obs = state.obs
    robot_state = obs[self._obs_slice_start:]
    action_clipped = jp.clip(action, -1, 1)
    driving_wheel_torque = -self.K_pitch @ robot_state[_PITCH_CTRL_INDICES]
    balancing_wheel_torque = -self.K_roll @ robot_state[_ROLL_CTRL_INDICES]

    LC_torque = jp.array([driving_wheel_torque, balancing_wheel_torque])
    applied_torque = jp.clip(self.action_scale * action_clipped + LC_torque, -self.action_lim, self.action_lim)

    return applied_torque, action_clipped

  def batched_model_step(
      self, state, applied_torque, model, obs_mean, obs_std,
      next_state_delta_mean, next_state_delta_std, binning_entropy
  ):
    """Run one Infoprop Dyna step across the full ensemble.

    Queries all E ensemble members, computes precision-weighted fused posterior,
    epistemic variance, Kalman gain, and per-step conditional entropy.

    Output uncertainty quantities:
      - fused_var:    precision-weighted variance (inverse-variance pooling)
      - fused_mean:   precision-weighted mean
      - epist_var:    epistemic variance — disagreement between ensemble means
      - kalman_gain:  K = Sigma_GT / (Sigma_GT + Sigma_epist)
      - conditional_var: (1 - K) * Sigma_GT, posterior variance after Kalman update
      - conditional_entropy: H(s_tilde), per-step information loss

    Returns the fused next-state mean and the full uncertainty dict.
    """
    curr_physics_state = state.info['physics_state']
    curr_odom_state = state.info['invariant_physics_state']
    curr_rng, rng = jax.vmap(jax.random.split, out_axes=1)(state.info['rng'])
    physics_state_history = state.info['phys_state_history']
    act_history = state.info['act_history']

    model_input = jp.concatenate([physics_state_history, act_history], axis=-1)

    means_, logvars_ = model.apply_fn(
            {"params": model.params}, model_input, applied_torque, obs_mean, obs_std
        )
    # transform to output space
    VARS = jp.exp(logvars_)
    vars_ = VARS * (next_state_delta_std + 1e-6) ** 2 * 0.006 ** 2
    R = jax.vmap(YRP)(curr_odom_state[:,0], curr_physics_state[:,0], curr_physics_state[:,1])[:,None, :,:]
    vars__1 = (0.006/2) ** 2 * vars_[:,:,jp.array([2,5,6])]
    # x,y odom variance: integrate all body_vel components through rotation, take first 2 world rows
    vars__2 = (0.006/2) ** 2 * (jp.square(R) @ vars_[:,:,jp.array([7,8,9]), None]).squeeze(-1)[:,:,:2]
    vars__ = jp.concatenate((vars__1, vars__2), axis=-1)
    vars = jp.concatenate((vars_, vars__), axis=-1)
    means_ = (means_ * (next_state_delta_std + 1e-6) + next_state_delta_mean) * 0.006 + curr_physics_state[:, None, :]
    means__1 = curr_odom_state[:,None, 0:3] + 0.006 * (curr_physics_state[:,jp.array([2,5,6])][:,None,:] + means_[:,:,jp.array([2,5,6])]) / 2
    # x,y integration only; z is now directly predicted by the model at physics_state index 10
    means__2 = curr_odom_state[:,None, 3:5] + 0.006 * (R @ curr_physics_state[:,jp.array([7,8,9])][:,None,:, None] + R @ means_[:,:,jp.array([7,8,9])][:,:,:,None]).squeeze(-1)[:,:,:2] / 2
    means__ = jp.concatenate((means__1, means__2), axis=-1)
    means = jp.concatenate((means_, means__), axis=-1)

    inv_vars = 1 / (vars + 1e-12)
    fused_var = 1 / jp.mean(inv_vars, axis=1)
    fused_mean = fused_var * jp.mean(means * inv_vars, axis=1)
    epist_var = jp.mean((means - fused_mean[:, None, :]) ** 2, axis=1)
    kalman_gain = jp.clip((fused_var) / (fused_var + epist_var), 0, 1)
    conditional_var = ((1 - kalman_gain) * fused_var)

    diff_entropy = 0.5 * jp.log2(2 * jp.pi * jp.e * conditional_var)
    conditional_entropy = jp.clip(diff_entropy - binning_entropy, 0, None)

    def f(rng):
       return jax.random.normal(rng, shape = (16,))
    next_full_physics_state = fused_mean + jp.sqrt(fused_var) * jax.vmap(f)(rng=curr_rng)

    next_physics_state = next_full_physics_state[:,:-5]
    next_odom_state = next_full_physics_state[:,-5:]
    return next_physics_state, rng, conditional_entropy, next_odom_state

  def second_half_of_step(
      self, state: State, applied_torque, next_physics_state,
      action_clipped, conditional_entropy, rng, next_odom_state
  ):
    """Update observation, state history, and info dict after the model prediction."""
    track_seed = state.info['track_seed']
    data, obs = self._get_model_data_and_obs(
        state, next_physics_state, next_odom_state, action_clipped, track_seed
    )

    info = state.info
    info['applied_torque'] = applied_torque
    info['physics_state'] = next_physics_state
    info['rng'] = rng
    info['accumulated_conditional_entropy'] = state.info['accumulated_conditional_entropy'] + conditional_entropy
    info['current_conditional_entropy'] = conditional_entropy
    info['invariant_physics_state'] = next_odom_state
    # Update history arrays using shift function
    updated_phys_history = self.shift_phys(info['phys_state_history'], next_physics_state)
    info['phys_state_history'] = updated_phys_history

    updated_act_history = self.shift_action(info['act_history'], applied_torque)
    info['act_history'] = updated_act_history

    state = state.replace(info=info, pipeline_state=data, obs=obs)

    reward, done, reward_metrics = self._get_rew(state, action_clipped)
    info['reward_metrics'] = reward_metrics
    state = state.replace(reward=reward, done=done, info=info)

    return state
  
  def batch_entropy_cutoff(
      self, state, curr_entropy, accumulated_entropy, per_step_cutoff, accumulated_cutoff
  ):
    """Terminate rollouts whose information loss exceeds the thresholds.

    Marks the episode as done if either:
      - per-step conditional entropy > lambda_1  (per_step_cutoff), or
      - accumulated conditional entropy > lambda_2  (accumulated_cutoff).

    Rollouts are terminated as soon as either threshold is exceeded.
    """
    done = state.done
    ones = jp.ones_like(done)
    zeros = jp.zeros_like(done)
    per_step_violation = (curr_entropy > per_step_cutoff).any(axis=-1)
    accumulated_violation = (accumulated_entropy > accumulated_cutoff).any(axis=-1)
    done = jp.where(jp.logical_or(per_step_violation, accumulated_violation), ones, done)

    info = state.info

    info['info_cutoff'] = jp.where(
        jp.logical_or(per_step_violation, accumulated_violation), ones - state.done, zeros
    )

    state = state.replace(done=done, info=info)
    return state
  
  def shift_action(self, curr_act_history: jp.ndarray, to_add: jp.ndarray) -> jp.ndarray:
     if self.act_history == 0:
         return curr_act_history
     act_history = jp.concatenate([curr_act_history[2:], to_add])
     return act_history

  def shift_phys(self, curr_phys_history: jp.ndarray, to_add: jp.ndarray) -> jp.ndarray:
     if self.obs_history == 0:
         return curr_phys_history
     phys_history = jp.concatenate([curr_phys_history[11:], to_add])
     return phys_history
  
  def step(self, state: State, action: jp.ndarray) -> State:
    """Advance the environment by one step using the learned dynamics model.

    Applies the RL action through the balancing prior, runs the InfoProp ensemble step,
    checks entropy-based termination, and returns the updated Brax State.
    """
    obs = state.obs
    track_seed = state.info['track_seed']
    robot_state = obs[self._obs_slice_start:]
    action_clipped = jp.clip(action, -1, 1)
    driving_wheel_torque = -self.K_pitch @ robot_state[_PITCH_CTRL_INDICES]
    balancing_wheel_torque = -self.K_roll @ robot_state[_ROLL_CTRL_INDICES]

    LC_torque = jp.array([driving_wheel_torque, balancing_wheel_torque])
    applied_torque = jp.clip(self.action_scale * action_clipped + LC_torque, -self.action_lim, self.action_lim)

    binning_entropy = state.info['binning_entropy']
    next_physics_state, rng, conditional_entropy, next_odom_state, kalman_gain, conditional_var, fused_var, fused_mean, epist_var = self.model_step(state, applied_torque, binning_entropy)



    data, obs = self._get_model_data_and_obs(
        state, next_physics_state, next_odom_state, action_clipped, track_seed
    )

    info = state.info
    
    # Update history arrays using shift function
    updated_phys_history = self.shift_phys(info['phys_state_history'], next_physics_state)
    info['phys_state_history'] = updated_phys_history

    updated_act_history = self.shift_action(info['act_history'], applied_torque)
    info['act_history'] = updated_act_history
    
    info['applied_torque'] = applied_torque
    info['physics_state'] = next_physics_state
    info['rng'] = rng
    info['accumulated_conditional_entropy'] = state.info['accumulated_conditional_entropy'] + conditional_entropy
    info['invariant_physics_state'] = next_odom_state
    info['current_conditional_entropy'] = conditional_entropy
    info['kalman_gain'] = kalman_gain
    info['conditional_var'] = conditional_var
    info['fused_var'] = fused_var
    info['fused_mean'] = fused_mean
    info['epist_var'] = epist_var

    state = state.replace(obs=obs, info=info , pipeline_state=data)
    reward, done, reward_metrics = self._get_rew(state, action_clipped)
    info['reward_metrics'] = reward_metrics

    done = jp.where((conditional_entropy > state.info['per_step_cutoff']).any(), 1.0, done) #if the model is uncertain, then done

    done = jp.where((state.info['accumulated_conditional_entropy'] > state.info['accumulated_cutoff']).any(), 1.0, done)

    return state.replace(
        reward=reward, done=done, info=info
    )
  
  def direct_step(self, state: State, action: jp.ndarray) -> State:
    """Variant of step used directly by the SAC agent (bypasses control-law splitting)."""
    obs = state.obs
    track_seed = state.info['track_seed']
    # robot_state = obs[2 * lookahead + 2:]
    # action_clipped = jp.clip(action, -1, 1) 
    # driving_wheel_torque = -self.K_pitch @ (robot_state[tuple([[2, 5, 6, 7 ]])]) 
    # balancing_wheel_torque =  -self.K_roll @ (robot_state[tuple([[1, 4, 8, 9]])]) 
    
    # LC_torque = jp.array([driving_wheel_torque, balancing_wheel_torque])
    # applied_torque = jp.clip( self.action_scale * action_clipped + LC_torque, -self.action_lim, self.action_lim)

    binning_entropy = state.info['binning_entropy']
    next_physics_state, rng, conditional_entropy, next_odom_state, kalman_gain, conditional_var, fused_var, fused_mean, epist_var = self.model_step(state, action, binning_entropy)



    data, obs = self._get_model_data_and_obs(
        state, next_physics_state, next_odom_state, action, track_seed
    )

    info = state.info
    
    # Update history arrays using shift function
    updated_phys_history = self.shift_phys(info['phys_state_history'], next_physics_state)
    info['phys_state_history'] = updated_phys_history

    updated_act_history = self.shift_action(info['act_history'], action)
    info['act_history'] = updated_act_history
    
    info['applied_torque'] = action
    info['physics_state'] = next_physics_state
    info['rng'] = rng
    info['accumulated_conditional_entropy'] = state.info['accumulated_conditional_entropy'] + conditional_entropy
    info['invariant_physics_state'] = next_odom_state
    info['current_conditional_entropy'] = conditional_entropy
    info['kalman_gain'] = kalman_gain
    info['conditional_var'] = conditional_var
    info['fused_var'] = fused_var
    info['fused_mean'] = fused_mean
    info['epist_var'] = epist_var

    state = state.replace(obs=obs, info=info , pipeline_state=data)
    reward, done, reward_metrics = self._get_rew(state, action)
    info['reward_metrics'] = reward_metrics

    done = jp.where((conditional_entropy > state.info['per_step_cutoff']).any(), 1.0, done) #if the model is uncertain, then done

    done = jp.where((state.info['accumulated_conditional_entropy'] > state.info['accumulated_cutoff']).any(), 1.0, done)

    return state.replace(
        reward=reward, done=done, info=info
    )
  
  
  def model_step(self, state, action, binning_entropy):
    """Single Infoprop ensemble prediction for one (obs, action) pair.

    Returns fused_mean, fused_var, epist_var, kalman_gain, conditional_var,
    conditional_entropy, and binning_entropy.
    """
    model = state.info['model']
    obs_mean = state.info['model_obs_mean']
    obs_std = state.info['model_obs_std']
    next_state_delta_mean = state.info['next_state_delta_mean']
    next_state_delta_std = state.info['next_state_delta_std']
    curr_physics_state = state.info['physics_state']
    curr_odom_state = state.info['invariant_physics_state']
    curr_rng, rng = jax.random.split(state.info['rng']) #jax.random.split(state.info.get('rng', jax.random.PRNGKey(0)))
    physics_state_history = state.info['phys_state_history']
    act_history = state.info['act_history']
    model_input = jp.concatenate([physics_state_history, act_history], axis=-1)
    means_, logvars_ = model.apply_fn(
            {"params": model.params}, model_input, action, obs_mean, obs_std
        )
    vars_ = jp.exp(logvars_) * (next_state_delta_std + 1e-6) ** 2 * self.dt ** 2
    
    means_ = (means_ * (next_state_delta_std + 1e-6) + next_state_delta_mean) * self.dt + curr_physics_state
    means__1 = curr_odom_state[None,0:3] + self.dt * (curr_physics_state[None,jp.array([2,5,6])]+means_[:, jp.array([2,5,6])])/2
    # R = RZ(means__1[:,0]) @ RX(means_[:,0]) @ RY(means_[:,1])
    R = jax.vmap(YRP)(curr_odom_state[None, 0], curr_physics_state[None, 0], curr_physics_state[None, 1])
    # x,y integration only; z is now directly predicted by the model at physics_state index 10
    means__2 = curr_odom_state[None, 3:5] + self.dt * (R @ curr_physics_state[None, jp.array([7,8,9]), None] + R @ means_[:,jp.array([7,8,9]),None]).squeeze(axis=-1)[:,:2]/2
    means__ = jp.concatenate((means__1, means__2), axis=-1)
    means = jp.concatenate((means_, means__), axis=-1)

    vars__1 = (self.dt/2) ** 2 * vars_[:,jp.array([2,5,6])]
    # x,y variance only from body_vel propagation; z variance comes from model directly
    vars__2 = (self.dt/2) ** 2 * (jp.square(R) @ vars_[:,jp.array([7,8,9]),None]).squeeze(-1)[:,:2]
    vars__ = jp.concatenate((vars__1, vars__2), axis=-1)
    vars = jp.concatenate((vars_, vars__), axis=-1)

    fused_var = 1/jp.mean(1/(vars+1e-12), axis=0)
    fused_mean = fused_var * jp.mean(means / (vars+1e-12), axis=0)
    epist_var = jp.mean((means - fused_mean[None,:]) ** 2, axis=0)
    kalman_gain = jp.clip((fused_var) / (fused_var + epist_var), 0, 1)
    conditional_var = ((1 - kalman_gain) * fused_var)
    conditional_entropy = jp.clip( 0.5 * jp.log2(2 * jp.pi * jp.e * conditional_var) - binning_entropy, 0, None)

    next_full_physics_state = fused_mean + jax.random.normal(curr_rng, shape = (fused_mean.shape[0],)) * jp.sqrt(fused_var)
    next_physics_state = next_full_physics_state[:-5]
    next_odom_state = next_full_physics_state[-5:]
    return next_physics_state, rng, conditional_entropy, next_odom_state, kalman_gain, conditional_var, fused_var, fused_mean, epist_var

  def init_NN_trainer(self, seed, learning_rate, weight_decay, hidden_layer_sizes, model_layer_norm):
    """Instantiate and return a ModelTrainer for the ensemble dynamics model."""
    model_trainer = ModelTrainer(
      seed=seed,
      observation_size=(11),
      action_size=2,
      model_lr=learning_rate,
      model_wd=weight_decay,
      model_hidden_dims=hidden_layer_sizes[0],
      model_num_layers=len(hidden_layer_sizes)+1, # seems like GaussianMLP requires this
      model_min_log_var=self.min_log_var,
      model_max_log_var=self.max_log_var,
      model_layer_norm=model_layer_norm,
      obs_history=self.obs_history,
      act_history=self.act_history
    )
    return model_trainer
