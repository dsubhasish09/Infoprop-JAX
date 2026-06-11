"""MJX Humanoid environment adapted from the MuJoCo MJX tutorial.

The real environment follows the tutorial's Brax ``PipelineEnv`` implementation.
For Infoprop, the learned model state is a local floating-base representation and
the odometry context is ``[yaw, x, y]``.  Policy observations omit MJX-derived
fields so model rollouts can use the fast path without rebuilding pipeline data.
"""

from etils import epath
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from jax.scipy.spatial.transform import Rotation
from brax import envs
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.training.types import Transition
from omegaconf import dictconfig

from infoprop_jax.envs.infoprop_wrappable_env import InfopropWrappable


HUMANOID_ROOT_PATH = (
    epath.Path(epath.resource_path("mujoco")) / "mjx" / "test_data" / "humanoid"
)


def _rx(theta):
    return jp.array(
        [
            [1, 0, 0],
            [0, jp.cos(theta), -jp.sin(theta)],
            [0, jp.sin(theta), jp.cos(theta)],
        ]
    )


def _ry(theta):
    return jp.array(
        [
            [jp.cos(theta), 0, jp.sin(theta)],
            [0, 1, 0],
            [-jp.sin(theta), 0, jp.cos(theta)],
        ]
    )


def _rz(theta):
    return jp.array(
        [
            [jp.cos(theta), -jp.sin(theta), 0],
            [jp.sin(theta), jp.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def _yrp(yaw, roll, pitch):
    return _rz(yaw) @ _rx(roll) @ _ry(pitch)


def _mat_to_quat(mat):
    return Rotation.from_matrix(mat).as_quat(scalar_first=True)


def _quat_to_mat(q):
    return Rotation.from_quat(q[jp.array([1, 2, 3, 0])]).as_matrix()


def _quat_to_yrp(q):
    mat = _quat_to_mat(q).reshape((3, 3))
    yaw = jp.arctan2(-mat[0, 1], mat[1, 1])
    roll = jp.arcsin(mat[2, 1])
    pitch = jp.arctan2(-mat[2, 0], mat[2, 2])
    return jp.array([yaw, roll, pitch])


def _jacobian_w2euler(roll: jp.ndarray, pitch: jp.ndarray) -> jp.ndarray:
    return jp.array(
        [
            [jp.cos(pitch), 0, jp.sin(pitch)],
            [jp.sin(pitch) * jp.tan(roll), 1, -jp.cos(pitch) * jp.tan(roll)],
            [-jp.sin(pitch) / jp.cos(roll), 0, jp.cos(pitch) / jp.cos(roll)],
        ]
    )



class HumanoidEnv(PipelineEnv, InfopropWrappable):
    """Classic Humanoid MJX task plus Infoprop wrapping hooks."""

    def __init__(
        self,
        cfg: dictconfig.DictConfig = dictconfig.DictConfig({}),
        forward_reward_weight: float | None = None,
        ctrl_cost_weight: float | None = None,
        healthy_reward: float | None = None,
        terminate_when_unhealthy: bool | None = None,
        healthy_z_range: tuple[float, float] | None = None,
        reset_noise_scale: float | None = None,
        exclude_current_positions_from_observation: bool | None = None,
        eval_mode: bool = False,
        mj_model: mujoco.MjModel | None = None,
        **kwargs,
    ):
        if mj_model is None:
            mj_model = mujoco.MjModel.from_xml_path(
                (HUMANOID_ROOT_PATH / "humanoid.xml").as_posix()
            )
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)
        kwargs["n_frames"] = kwargs.get("n_frames", cfg.get("control_interval", 5))
        kwargs["backend"] = "mjx"
        super().__init__(sys, **kwargs)

        self._reward_scale = cfg.get("reward_scale", 0.1)
        self._forward_reward_weight = cfg.get(
            "forward_reward_weight",
            1.25 if forward_reward_weight is None else forward_reward_weight,
        )
        self._ctrl_cost_weight = cfg.get(
            "ctrl_cost_weight", 0.1 if ctrl_cost_weight is None else ctrl_cost_weight
        )
        self._healthy_reward = cfg.get(
            "healthy_reward", 5.0 if healthy_reward is None else healthy_reward
        )
        self._terminate_when_unhealthy = cfg.get(
            "terminate_when_unhealthy",
            True if terminate_when_unhealthy is None else terminate_when_unhealthy,
        )
        self._healthy_z_range = tuple(
            cfg.get(
                "healthy_z_range",
                (1.0, 2.0) if healthy_z_range is None else healthy_z_range,
            )
        )
        self._reset_noise_scale = cfg.get(
            "reset_noise_scale",
            1e-2 if reset_noise_scale is None else reset_noise_scale,
        )
        self._exclude_current_positions_from_observation = cfg.get(
            "exclude_current_positions_from_observation",
            True
            if exclude_current_positions_from_observation is None
            else exclude_current_positions_from_observation,
        )
        self.obs_history = cfg.get("obs_history", 1)
        self.act_history = cfg.get("act_history", 0)
        # Env-owned fast-rollout flag: skip building the MJX pipeline_state during
        # model rollouts. The framework is agnostic to this; see InfopropWrappable.
        self.fast_model_rollout = cfg.get("fast_model_rollout", True)
        # model_state: [z, roll, pitch, yaw_rate, roll_rate, pitch_rate, body_vel(3),
        #               joint_qpos(21), joint_qvel(21)]
        self.model_state_size = 3 + self.sys.nv + (self.sys.nq - 7)
        # context / odometry: [yaw, x, y]
        self.context_size = 3
        self.full_state_size = self.model_state_size + self.context_size

    # ---------------------------------------------------- physics-buffer contract
    @property
    def dummy_physics_transition(self) -> Transition:
        ms, oh, ah = self.model_state_size, self.obs_history, self.act_history
        return Transition(
            observation=jp.zeros(ms * oh + self.action_size * ah),
            action=jp.zeros(self.action_size),
            reward=0.0,
            discount=0.0,
            next_observation=jp.zeros(ms),
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "invariant_physics_state": jp.zeros(self.context_size),
                },
                "policy_extras": {},
            },
        )

    def extract_physics_transition(
        self, prev_state: State, next_state: State, policy_extras
    ) -> Transition:
        return Transition(
            observation=jp.concatenate(
                [prev_state.info["phys_state_history"], prev_state.info["act_history"]],
                axis=-1,
            ),
            action=next_state.info["applied_action"],
            reward=jp.zeros(next_state.reward.shape, dtype=jp.float32),
            discount=1 - next_state.done,
            next_observation=next_state.info["physics_state"],
            extras={
                "policy_extras": policy_extras,
                "state_extras": {
                    "truncation": next_state.info.get("truncation", 0.0),
                    # Context at the same timestep as the last history entry: consumers
                    # (reset_from_buffer, get_cutoffs) pair it with that state.
                    "invariant_physics_state": prev_state.info[
                        "invariant_physics_state"
                    ],
                },
            },
        )

    def context_from_transition(self, transition: Transition):
        return transition.extras["state_extras"]["invariant_physics_state"]

    @property
    def reset_carry_keys(self):
        return [
            "physics_state",
            "invariant_physics_state",
            "applied_action",
            "phys_state_history",
            "act_history",
        ]

    # ------------------------------------------------------------- real MJX env
    def _physics_state(self, data: mjx.Data) -> jp.ndarray:
        quat = data.qpos[3:7]
        yaw, roll, pitch = _quat_to_yrp(quat)
        euler_rates = (_jacobian_w2euler(roll, pitch) @ data.qvel[3:6])[
            jp.array([2, 0, 1])
        ]
        body_vel = (_ry(-pitch) @ _rx(-roll) @ _rz(-yaw) @ data.qvel[:3, None]).flatten()
        return jp.concatenate(
            [
                data.qpos[2:3],
                jp.array([roll, pitch]),
                euler_rates,
                body_vel,
                data.qpos[7:],
                data.qvel[6:],
            ],
            axis=-1,
        )

    def _invariant_physics_state(self, data: mjx.Data) -> jp.ndarray:
        yaw = _quat_to_yrp(data.qpos[3:7])[0]
        return jp.array([yaw, data.qpos[0], data.qpos[1]])

    def _split_physics_state(self, physics_state: jp.ndarray, context: jp.ndarray):
        z = physics_state[0:1]
        roll_pitch = physics_state[1:3]
        euler_rates = physics_state[3:6]
        body_vel = physics_state[6:9]
        joint_qpos = physics_state[9 : 9 + (self.sys.nq - 7)]
        joint_qvel = physics_state[9 + (self.sys.nq - 7) :]
        yaw, x, y = context
        quat = _mat_to_quat(_yrp(yaw, roll_pitch[0], roll_pitch[1]))
        qpos = jp.concatenate([jp.array([x, y]), z, quat, joint_qpos], axis=-1)
        yaw_rate, roll_rate, pitch_rate = euler_rates
        rot = _yrp(yaw, roll_pitch[0], roll_pitch[1])
        world_omega = (
            jp.array([0.0, 0.0, yaw_rate])
            + _rz(yaw) @ jp.array([roll_rate, 0.0, 0.0])
            + _rz(yaw) @ _rx(roll_pitch[0]) @ jp.array([0.0, pitch_rate, 0.0])
        )
        # MuJoCo free-joint qvel: linear velocity in world frame, angular in body frame.
        qvel = jp.concatenate([rot @ body_vel, rot.T @ world_omega, joint_qvel])
        return qpos, qvel

    def _get_obs_from_states(
        self, physics_state: jp.ndarray, invariant_physics_state: jp.ndarray
    ) -> jp.ndarray:
        z = physics_state[0:1]
        roll_pitch = physics_state[1:3]
        yaw = invariant_physics_state[0:1]
        joint_qpos = physics_state[9 : 9 + (self.sys.nq - 7)]
        joint_qvel = physics_state[9 + (self.sys.nq - 7) :]
        orientation = jp.concatenate([yaw, roll_pitch])
        velocity = jp.concatenate([physics_state[6:9], physics_state[3:6], joint_qvel])
        position = jp.concatenate([invariant_physics_state[1:], z, orientation, joint_qpos])
        if self._exclude_current_positions_from_observation:
            position = position[2:]
        return jp.concatenate([position, velocity])

    def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
        return self._get_obs_from_states(
            self._physics_state(data), self._invariant_physics_state(data)
        )

    def _get_rew_from_states(
        self,
        physics_state: jp.ndarray,
        context: jp.ndarray,
        action: jp.ndarray,
    ):
        world_velocity = _yrp(context[0], physics_state[1], physics_state[2]) @ physics_state[6:9]
        forward_reward = self._forward_reward_weight * world_velocity[0]
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(physics_state[0] < min_z, 0.0, 1.0)
        is_healthy = jp.where(physics_state[0] > max_z, 0.0, is_healthy)
        healthy_reward = jp.where(
            self._terminate_when_unhealthy,
            self._healthy_reward,
            self._healthy_reward * is_healthy,
        )
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        done = jp.where(
            self._terminate_when_unhealthy, 1.0 - is_healthy, jp.array(0.0)
        )
        reward = self._reward_scale * (forward_reward + healthy_reward - ctrl_cost)
        reward_metrics = {
            "forward_reward": forward_reward,
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "x_position": context[1],
            "y_position": context[2],
            "distance_from_origin": jp.linalg.norm(context[1:]),
            "x_velocity": world_velocity[0],
            "y_velocity": world_velocity[1],
        }
        return reward, done, reward_metrics

    def _get_rew(self, state: State, action: jp.ndarray):
        return self._get_rew_from_states(
            state.info["physics_state"],
            state.info["invariant_physics_state"],
            action,
        )

    def reset(self, rng: jp.ndarray) -> State:
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        euler_noise = jax.random.uniform(rng3, (3,), minval=low, maxval=hi)
        qpos = qpos.at[3:7].set(_mat_to_quat(_yrp(*euler_noise)))
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)
        data = self.pipeline_init(qpos, qvel)
        action = jp.zeros(self.action_size)
        obs = self._get_obs(data, action)
        physics_state = self._physics_state(data)
        invariant_physics_state = self._invariant_physics_state(data)
        phys_history = (
            jp.zeros(self.model_state_size * self.obs_history)
            if self.obs_history > 0
            else jp.array([])
        )
        act_history = (
            jp.zeros(self.action_size * self.act_history)
            if self.act_history > 0
            else jp.array([])
        )
        reward_metrics = {
            "forward_reward": 0.0,
            "reward_linvel": 0.0,
            "reward_quadctrl": 0.0,
            "reward_alive": 0.0,
            "x_position": 0.0,
            "y_position": 0.0,
            "distance_from_origin": 0.0,
            "x_velocity": 0.0,
            "y_velocity": 0.0,
        }
        info = {
            "physics_state": physics_state,
            "invariant_physics_state": invariant_physics_state,
            "applied_action": action,
            "phys_state_history": phys_history,
            "act_history": act_history,
            "reward_metrics": reward_metrics,
            "rng": rng,
        }
        state = State(data, obs, 0.0, 0.0, {}, info)
        warmup_steps = max(self.obs_history, self.act_history)

        def func(carry, _):
            state = self.step(carry, jp.zeros(self.action_size))
            return state, None

        state, _ = jax.lax.scan(func, state, None, warmup_steps)
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        data0 = state.pipeline_state
        action = jp.clip(action, -1.0, 1.0)
        data = self.pipeline_step(data0, action)
        physics_state = self._physics_state(data)
        invariant_physics_state = self._invariant_physics_state(data)
        obs = self._get_obs_from_states(physics_state, invariant_physics_state)
        info = state.info
        info["physics_state"] = physics_state
        info["invariant_physics_state"] = invariant_physics_state
        info["applied_action"] = action
        info["phys_state_history"] = self.shift_phys(
            info["phys_state_history"], physics_state
        )
        info["act_history"] = self.shift_action(info["act_history"], action)
        state = state.replace(pipeline_state=data, obs=obs, info=info)
        reward, done, reward_metrics = self._get_rew(state, action)
        info["reward_metrics"] = reward_metrics
        return state.replace(reward=reward, done=done, info=info)

    # ------------------------------------------------------------- Infoprop hooks
    def preprocess(self, state: State, action: jp.ndarray):
        action = jp.clip(action, -1.0, 1.0)
        nn_input = jp.concatenate(
            [state.info["phys_state_history"], state.info["act_history"]], axis=-1
        )
        return (
            nn_input,
            state.info["physics_state"],
            state.info["invariant_physics_state"],
            action,
            action,
        )

    def augment_prediction(self, member_mean, member_var, curr_model_state, curr_context):
        dt = self.dt
        curr_yaw_xy = curr_context
        curr_euler_rates = curr_model_state[3:6]
        next_euler_rates = member_mean[:, 3:6]
        yaw = curr_yaw_xy[0] + dt * (curr_euler_rates[0] + next_euler_rates[:, 0]) / 2

        curr_rot = _yrp(curr_yaw_xy[0], curr_model_state[1], curr_model_state[2])
        curr_world_vel = curr_rot @ curr_model_state[6:9]
        next_world_vel = (curr_rot[None] @ member_mean[:, 6:9, None]).squeeze(-1)
        xy = curr_yaw_xy[1:3] + dt * (curr_world_vel[None, :2] + next_world_vel[:, :2]) / 2
        context_mean = jp.concatenate([yaw[:, None], xy], axis=-1)
        full_mean = jp.concatenate([member_mean, context_mean], axis=-1)

        yaw_var = (dt / 2) ** 2 * member_var[:, 3]
        xy_var = (dt / 2) ** 2 * (jp.square(curr_rot)[None] @ member_var[:, 6:9, None]).squeeze(-1)[:, :2]
        context_var = jp.concatenate([yaw_var[:, None], xy_var], axis=-1)
        full_var = jp.concatenate([member_var, context_var], axis=-1)
        return full_mean, full_var

    def postprocess(
        self,
        state,
        applied_action,
        next_model_state,
        next_context,
        processed_action,
    ):
        build_pipeline_state = not self.fast_model_rollout
        data = None
        if build_pipeline_state:
            qpos, qvel = self._split_physics_state(next_model_state, next_context)
            data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs_from_states(next_model_state, next_context)
        info = state.info
        info["physics_state"] = next_model_state
        info["invariant_physics_state"] = next_context
        info["applied_action"] = applied_action
        info["phys_state_history"] = self.shift_phys(
            info["phys_state_history"], next_model_state
        )
        info["act_history"] = self.shift_action(info["act_history"], applied_action)
        state = state.replace(pipeline_state=data, obs=obs, info=info)
        reward, done, reward_metrics = self._get_rew(state, processed_action)
        info["reward_metrics"] = reward_metrics
        return state.replace(reward=reward, done=done, info=info)

    def reset_from_buffer(self, rng, init_transition):
        build_pipeline_state = not self.fast_model_rollout
        init_history = init_transition.observation
        ms = self.model_state_size
        init_phys_history = init_history[: ms * self.obs_history]
        init_act_history = init_history[ms * self.obs_history :]
        init_physics_state = init_phys_history[-ms:]
        invariant_physics_state = init_transition.extras["state_extras"][
            "invariant_physics_state"
        ]
        data = None
        if build_pipeline_state:
            qpos, qvel = self._split_physics_state(
                init_physics_state, invariant_physics_state
            )
            data = self.pipeline_init(qpos, qvel)
        action = jp.zeros(self.action_size)
        obs = self._get_obs_from_states(init_physics_state, invariant_physics_state)
        info = {
            "physics_state": init_physics_state,
            "invariant_physics_state": invariant_physics_state,
            "applied_action": action,
            "phys_state_history": init_phys_history,
            "act_history": init_act_history,
            "accumulated_conditional_entropy": jp.zeros((self.full_state_size,)),
            "current_conditional_entropy": jp.zeros((self.full_state_size,)),
            "reward_metrics": {
                "forward_reward": 0.0,
                "reward_linvel": 0.0,
                "reward_quadctrl": 0.0,
                "reward_alive": 0.0,
                "x_position": 0.0,
                "y_position": 0.0,
                "distance_from_origin": 0.0,
                "x_velocity": 0.0,
                "y_velocity": 0.0,
            },
        }
        return State(data, obs, 0.0, 0.0, {}, info)


envs.register_environment("humanoid", HumanoidEnv)
