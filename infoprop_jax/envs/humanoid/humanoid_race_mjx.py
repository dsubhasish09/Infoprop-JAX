"""Humanoid racing environment: the MJX Humanoid on scaled Wheelbot tracks.

Reuses the Wheelbot's trajectory logic (cross-track / cross-angle errors and
lookahead waypoints, see infoprop_jax/envs/wheelbot/trajectory.py) on tracks
scaled by the humanoid/wheelbot height ratio (see race_track.py).

Observation: [trajectory state, humanoid observation minus yaw]
  trajectory state: [cte, cae, lookahead distances, lookahead angles]
                    (2 + 2*lookahead dims, or 3 + 3*lookahead with sin/cos encoding)
  humanoid part:    [z, roll, pitch, joint_qpos(21), body_vel(3), euler_rates(3),
                     joint_qvel(21)] (51 dims; x/y excluded as before, yaw replaced
                     by the trajectory features)

Reward (Wheelbot racing form):
  reward = rew_scale * [(1 - done) * (ct_rew + ca_rew + driving_weight * v_proj)
                        + done * crash_penalty]
  with done = off-track (|cte| > track_width/2) OR unhealthy (z outside
  healthy_z_range), and v_proj the heading-frame forward velocity projected onto
  the track direction (analog of the Wheelbot's get_projected_velocity).
"""

import xml.etree.ElementTree as ET
from typing import Optional

import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from brax.envs.base import State
from brax.training.types import Transition
from omegaconf import dictconfig

from infoprop_jax.envs.humanoid.humanoid_mjx import (
    HUMANOID_ROOT_PATH,
    HumanoidEnv,
    _mat_to_quat,
    _rx,
    _ry,
    _yrp,
)
from infoprop_jax.envs.humanoid.race_track import (
    NUM_TRACKS,
    get_trajectory_by_seed,
    scaled_cones,
    track_width,
)
from infoprop_jax.envs.wheelbot.utils import compute_line_element


class HumanoidRaceEnv(HumanoidEnv):
    """Humanoid MJX racing task: drive the scaled Wheelbot tracks.

    Inherits the humanoid physics-state representation and all Infoprop hooks that
    are track-agnostic (preprocess, augment_prediction, ...); overrides observation,
    reward and the reset protocol to mirror the Wheelbot racing env.
    """

    def __init__(
        self,
        cfg: dictconfig.DictConfig = dictconfig.DictConfig({}),
        visualize: bool = False,
        track_seed: Optional[int] = None,
        eval_mode: bool = False,
        **kwargs,
    ):
        mj_model = None
        if visualize and track_seed is not None:
            xml_str = (HUMANOID_ROOT_PATH / "humanoid.xml").read_text()
            root = ET.fromstring(xml_str)
            worldbody = root.find("worldbody")
            inner_cones, outer_cones = scaled_cones(track_seed)
            for cones in (inner_cones, outer_cones):
                for i in range(len(cones) - 1):
                    worldbody.append(
                        compute_line_element(
                            cones[i], cones[i + 1],
                            half_width=0.1, half_height=0.001, z=0.001,
                        )
                    )
            mj_model = mujoco.MjModel.from_xml_string(
                ET.tostring(root, encoding="unicode")
            )

        super().__init__(cfg=cfg, eval_mode=eval_mode, mj_model=mj_model, **kwargs)
        self.eval_mode = eval_mode
        self.track_seed = track_seed
        # Reward parameters (Wheelbot racing form).
        self.ct_weight = cfg.get("ct_weight", 1.0)
        self.ca_weight = cfg.get("ca_weight", 1.0)
        self.driving_weight = cfg.get("driving_weight", 1.0)
        self.crash_penalty = cfg.get("crash_penalty", -200.0)
        self.rew_scale = cfg.get("rew_scale", 0.1)
        # Trajectory observation parameters.
        self.lookahead = cfg.get("lookahead", 10)
        self.sin_cos_encoding = cfg.get("sin_cos_encoding", False)
        # Randomize env parameters (real-env data-collection reset).
        self.init_xy_std = cfg.get("init_xy_std", track_width / 16)
        self.init_angle_std = cfg.get("init_angle_std", jp.pi / 16)
        # Wider spread used when re-seeding model rollouts from the replay buffer.
        self.resample_init_xy_std = cfg.get("resample_init_xy_std", track_width / 3)
        self.resample_init_angle_std = cfg.get("resample_init_angle_std", jp.pi / 3)
        _enc = self.sin_cos_encoding
        self._obs_slice_start = (3 if _enc else 2) * self.lookahead + (3 if _enc else 2)

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
                    "track_seed": 0,
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
                    "track_seed": next_state.info["track_seed"],
                    # Context at the same timestep as the last history entry: consumers
                    # (reset_from_buffer, get_cutoffs) pair it with that state.
                    "invariant_physics_state": prev_state.info[
                        "invariant_physics_state"
                    ],
                },
            },
        )

    # ------------------------------------------------------------- real MJX env
    def _get_obs_from_states(
        self,
        physics_state: jp.ndarray,
        invariant_physics_state: jp.ndarray,
        track_seed: jp.ndarray,
    ) -> jp.ndarray:
        trajectory = get_trajectory_by_seed(track_seed, self.lookahead)
        traj_state = trajectory.get_state(
            invariant_physics_state[1:3],  # x, y
            invariant_physics_state[0],    # yaw
            self.sin_cos_encoding,
        )
        nj = self.sys.nq - 7
        # Humanoid observation minus yaw (x/y already excluded).
        position = jp.concatenate(
            [physics_state[0:3], physics_state[9 : 9 + nj]]  # z, roll, pitch, joint_qpos
        )
        velocity = jp.concatenate(
            [physics_state[6:9], physics_state[3:6], physics_state[9 + nj :]]
        )
        return jp.concatenate([traj_state, position, velocity])

    def _get_obs(
        self, data: mjx.Data, action: jp.ndarray, track_seed: jp.ndarray
    ) -> jp.ndarray:
        return self._get_obs_from_states(
            self._physics_state(data), self._invariant_physics_state(data), track_seed
        )

    def _get_rew(self, state: State, action: jp.ndarray):
        """Compute the step reward.

        reward = rew_scale * [
            (1 - done) * ct_weight  * (track_width/2 - |cte|) / (track_width/2)
          + (1 - done) * ca_weight  * (pi/2 - |cae|) / (pi/2)
          + (1 - done) * driving_weight * projected_velocity
          + done       * crash_penalty
        ]
        done = off-track OR unhealthy (z outside healthy_z_range).
        """
        obs = state.obs
        physics_state = state.info["physics_state"]
        cross_track_error = obs[0]
        cross_angle_error = (
            obs[1] if not self.sin_cos_encoding else jp.arctan2(obs[1], obs[2])
        )
        cross_track_rew = (
            self.ct_weight
            * (track_width / 2 - jp.abs(cross_track_error))
            / (track_width / 2)
        )
        cross_angle_rew = (
            self.ca_weight * (jp.pi / 2 - jp.abs(cross_angle_error)) / (jp.pi / 2)
        )
        # Heading-frame forward velocity projected onto the track direction
        # (analog of the Wheelbot's get_projected_velocity).
        roll, pitch = physics_state[1], physics_state[2]
        forward_velocity = (_rx(roll) @ _ry(pitch) @ physics_state[6:9, None])[0, 0]
        projected_velocity = forward_velocity * jp.cos(cross_angle_error)
        driving_reward = self.driving_weight * projected_velocity

        min_z, max_z = self._healthy_z_range
        unhealthy = jp.logical_or(
            physics_state[0] < min_z, physics_state[0] > max_z
        )
        off_track = jp.abs(cross_track_error) > track_width / 2
        done = jp.where(jp.logical_or(unhealthy, off_track), 1.0, 0.0)
        crash_penalty = jp.float32(self.crash_penalty)

        reward = self.rew_scale * (
            (1 - done) * cross_track_rew
            + (1 - done) * cross_angle_rew
            + (1 - done) * driving_reward
            + done * crash_penalty
        )
        reward_metrics = {
            "cross_track_rew": cross_track_rew,
            "cross_angle_rew": cross_angle_rew,
            "driving_reward": driving_reward,
            "crash_penalty": crash_penalty,
        }
        return reward, done, reward_metrics

    @staticmethod
    def _zero_reward_metrics():
        return {
            "cross_track_rew": 0.0,
            "cross_angle_rew": 0.0,
            "driving_reward": 0.0,
            "crash_penalty": 0.0,
        }

    def _sample_track_pose(self, rng, trajectory, xy_std, angle_std):
        """Sample a start pose on the trajectory with lateral and yaw noise."""
        rng, pos_key = jax.random.split(rng)
        init_xy, init_angle = trajectory.get_rand_init_pos(pos_key)
        rng, xy_key = jax.random.split(rng)
        d = jp.clip(
            jax.random.normal(xy_key, shape=()) * xy_std,
            -track_width / 2,
            track_width / 2,
        )
        perp_dir = jp.array([-jp.sin(init_angle), jp.cos(init_angle)])
        init_xy = init_xy + d * perp_dir
        rng, angle_key = jax.random.split(rng)
        offset_angle = jp.clip(
            jax.random.normal(angle_key, shape=init_angle.shape) * angle_std,
            -jp.pi,
            jp.pi,
        )
        init_angle = init_angle + offset_angle
        return rng, init_xy, init_angle

    def reset(self, rng: jp.ndarray) -> State:
        """Reset to a random point on the track with the humanoid's standing pose."""
        rng, track_key = jax.random.split(rng)
        if self.eval_mode:
            track_seed = jax.random.randint(track_key, shape=(), minval=0, maxval=NUM_TRACKS)
        else:
            track_seed = jp.array(21)
        trajectory = get_trajectory_by_seed(track_seed, self.lookahead)
        rng, init_xy, init_angle = self._sample_track_pose(
            rng, trajectory, self.init_xy_std, self.init_angle_std
        )
        return self._reset_at(rng, track_seed, init_xy, init_angle)

    def reset_to_start(self, rng: jp.ndarray) -> State:
        """Reset to the beginning of the track (segment 0) without randomisation.

        Requires the env to have been constructed with a concrete ``track_seed``
        (used by video evaluation on the visualised track).
        """
        track_seed = jp.array(self.track_seed)
        trajectory = get_trajectory_by_seed(track_seed, self.lookahead)
        init_xy, init_angle = trajectory.get_init_pos(0)
        return self._reset_at(rng, track_seed, init_xy, init_angle)

    def _reset_at(self, rng, track_seed, init_xy, init_angle) -> State:
        # Humanoid standing pose with the usual reset noise, placed on the track.
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        euler_noise = jax.random.uniform(rng3, (3,), minval=low, maxval=hi)
        qpos = qpos.at[3:7].set(
            _mat_to_quat(
                _yrp(init_angle + euler_noise[0], euler_noise[1], euler_noise[2])
            )
        )
        qpos = qpos.at[0:2].set(init_xy)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)
        data = self.pipeline_init(qpos, qvel)

        action = jp.zeros(self.action_size)
        obs = self._get_obs(data, action, track_seed)
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
        info = {
            "track_seed": track_seed,
            "physics_state": physics_state,
            "invariant_physics_state": invariant_physics_state,
            "applied_action": action,
            "phys_state_history": phys_history,
            "act_history": act_history,
            "reward_metrics": self._zero_reward_metrics(),
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
        track_seed = state.info["track_seed"]
        action = jp.clip(action, -1.0, 1.0)
        data = self.pipeline_step(data0, action)
        physics_state = self._physics_state(data)
        invariant_physics_state = self._invariant_physics_state(data)
        obs = self._get_obs_from_states(
            physics_state, invariant_physics_state, track_seed
        )
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
    def postprocess(
        self,
        state,
        applied_action,
        next_model_state,
        next_context,
        processed_action,
    ):
        track_seed = state.info["track_seed"]
        build_pipeline_state = not self.fast_model_rollout
        data = None
        if build_pipeline_state:
            qpos, qvel = self._split_physics_state(next_model_state, next_context)
            data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs_from_states(next_model_state, next_context, track_seed)
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
        """Reset a model rollout from a sampled real-data physics transition.

        Like the Wheelbot: the buffered local state history is kept, but the track
        and the global pose are resampled so model rollouts branch from a diverse
        initial-state distribution.
        """
        build_pipeline_state = not self.fast_model_rollout
        init_history = init_transition.observation
        ms = self.model_state_size
        init_phys_history = init_history[: ms * self.obs_history]
        init_act_history = init_history[ms * self.obs_history :]
        init_physics_state = init_phys_history[-ms:]

        rng, track_key = jax.random.split(rng)
        track_seed = jax.random.randint(track_key, shape=(), minval=0, maxval=NUM_TRACKS)
        trajectory = get_trajectory_by_seed(track_seed, self.lookahead)
        rng, init_xy, init_angle = self._sample_track_pose(
            rng, trajectory, self.resample_init_xy_std, self.resample_init_angle_std
        )
        invariant_physics_state = jp.array([init_angle, init_xy[0], init_xy[1]])

        data = None
        if build_pipeline_state:
            qpos, qvel = self._split_physics_state(
                init_physics_state, invariant_physics_state
            )
            data = self.pipeline_init(qpos, qvel)
        action = jp.zeros(self.action_size)
        obs = self._get_obs_from_states(
            init_physics_state, invariant_physics_state, track_seed
        )
        info = {
            "track_seed": track_seed,
            "physics_state": init_physics_state,
            "invariant_physics_state": invariant_physics_state,
            "applied_action": action,
            "phys_state_history": init_phys_history,
            "act_history": init_act_history,
            "accumulated_conditional_entropy": jp.zeros((self.full_state_size,)),
            "current_conditional_entropy": jp.zeros((self.full_state_size,)),
            "reward_metrics": self._zero_reward_metrics(),
        }
        return State(data, obs, 0.0, 0.0, {}, info)

    def reset_with_init_state_eval(self, rng, init_history, track_seed, init_xy, init_angle):
        """Deterministic reset for evaluation (always builds the pipeline_state for rendering)."""
        ms = self.model_state_size
        init_phys_history = init_history[: ms * self.obs_history]
        init_act_history = init_history[ms * self.obs_history :]
        init_physics_state = init_phys_history[-ms:]
        invariant_physics_state = jp.array([init_angle, init_xy[0], init_xy[1]])

        qpos, qvel = self._split_physics_state(
            init_physics_state, invariant_physics_state
        )
        data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs_from_states(
            init_physics_state, invariant_physics_state, track_seed
        )
        info = {
            "track_seed": track_seed,
            "physics_state": init_physics_state,
            "invariant_physics_state": invariant_physics_state,
            "applied_action": jp.zeros(self.action_size),
            "phys_state_history": init_phys_history,
            "act_history": init_act_history,
            "accumulated_conditional_entropy": jp.zeros((self.full_state_size,)),
            "current_conditional_entropy": jp.zeros((self.full_state_size,)),
            "reward_metrics": self._zero_reward_metrics(),
        }
        return State(data, obs, 0.0, 0.0, {}, info)
