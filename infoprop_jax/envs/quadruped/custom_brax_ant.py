# Copyright (c) 2026 Devdutt Subhasish
# SPDX-License-Identifier: MIT
"""Ant: A custom version of stock Brax Ant, that uses Euler angles instead of quaternions for observations.
"""

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
import mujoco
from jax.scipy.spatial.transform import Rotation


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

class Ant(PipelineEnv):



  # pyformat: disable
  """
  ### Description

  This environment is based on the environment introduced by Schulman, Moritz,
  Levine, Jordan and Abbeel in
  ["High-Dimensional Continuous Control Using Generalized Advantage Estimation"](https://arxiv.org/abs/1506.02438).

  The ant is a 3D robot consisting of one torso (free rotational body) with four
  legs attached to it with each leg having two links.

  The goal is to coordinate the four legs to move in the forward (right)
  direction by applying torques on the eight hinges connecting the two links of
  each leg and the torso (nine parts and eight hinges).

  ### Action Space

  The agent take a 8-element vector for actions.

  The action space is a continuous `(action, action, action, action, action,
  action, action, action)` all in `[-1, 1]`, where `action` represents the
  numerical torques applied at the hinge joints.

  | Num | Action                                                             | Control Min | Control Max | Name (in corresponding config)   | Joint | Unit         |
  |-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
  | 0   | Torque applied on the rotor between the torso and front left hip   | -1          | 1           | hip_1 (front_left_leg)           | hinge | torque (N m) |
  | 1   | Torque applied on the rotor between the front left two links       | -1          | 1           | ankle_1 (front_left_leg)         | hinge | torque (N m) |
  | 2   | Torque applied on the rotor between the torso and front right hip  | -1          | 1           | hip_2 (front_right_leg)          | hinge | torque (N m) |
  | 3   | Torque applied on the rotor between the front right two links      | -1          | 1           | ankle_2 (front_right_leg)        | hinge | torque (N m) |
  | 4   | Torque applied on the rotor between the torso and back left hip    | -1          | 1           | hip_3 (back_leg)                 | hinge | torque (N m) |
  | 5   | Torque applied on the rotor between the back left two links        | -1          | 1           | ankle_3 (back_leg)               | hinge | torque (N m) |
  | 6   | Torque applied on the rotor between the torso and back right hip   | -1          | 1           | hip_4 (right_back_leg)           | hinge | torque (N m) |
  | 7   | Torque applied on the rotor between the back right two links       | -1          | 1           | ankle_4 (right_back_leg)         | hinge | torque (N m) |

  ### Observation Space

  The state space consists of positional values of different body parts of the
  ant, followed by the velocities of those individual parts (their derivatives)
  with all the positions ordered before all the velocities.

  The observation is a `ndarray` with shape `(26,)` where the elements correspond to the following:

  | Num | Observation                                                  | Min  | Max | Name (in corresponding config)   | Joint | Unit                     |
  |-----|--------------------------------------------------------------|------|-----|----------------------------------|-------|--------------------------|
  | 0   | z-coordinate of the torso (centre)                           | -Inf | Inf | torso                            | free  | position (m)             |
  | 1   | yaw of the torso (rotation about z-axis)                     | -Inf | Inf | torso                            | free  | angle (rad)              |
  | 2   | roll of the torso (rotation about x-axis)                    | -Inf | Inf | torso                            | free  | angle (rad)              |
  | 3   | pitch of the torso (rotation about y-axis)                   | -Inf | Inf | torso                            | free  | angle (rad)              |
  | 4   | angle between torso and first link on front left             | -Inf | Inf | hip_1 (front_left_leg)           | hinge | angle (rad)              |
  | 5   | angle between the two links on the front left                | -Inf | Inf | ankle_1 (front_left_leg)         | hinge | angle (rad)              |
  | 6   | angle between torso and first link on front right            | -Inf | Inf | hip_2 (front_right_leg)          | hinge | angle (rad)              |
  | 7   | angle between the two links on the front right               | -Inf | Inf | ankle_2 (front_right_leg)        | hinge | angle (rad)              |
  | 8   | angle between torso and first link on back left              | -Inf | Inf | hip_3 (back_leg)                 | hinge | angle (rad)              |
  | 9   | angle between the two links on the back left                 | -Inf | Inf | ankle_3 (back_leg)               | hinge | angle (rad)              |
  | 10  | angle between torso and first link on back right             | -Inf | Inf | hip_4 (right_back_leg)           | hinge | angle (rad)              |
  | 11  | angle between the two links on the back right                | -Inf | Inf | ankle_4 (right_back_leg)         | hinge | angle (rad)              |
  | 12  | x-coordinate velocity of the torso (body frame)              | -Inf | Inf | torso                            | free  | velocity (m/s)           |
  | 13  | y-coordinate velocity of the torso (body frame)              | -Inf | Inf | torso                            | free  | velocity (m/s)           |
  | 14  | z-coordinate velocity of the torso (body frame)              | -Inf | Inf | torso                            | free  | velocity (m/s)           |
  | 15  | yaw rate of the torso                                        | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
  | 16  | roll rate of the torso                                       | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
  | 17  | pitch rate of the torso                                      | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
  | 18  | angular velocity of angle between torso and front left link  | -Inf | Inf | hip_1 (front_left_leg)           | hinge | angle (rad)              |
  | 19  | angular velocity of the angle between front left links       | -Inf | Inf | ankle_1 (front_left_leg)         | hinge | angle (rad)              |
  | 20  | angular velocity of angle between torso and front right link | -Inf | Inf | hip_2 (front_right_leg)          | hinge | angle (rad)              |
  | 21  | angular velocity of the angle between front right links      | -Inf | Inf | ankle_2 (front_right_leg)        | hinge | angle (rad)              |
  | 22  | angular velocity of angle between torso and back left link   | -Inf | Inf | hip_3 (back_leg)                 | hinge | angle (rad)              |
  | 23  | angular velocity of the angle between back left links        | -Inf | Inf | ankle_3 (back_leg)               | hinge | angle (rad)              |
  | 24  | angular velocity of angle between torso and back right link  | -Inf | Inf | hip_4 (right_back_leg)           | hinge | angle (rad)              |
  | 25  | angular velocity of the angle between back right links       | -Inf | Inf | ankle_4 (right_back_leg)         | hinge | angle (rad)              |

  The (x,y,z) coordinates are translational DOFs while the orientations are
  rotational DOFs expressed as (yaw, roll, pitch) Euler angles
  (Rz(yaw) @ Rx(roll) @ Ry(pitch)). The torso's linear velocity is expressed
  in the body frame, and the torso's angular velocity is expressed as
  (yaw, roll, pitch) Euler-angle rates rather than the world-frame angular
  velocity vector.

  ### Rewards

  The reward consists of three parts:

  - *reward_survive*: Every timestep that the ant is alive, it gets a reward of
    1.
  - *reward_forward*: A reward of moving forward which is measured as
    *(x-coordinate before action - x-coordinate after action)/dt*. *dt* is the
    time between actions - the default *dt = 0.05*. This reward would be
    positive if the ant moves forward (right) desired.
  - *reward_ctrl*: A negative reward for penalising the ant if it takes actions
    that are too large. It is measured as *coefficient **x**
    sum(action<sup>2</sup>)* where *coefficient* is a parameter set for the
    control and has a default value of 0.5.
  - *contact_cost*: A negative reward for penalising the ant if the external
    contact force is too large. It is calculated *0.5 * 0.001 *
    sum(clip(external contact force to [-1,1])<sup>2</sup>)*.

  ### Starting State

  All observations start in state (0.0, 0.0,  0.75, 0.0, 0.0  ... 0.0) with a
  uniform noise in the range of [-0.1, 0.1] added to the positional values and
  standard normal noise with 0 mean and 0.1 standard deviation added to the
  velocity values for stochasticity.

  Note that the initial z coordinate is intentionally selected to be slightly
  high, thereby indicating a standing up ant. The initial orientation is
  designed to make it face forward as well.

  ### Episode Termination

  The episode terminates when any of the following happens:

  1. The episode duration reaches a 1000 timesteps
  2. The z-coordinate of the torso (index 0) is **not** in the range
     `[0.2, 1.0]`
  """
  # pyformat: enable


  def __init__(
      self,
      ctrl_cost_weight=0.5,
      use_contact_forces=False,
      contact_cost_weight=5e-4,
      healthy_reward=1.0,
      terminate_when_unhealthy=True,
      healthy_z_range=(0.2, 1.0),
      contact_force_range=(-1.0, 1.0),
      reset_noise_scale=0.1,
      exclude_current_positions_from_observation=True,
      backend='generalized',
      **kwargs,
  ):
    path = epath.resource_path('brax') / 'envs/assets/ant.xml'
    sys = mjcf.load(path)

    n_frames = 5

    if backend in ['spring', 'positional']:
      sys = sys.tree_replace({'opt.timestep': 0.005})
      n_frames = 10

    if backend == 'mjx':
      sys = sys.tree_replace({
          'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
          'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
          'opt.iterations': 1,
          'opt.ls_iterations': 4,
      })

    if backend == 'positional':
      # TODO(brax-team): does the same actuator strength work as in spring
      sys = sys.replace(
          actuator=sys.actuator.replace(
              gear=200 * jp.ones_like(sys.actuator.gear)
          )
      )

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)

    self._ctrl_cost_weight = ctrl_cost_weight
    self._use_contact_forces = use_contact_forces
    self._contact_cost_weight = contact_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._contact_force_range = contact_force_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )

    if self._use_contact_forces:
      raise NotImplementedError('use_contact_forces not implemented.')

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    q = self.sys.init_q + jax.random.uniform(
        rng1, (self.sys.q_size(),), minval=low, maxval=hi
    )
    qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

    pipeline_state = self.pipeline_init(q, qd)
    obs = self._get_obs(pipeline_state)

    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_forward': zero,
        'reward_survive': zero,
        'reward_ctrl': zero,
        'reward_contact': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
        'forward_reward': zero,
    }
    return State(pipeline_state, obs, reward, done, metrics)

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""
    pipeline_state0 = state.pipeline_state
    assert pipeline_state0 is not None
    pipeline_state = self.pipeline_step(pipeline_state0, action)

    velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
    forward_reward = velocity[0]

    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
    is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
    contact_cost = 0.0

    obs = self._get_obs(pipeline_state)
    reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    state.metrics.update(
        reward_forward=forward_reward,
        reward_survive=healthy_reward,
        reward_ctrl=-ctrl_cost,
        reward_contact=-contact_cost,
        x_position=pipeline_state.x.pos[0, 0],
        y_position=pipeline_state.x.pos[0, 1],
        distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
        forward_reward=forward_reward,
    )
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )

  def _get_obs(self, pipeline_state: base.State) -> jax.Array:
    """Observe ant body position and velocities."""
    qpos = pipeline_state.q
    qvel = pipeline_state.qd
    yaw, roll, pitch = _quat_to_yrp(qpos[3:7])
    qpos = jp.concatenate([qpos[:3], jp.array([yaw, roll, pitch]), qpos[7:]])
    euler_rates = (_jacobian_w2euler(roll, pitch) @ qvel[3:6])[
            jp.array([2, 0, 1])
        ]
    body_vel = (_ry(-pitch) @ _rx(-roll) @ _rz(-yaw) @ qvel[:3, None]).flatten()
    qvel = jp.concatenate([body_vel, euler_rates, qvel[6:]])
    if self._exclude_current_positions_from_observation:
      qpos = qpos[2:]

    return jp.concatenate([qpos] + [qvel])