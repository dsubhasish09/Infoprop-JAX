# Environments

This directory contains the two Brax environments and the trajectory utility used across training and evaluation.

## Two Environments, One Interface

The codebase has two environments that share the same `Wheelbot` class name and Brax `PipelineEnv` interface but differ in what happens inside `step()`:

| | `wheelbot_brax_mjx.py` | `wheelbot_brax_infoprop.py` |
|---|---|---|
| **Physics** | MuJoCo MJX (ground truth) | Learned probabilistic ensemble |
| **Used for** | Data collection + evaluation | Model-based training rollouts |
| **Speed** | ~1 env in real time | Many envs in parallel on GPU |
| **Termination** | Physics-based (robot falls) | Physics + entropy cutoffs |

During training, `brax_infoprop_train.py` registers both under different names and passes them to `infoprop.train()`.

## State Vector

The full state is 16D, split into variant and invariant components:

```
Variant   s^V — body-frame dynamics + height, learned by the model (11D):
  [roll, pitch, yaw_rate, roll_rate, pitch_rate,
   drive_angle_rate, balance_angle_rate,
   body_vx, body_vy, body_vz, z]

Invariant s^I — global odometry, invariant to dynamics (5D):
  [yaw, drive_angle, balance_angle, x, y]
```

The variant state changes with robot dynamics; the invariant state tracks global heading, wheel angles, and planar position.
The Infoprop model is trained on the concatenated state `[s^V, s^I]`.

## Observation Space

The agent observes a feature vector composed of:

```
[trajectory_features | masked_robot_state (10D) | z, body_vx, body_vy, body_vz]
```

**Trajectory features** (computed by `trajectory.py`):
- Cross-track error: signed distance from robot position to the nearest centreline segment
- Cross-angle error: difference between robot heading and centreline direction
- `lookahead` upcoming waypoint distances
- `lookahead` upcoming waypoint angles (relative to robot heading)

Angles can optionally be encoded as (sin, cos) pairs via `sin_cos_encoding` in the config.

**Masked robot state (10D):** the full 10D state from `qpos_qvel_to_robot_state` with yaw, drive_angle, and balance_angle zeroed out (mask `[0,1,1,1,1,1,0,1,0,1]`).

**Extras:** height `z` and body-frame velocity `[body_vx, body_vy, body_vz]`.

The dynamics model additionally receives a history of the last `obs_history` steps of `physics_state` (11D variant state) and the last `act_history` applied torques, stored in `state.info['phys_state_history']` and `state.info['act_history']`.

## Reward Function

```
r = rew_scale * [
    (1 - done) * ct_weight  * (track_width/2 - |cross_track_error|) / (track_width/2)
  + (1 - done) * ca_weight  * (pi/2 - |cross_angle_error|) / (pi/2)
  + (1 - done) * driving_weight * longitudinal_velocity
  + done       * crash_penalty
]
```

Reward weights are defined in `config/env/wheelbot.yaml`.

## Action Space

2D continuous: `[tau_drive, tau_balance]` — torque commands for the driving and reaction wheels.

The RL agent's output is *added to* a linear balancing prior `tau_bal` (a PD controller with gains `K_roll` and `K_pitch`). The agent has enough authority to fully override the prior.

## Infoprop-specific State in `wheelbot_brax_infoprop.py`

The model environment carries extra per-step uncertainty metrics in `state.info`:

| Key | Description |
|---|---|
| `accumulated_conditional_entropy` | Sum of per-step H(s̃) over the rollout so far |
| `current_conditional_entropy` | Per-step H(s̃) for the most recent step |
| `per_step_cutoff` | λ₁ threshold (computed from real-data buffer) |
| `accumulated_cutoff` | λ₂ threshold |
| `kalman_gain` | K per state dimension |
| `conditional_var` | (1 − K) · Sigma_GT per state dimension |
| `fused_mean` / `fused_var` | Precision-weighted ensemble fusion |
| `epist_var` | Epistemic variance (ensemble disagreement) |

## File Map

| File | Role |
|---|---|
| `wheelbot_brax_mjx.py` | Ground-truth MJX environment. Used for data collection and evaluation. |
| `wheelbot_brax_infoprop.py` | Model-based environment. Used for parallel training rollouts. |
| `trajectory.py` | `Trajectory` dataclass: cross-track/angle errors, lookahead waypoints, initial-position sampling. |
| `utils.py` | `compute_line_element` geometry helper (imported by both env files). |
