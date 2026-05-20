# Wheelbot Environment

This directory contains the Wheelbot-specific real environment, trajectory features, and bundled assets used by the current Infoprop training setup.

## Files

| File | Role |
|---|---|
| `wheelbot_brax_mjx.py` | Ground-truth MuJoCo/MJX environment for real data collection and evaluation. |
| `trajectory.py` | `Trajectory` dataclass: cross-track/angle errors, lookahead waypoints, initial-position sampling. |
| `utils.py` | Geometry helpers for visualizing tracks in MuJoCo XML. |
| `assets/mjcf/` | Wheelbot XML and mesh assets. |
| `assets/track/` | Procedural track generation code and bundled pre-generated `.npz` tracks. |

## State Vector

The full Wheelbot state is 16D, split into variant and invariant components:

```
Variant s^V, learned by the model:
  [roll, pitch, yaw_rate, roll_rate, pitch_rate,
   drive_angle_rate, balance_angle_rate,
   body_vx, body_vy, body_vz, z]

Invariant s^I, global odometry:
  [yaw, drive_angle, balance_angle, x, y]
```

The Infoprop model is trained on the concatenated state `[s^V, s^I]`.

The robot's global XY position, yaw, and wheel angles are invariant to the learned dynamics, so model rollouts can start from arbitrary positions on any pre-generated track rather than only from states visited in real data.

Partial observability is handled by conditioning the dynamics model on a history of past states and actions. The default history lengths are configured by `obs_history` and `act_history` in `infoprop_jax/config/env/wheelbot.yaml`.

## Observation And Reward

The agent observes trajectory features, a masked 10D robot state, height, and body-frame velocity. Trajectory features include cross-track error, cross-angle error, and lookahead waypoint distances/angles.

The action is 2D continuous torque `[tau_drive, tau_balance]`. The RL action is added to a linear balancing prior with gains from `infoprop_jax/config/env/wheelbot.yaml`.

The reward combines cross-track alignment, cross-angle alignment, longitudinal velocity, and crash penalty. Weights and noise settings live in `infoprop_jax/config/env/wheelbot.yaml`.
