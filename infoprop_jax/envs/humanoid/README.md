# Humanoid

`HumanoidEnv` is the MJX humanoid task from the MuJoCo MJX tutorial, adapted to
Infoprop by implementing the `InfopropWrappable` methods.

- Real physics follows the tutorial: MuJoCo's bundled
  `mjx/test_data/humanoid/humanoid.xml`, CG solver, 5 physics steps per control
  step by default.
- The Infoprop model state is a Wheelbot-style local floating-base representation:
  `[z, roll, pitch, yaw_rate, roll_rate, pitch_rate, body_vel, joint_qpos, joint_qvel]`.
- `context_size = 3`, carrying integrated odometry `[yaw, x, y]`.
- Policy observations omit MJX-derived fields (`cinert`, `cvel`,
  `qfrc_actuator`), so model rollouts can use the fast path.

Use it with Hydra:

```bash
python -m infoprop_jax.main env=humanoid
```

## HumanoidRace

`HumanoidRaceEnv` (in `humanoid_race_mjx.py`) puts the same humanoid on the
Wheelbot's racing tracks. It subclasses `HumanoidEnv` and inherits the physics-state
representation and all track-agnostic Infoprop methods (`preprocess`,
`augment_prediction`, `postprocess`, `reset_from_buffer`); only observation, reward
and the reset protocol are overridden.

- Tracks: the same 200 pre-generated Wheelbot tracks, scaled by the ratio of the
  two robots' nominal body-centre heights (`race_track.py`), so the humanoid races
  tracks of equivalent relative size.
- Observation: `[trajectory state, humanoid observation minus yaw]` - the
  trajectory state holds cross-track error, cross-angle error and lookahead
  waypoints (reusing `../wheelbot/trajectory.py`); yaw is replaced by the
  trajectory features, x/y are excluded as in the plain humanoid.
- Reward (Wheelbot racing form): cross-track + cross-angle terms plus
  `driving_weight` times the forward velocity projected onto the track direction;
  going off-track or leaving `healthy_z_range` terminates with `crash_penalty`.

```bash
python -m infoprop_jax.main env=humanoid_race
```

## File map

| Path | Role |
|---|---|
| `humanoid_mjx.py` | `HumanoidEnv`: MJX tutorial humanoid + Infoprop methods. |
| `humanoid_race_mjx.py` | `HumanoidRaceEnv`: racing variant (observation/reward/reset overrides). |
| `race_track.py` | Wheelbot tracks scaled to humanoid proportions; shared trajectory data. |
