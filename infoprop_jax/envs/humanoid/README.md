# Humanoid

`HumanoidEnv` is the MJX humanoid task from the MuJoCo MJX tutorial, adapted to
the `InfopropWrappable` contract.

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
