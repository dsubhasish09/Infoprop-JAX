## Plan: Compact wheelbot agent state

Update the agent-facing observation in both wheelbot environments so it stops carrying zeroed placeholders and instead exposes the useful physics values the policy can act on: keep the seven non-masked robot-state dynamics terms, append body-frame x/y/z velocity, and append the body z position. Keep the internal physics-state and invariant-state layouts stable unless a validation step proves they also need to change. This keeps the change focused on the policy input while still propagating the new observation width through controller slices, termination checks, training init, and docs.

**Steps**
1. Define one canonical compact agent-state layout and apply it in both envs. The current masked 10D tail should be replaced with an explicit 11D tail containing the seven meaningful robot-state entries plus body-frame velocity components and z, with the same ordering used everywhere in the repo. *Depends on none.*
2. Rewrite the observation builders in `wheelbot_brax_mjx.py` and `wheelbot_brax_infoprop.py` to construct that compact tail directly from the measured physics data instead of multiplying by `robot_state_mask`. Update any helper signatures if needed so reset and step both call the same construction path. *Depends on step 1.*
3. Update all slice-based control and termination logic that currently assumes a zero-padded robot tail. That includes `_obs_slice_start` consumers, the rollout torque helpers, and the `max_state` comparison so they index the compact tail correctly and do not rely on empty slots. *Depends on step 2.*
4. Expand the env config in `config/env/wheelbot.yaml` so `max_state` matches the new 11D agent-state tail and preserves the intended safety thresholds for the retained dynamics terms and the added z/velocity features. *Depends on step 3.*
5. Propagate the new observation width to downstream code that constructs SAC networks, normalizers, replay samples, and evaluation policies. Most of this should continue to work because the code already reads `env.observation_size`, but the plan should explicitly verify `infoprop.py` and `video_eval.py` still infer the new width correctly after the env change. *Depends on step 2.*
6. Refresh documentation in `envs/README.md` so it describes the new observation tail and no longer mentions zeroed masked slots. If any checkpoint or replay artifact becomes shape-incompatible, document that the new training run is not backward-compatible with older policy checkpoints. *Depends on step 2.*

**Relevant files**
- `/home/ku537617/DSME/wheelbot-racing/wheelbot_sim_python/envs/wheelbot_brax_mjx.py` — update observation construction, control slices, and reward/done slicing.
- `/home/ku537617/DSME/wheelbot-racing/wheelbot_sim_python/envs/wheelbot_brax_infoprop.py` — mirror the same agent-state layout and slice logic for model-rollout training.
- `/home/ku537617/DSME/wheelbot-racing/wheelbot_sim_python/config/env/wheelbot.yaml` — extend `max_state` to the new tail length and semantics.
- `/home/ku537617/DSME/wheelbot-racing/wheelbot_sim_python/algorithms/infoprop.py` — verify training init, normalizer state, and model rollout code consume `env.observation_size` and do not hardcode the old width.
- `/home/ku537617/DSME/wheelbot-racing/wheelbot_sim_python/eval_scripts/video_eval.py` — verify policy-size inference and checkpoint loading still work with the new observation width.
- `/home/ku537617/DSME/wheelbot-racing/wheelbot_sim_python/envs/README.md` — update the observation/state documentation.

**Verification**
1. Run a focused smoke check that instantiates both envs with the wheelbot config, calls `reset`, and asserts the observation length matches the new compact tail plus trajectory features.
2. Step each env once and confirm the reward, done check, and torque-prior helpers still index the intended state components with the new observation layout.
3. Run a narrow training-init sanity check for `infoprop.train` so SAC network creation, running-statistics init, and replay-buffer dummy samples all match `env.observation_size`.
4. Run the evaluation size-inference path in `video_eval.py` against a fresh checkpoint or a minimal mock of the policy params to confirm it still reconstructs the correct observation width.

**Decisions**
- Keep the internal physics-state and invariant-state representations stable for now; the change is only to the policy-facing observation tail and its dependent slices.
- Treat older policy checkpoints and logged replay data as shape-incompatible unless a compatibility shim is explicitly needed later.
- Use a single compact tail ordering everywhere instead of preserving zero placeholders, so the new observation is easier to reason about and less wasteful.
