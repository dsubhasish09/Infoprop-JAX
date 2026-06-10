# Copyright (c) 2026 Devdutt Subhasish
# SPDX-License-Identifier: MIT
# NOTE: These registrations are disabled because they reference non-existent modules.
# The main training loop uses direct environment instantiation from:
#   - infoprop_jax.envs.wheelbot.wheelbot_brax_mjx (ground-truth MJX environment)
#   - infoprop_jax.envs.infoprop_env (model-based environment)
# If you need gymnasium-compatible wrappers, add WheelbotEnv implementations under envs/wheelbot/.
#
# from gymnasium.envs.registration import register
# 
# register(
#     id='wheelbot-v0',
#     entry_point='infoprop_jax.envs.wheelbot:WheelbotEnv',
# )
# 
# register(
#     id='wheelbot-v1',
#     entry_point='infoprop_jax.envs.wheelbot:WheelbotEnv_V1',
# )
# 
# register(
#     id='wheelbot-v2',
#     entry_point='infoprop_jax.envs.wheelbot:WheelbotEnv_V2',
# )
# 
# register(
#     id='wheelbot-v3',
#     entry_point='infoprop_jax.envs.wheelbot:WheelbotEnv_V3',
# )
