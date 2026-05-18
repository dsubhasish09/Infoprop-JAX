# NOTE: These registrations are disabled because they reference non-existent modules.
# The main training loop uses direct environment instantiation from:
#   - wheelbot_sim_python.envs.wheelbot_brax_mjx (ground-truth MJX environment)
#   - wheelbot_sim_python.envs.wheelbot_brax_infoprop (model-based environment)
# If you need gymnasium-compatible wrappers, add WheelbotEnv implementations to envs/wheelbot.py
#
# from gymnasium.envs.registration import register
# 
# register(
#     id='wheelbot-v0',
#     entry_point='wheelbot_sim_python.envs.wheelbot:WheelbotEnv',
# )
# 
# register(
#     id='wheelbot-v1',
#     entry_point='wheelbot_sim_python.envs.wheelbot:WheelbotEnv_V1',
# )
# 
# register(
#     id='wheelbot-v2',
#     entry_point='wheelbot_sim_python.envs.wheelbot:WheelbotEnv_V2',
# )
# 
# register(
#     id='wheelbot-v3',
#     entry_point='wheelbot_sim_python.envs.wheelbot:WheelbotEnv_V3',
# )