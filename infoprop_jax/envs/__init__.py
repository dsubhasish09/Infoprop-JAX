"""Infoprop Brax environments.

Importing this package registers every bundled environment with Brax's global
registry, so they can be instantiated by name via ``brax.envs.get_environment``.

To add a new environment:
  1. Implement it as an ``InfopropWrappable`` subclass in its own subpackage and
     export the class from that subpackage's ``__init__.py``.
  2. Add a ``name -> class`` entry to ``ENV_REGISTRY`` below.

Nothing else needs to change: importing ``infoprop_jax.envs`` (which the training
entry point does) will register it automatically.
"""
from brax import envs

from infoprop_jax.envs.humanoid import HumanoidEnv
from infoprop_jax.envs.wheelbot import WheelbotEnv

# Maps the config-facing environment name to its Brax env class.
# This is the single place to register new environments.
ENV_REGISTRY = {
    "wheelbot": WheelbotEnv,
    "humanoid": HumanoidEnv,
}


def register_envs():
    """Register all bundled environments with Brax's global registry.

    Idempotent: re-registering an existing name simply overwrites it with the
    same class, so calling this more than once is safe.
    """
    for name, env_cls in ENV_REGISTRY.items():
        envs.register_environment(name, env_cls)


# Register on import so that ``import infoprop_jax.envs`` is sufficient for
# ``brax.envs.get_environment(<name>)`` to resolve.
register_envs()

__all__ = ["ENV_REGISTRY", "register_envs"]
