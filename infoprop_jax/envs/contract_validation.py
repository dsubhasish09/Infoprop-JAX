# Copyright (c) 2026 Devdutt Subhasish
# SPDX-License-Identifier: MIT
"""Startup validation of the `InfopropWrappable` contract.

`validate_infoprop_contract(env)` is called once at the top of `infoprop.train`.
Every hook is exercised under `jax.eval_shape`, so MJX reset/step are only traced
(milliseconds, no physics compute) while still yielding exactly the information the
training loop depends on: output shapes and pytree structures. The scan-critical
check is #5 — `postprocess` and `reset_from_buffer` must produce structurally
identical `State`s, because they are carried together under `lax.scan`; a mismatch
there otherwise surfaces as an opaque tracer error deep inside the rollout scan.
"""

import jax
import jax.numpy as jnp
from jax import numpy as jp


def _specs(tree):
  """Map a pytree to {path: (shape, dtype)} using concrete or eval_shape leaves."""
  leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
  return {
      jax.tree_util.keystr(path): (jnp.shape(leaf), jnp.result_type(leaf))
      for path, leaf in leaves
  }


def _compare_trees(name_a, tree_a, name_b, tree_b, hook):
  specs_a, specs_b = _specs(tree_a), _specs(tree_b)
  problems = []
  for path in sorted(set(specs_a) | set(specs_b)):
    if path not in specs_a:
      problems.append(f'  {path}: missing from {name_a}')
    elif path not in specs_b:
      problems.append(f'  {path}: missing from {name_b}')
    elif specs_a[path][0] != specs_b[path][0]:
      problems.append(
          f'  {path}: shape {specs_a[path][0]} ({name_a}) != '
          f'{specs_b[path][0]} ({name_b})')
  if problems:
    raise ValueError(
        f'Infoprop contract violation in `{hook}`: {name_a} and {name_b} '
        'must have identical pytree structure and leaf shapes '
        '(they are carried together under scan):\n' + '\n'.join(problems))


def _check(condition, hook, message):
  if not condition:
    raise ValueError(f'Infoprop contract violation in `{hook}`: {message}')


def validate_infoprop_contract(env):
  """Validate the InfopropWrappable hooks of the (unwrapped) real env.

  Raises ValueError naming the offending hook and leaf on the first failure.
  """
  ms, cs, fs = env.model_state_size, env.context_size, env.full_state_size
  oh, ah = env.obs_history, env.act_history
  act = env.action_size
  rng_struct = jax.ShapeDtypeStruct((2,), jnp.uint32)

  # 1. dummy_physics_transition schema (concrete zeros, sizes the physics buffer).
  dummy = env.dummy_physics_transition
  _check(fs == ms + cs, 'full_state_size',
         f'full_state_size ({fs}) != model_state_size ({ms}) + '
         f'context_size ({cs})')
  _check(
      jnp.shape(dummy.observation) == (ms * oh + act * ah,),
      'dummy_physics_transition',
      f'observation shape {jnp.shape(dummy.observation)} != '
      f'(model_state_size*obs_history + action_size*act_history,) = '
      f'({ms * oh + act * ah},)')
  _check(
      jnp.shape(dummy.next_observation) == (ms,), 'dummy_physics_transition',
      f'next_observation shape {jnp.shape(dummy.next_observation)} != '
      f'(model_state_size,) = ({ms},)')
  _check('truncation' in dummy.extras['state_extras'],
         'dummy_physics_transition',
         "extras['state_extras'] must declare 'truncation' (actor_step "
         'records it on every real transition)')

  # 2. context_from_transition shape.
  ctx = env.context_from_transition(dummy)
  _check(
      jnp.shape(ctx) == (cs,), 'context_from_transition',
      f'returned shape {jnp.shape(ctx)} != (context_size,) = ({cs},)')

  # 3. reset_from_buffer: framework keys, carry keys, obs size.
  s0_spec = jax.eval_shape(env.reset_from_buffer, rng_struct, dummy)
  for key in ('accumulated_conditional_entropy', 'current_conditional_entropy'):
    _check(key in s0_spec.info, 'reset_from_buffer',
           f"info must contain '{key}' with shape (full_state_size,)")
    _check(s0_spec.info[key].shape == (fs,), 'reset_from_buffer',
           f"info['{key}'] shape {s0_spec.info[key].shape} != "
           f'(full_state_size,) = ({fs},)')
  missing = set(env.reset_carry_keys) - set(s0_spec.info)
  _check(not missing, 'reset_from_buffer',
         f'info is missing reset_carry_keys {sorted(missing)}')
  _check(s0_spec.obs.shape == (env.observation_size,), 'reset_from_buffer',
         f'obs shape {s0_spec.obs.shape} != (observation_size,) = '
         f'({env.observation_size},)')

  # Zero-filled materialization of the buffer-reset state for the hooks below
  # (None leaves, e.g. a skipped pipeline_state, pass through tree_map untouched).
  s0 = jax.tree_util.tree_map(lambda s: jp.zeros(s.shape, s.dtype), s0_spec)

  # 4. preprocess output shapes.
  pre = jax.eval_shape(env.preprocess, s0, jnp.zeros(act))
  _check(
      isinstance(pre, tuple) and len(pre) == 5, 'preprocess',
      'must return the 5-tuple (nn_input, curr_model_state, curr_context, '
      'applied_action, processed_action)')
  expected = (jnp.shape(dummy.observation), (ms,), (cs,), (act,), (act,))
  names = ('nn_input', 'curr_model_state', 'curr_context', 'applied_action',
           'processed_action')
  for out, shape, name in zip(pre, expected, names):
    _check(out.shape == shape, 'preprocess',
           f'{name} shape {out.shape} != {shape}')

  # 5. postprocess State structure == reset_from_buffer State structure.
  s1_spec = jax.eval_shape(env.postprocess, s0, jnp.zeros(act), jnp.zeros(ms),
                           jnp.zeros(cs), jnp.zeros(act))
  _compare_trees('postprocess output', s1_spec, 'reset_from_buffer output',
                 s0_spec, 'postprocess')

  # 6. augment_prediction maps [E, model_state] -> [E, full_state].
  member = jax.ShapeDtypeStruct((2, ms), jnp.float32)
  aug = jax.eval_shape(env.augment_prediction, member, member,
                       jnp.zeros(ms), jnp.zeros(cs))
  for out, name in zip(aug, ('mean', 'var')):
    _check(out.shape == (2, fs), 'augment_prediction',
           f'{name} shape {out.shape} != [E, full_state_size] = (2, {fs})')

  # 7. extract_physics_transition output matches the dummy transition, using a
  # real reset state with the EpisodeWrapper-provided keys simulated.
  s_real_spec = jax.eval_shape(env.reset, rng_struct)
  missing = set(env.reset_carry_keys) - set(s_real_spec.info)
  _check(not missing, 'reset',
         f'info is missing reset_carry_keys {sorted(missing)}')
  s_real = jax.tree_util.tree_map(lambda s: jp.zeros(s.shape, s.dtype),
                                  s_real_spec)
  info = dict(s_real.info)
  info.setdefault('truncation', jp.zeros(()))
  info.setdefault('steps', jp.zeros(()))
  s_real = s_real.replace(info=info)
  t_spec = jax.eval_shape(
      lambda a, b: env.extract_physics_transition(a, b, {}), s_real, s_real)
  dummy_cmp = dummy._replace(extras={
      'state_extras': dummy.extras['state_extras'],
      'policy_extras': {},
  })
  _compare_trees('extract_physics_transition output', t_spec,
                 'dummy_physics_transition', dummy_cmp,
                 'extract_physics_transition')
