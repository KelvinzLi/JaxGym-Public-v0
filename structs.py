from flax import struct       # Flax dataclasses

import jax

from typing import Any

################################## for trainer

@struct.dataclass
class History:
    obs: jax.Array
    reward: jax.Array
    action: jax.Array
    done: jax.Array

@struct.dataclass
class Transition:
    obs: jax.Array
    done: jax.Array = None
    carry: Any = None

################################## for sampler

@struct.dataclass
class Buffer:
    obs: jax.Array
    reward: jax.Array
    action: jax.Array
    done: jax.Array
    mask: jax.Array
    head: int = 0

@struct.dataclass
class Sample:
    obs: jax.Array
    next_obs: jax.Array
    reward: jax.Array
    action: jax.Array
    done: jax.Array