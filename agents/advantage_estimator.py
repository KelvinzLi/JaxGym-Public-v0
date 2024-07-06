import jax.numpy as jnp
from jax import jit
import jax

from functools import partial

class state_value_estimator:
    def __init__(self, discount):
        self.discount = discount

    def scan_discounted_reward(self, carry, x):
        reward, done = x
        v = reward + (1 - done) * self.discount * carry
        return jnp.squeeze(v), v

    @partial(jit, static_argnums=(0,))
    def __call__(self, pred_returns, rewards, dones):
        _, expected_return = jax.lax.scan(self.scan_discounted_reward, 0, (rewards, dones), reverse = True)
        advantage = expected_return - pred_returns[:-1] # observations and hence pred_returns records the final output observation (corresponding reward not recorded)

        return advantage