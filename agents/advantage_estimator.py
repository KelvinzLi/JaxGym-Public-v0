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
        advantage = expected_return - pred_returns

        return advantage

class gae_estimator:
    def __init__(self, discount, gae_factor):
        self.discount = discount
        self.gae_factor = gae_factor
        
    @partial(jit, static_argnums=(0,))
    def __call__(self, pred_returns, rewards, dones):
        def scan_gae(carry, vars):
            (gae, next_pred_return, flag) = carry

            pred_return, reward, done = vars

            td_error = reward + (1 - done) * self.discount * jax.lax.stop_gradient(next_pred_return) - pred_return
            gae = (td_error + (1 - done) * self.discount * self.gae_factor * gae) * flag

            flag = 1 # flag = 0 only for the last element
            
            return (jnp.squeeze(gae), jnp.squeeze(pred_return), flag), gae

        _, advantages = jax.lax.scan(
            scan_gae, 
            (0, 0, 0), 
            (pred_returns, rewards, dones), 
            reverse = True
        )

        return advantages