import jax.numpy as jnp
from jax import jit
import jax

from functools import partial

def normal_pdf(x, mean, std):
    return jnp.exp(-jnp.square((x - mean) / std) / 2) / jnp.sqrt(2 * jnp.pi) / std

class ActorCriticContinuous:
    def __init__(self, discount):
        self.discount = discount

    # no need to vmap
    def sample_action(self, x, actor, key):
        mean, std = actor.apply_fn({'params': actor.params}, x)

        z = jax.random.normal(key)
        action = mean + z * std
        action_prob = normal_pdf(z, 0, 1)

        return action, action_prob

    def suggest_action(self, x, actor):
        mean, std = actor.apply_fn({'params': actor.params}, x)

        return mean

    @partial(jit, static_argnums=(0,))
    def train_one_step(self, actor, critic, obs, expected_return, action, mask):
        def masked_mean(x, mask):
            return (x * mask).sum() / mask.sum()

        def loss_func(params):
            actor_params, critic_params = params

            action_mean, action_std = actor.apply_fn({'params': actor_params}, obs)
            pred_return = critic.apply_fn({'params': critic_params}, obs)

            log_action_prob = jax.scipy.stats.norm.logpdf(action, action_mean, action_std)

            advantage = expected_return - pred_return

            critic_loss = masked_mean(jnp.square(advantage), mask) / 2.0

            actor_loss = -masked_mean(jax.lax.stop_gradient(advantage) * log_action_prob, mask)

            return actor_loss + critic_loss, actor_loss

        grad_fn = jax.value_and_grad(loss_func, has_aux = True)
        (loss, aux), (actor_grads, critic_grads) = grad_fn((actor.params, critic.params))

        # aux = actor_grads['layers_0']['Dense_0']['kernel'][0]

        actor = actor.apply_gradients(grads=actor_grads)
        critic = critic.apply_gradients(grads=critic_grads)

        return actor, critic, loss, aux

    def scan_discounted_reward(self, carry, x):
        v = x + self.discount * carry
        return jnp.squeeze(v), v