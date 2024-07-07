import jax.numpy as jnp
from jax import jit
import jax

from functools import partial

def normal_pdf(x, mean, std):
    return jnp.exp(-jnp.square((x - mean) / std) / 2) / jnp.sqrt(2 * jnp.pi) / std

class ActorCriticContinuous:
    def __init__(self, advantage_estimator):
        self.advantage_estimator = advantage_estimator

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
    def train_one_step(self, actor, critic, obs, reward, action, done):
        
        def loss_func(params):
            actor_params, critic_params = params

            action_mean, action_std = actor.apply_fn({'params': actor_params}, obs)
            pred_return = critic.apply_fn({'params': critic_params}, obs)

            log_action_prob = jax.scipy.stats.norm.logpdf(action, action_mean[:, :-1], action_std[:, :-1])

            advantage = jax.vmap(self.advantage_estimator, in_axes = (0, 0, 0))(pred_return, reward, done)

            critic_loss = jnp.square(advantage).mean() / 2.0

            actor_loss = -(jax.lax.stop_gradient(advantage) * log_action_prob).mean()

            return actor_loss + critic_loss, actor_loss

        grad_fn = jax.value_and_grad(loss_func, has_aux = True)
        (loss, aux), (actor_grads, critic_grads) = grad_fn((actor.params, critic.params))

        # print(actor_grads)
        aux = actor_grads['layers_0']['layers']['layers_0']['kernel'][0, 0]

        actor = actor.apply_gradients(grads=actor_grads)
        critic = critic.apply_gradients(grads=critic_grads)

        return actor, critic, loss, aux