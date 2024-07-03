import jax.numpy as jnp
from jax import jit
import jax

from functools import partial

class ActorCriticDiscrete:
    def __init__(self, discount):
        self.discount = discount

    # no need to vmap
    def sample_action(self, x, actor, key):
        pred_logits = actor.apply_fn({'params': actor.params}, x)
        action_id = jax.random.categorical(key, pred_logits, -1)
        action_prob = jax.nn.softmax(pred_logits, -1)[jnp.arange(action_id.shape[0]), action_id]

        return action_id, action_prob

    def suggest_action(self, x, actor):
        pred = actor.apply_fn({'params': actor.params}, x)
        action_id = jnp.argmax(pred, axis=-1)

        return action_id

    @partial(jit, static_argnums=(0,))
    def train_one_step(self, actor, critic, obs, expected_return, action_id, mask):

        action_id = jnp.astype(jnp.squeeze(action_id), jnp.int32)

        def masked_mean(x, mask):
            return (x * mask).sum() / mask.sum()

        def loss_func(params):
            actor_params, critic_params = params

            action_logits = actor.apply_fn({'params': actor_params}, obs)
            pred_return = critic.apply_fn({'params': critic_params}, obs)

            action_prob = jax.nn.softmax(action_logits, -1)[jnp.arange(action_id.shape[0]), action_id]
            action_prob = jnp.expand_dims(action_prob, -1)

            advantage = expected_return - pred_return

            critic_loss = masked_mean(jnp.square(advantage), mask) / 2.0

            actor_loss = -masked_mean(jax.lax.stop_gradient(advantage) * jnp.log(action_prob + 1e-10), mask)

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