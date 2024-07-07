import jax.numpy as jnp
from jax import jit
import jax

from functools import partial

class ActorCriticDiscrete:
    def __init__(self, advantage_estimator):
        self.advantage_estimator = advantage_estimator

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
    def train_one_step(self, actor, critic, obs, reward, action_id, done):

        action_id = jnp.astype(jnp.squeeze(action_id), jnp.int32)

        def loss_func(params):
            actor_params, critic_params = params

            action_logits = actor.apply_fn({'params': actor_params}, obs)
            pred_return = critic.apply_fn({'params': critic_params}, obs)

            action_prob = jax.nn.softmax(action_logits, -1)
            action_prob = jax.vmap(lambda x, id: x[jnp.arange(id.shape[0]), id],
                                   in_axes = (0, 0))(
                                       jax.nn.softmax(action_logits, -1), action_id
                                       )
            action_prob = jnp.expand_dims(action_prob, -1)

            advantage = jax.vmap(self.advantage_estimator, in_axes = (0, 0, 0))(pred_return, reward, done)

            critic_loss = jnp.square(advantage).mean() / 2.0

            actor_loss = -(jax.lax.stop_gradient(advantage) * jnp.log(action_prob + 1e-10)).mean()

            return actor_loss + critic_loss, actor_loss

        grad_fn = jax.value_and_grad(loss_func, has_aux = True)
        (loss, aux), (actor_grads, critic_grads) = grad_fn((actor.params, critic.params))

        # aux = actor_grads['layers_0']['Dense_0']['kernel'][0]

        actor = actor.apply_gradients(grads=actor_grads)
        critic = critic.apply_gradients(grads=critic_grads)

        return actor, critic, loss, aux