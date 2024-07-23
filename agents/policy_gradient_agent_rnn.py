import jax.numpy as jnp
from jax import jit
import jax

from functools import partial

from agents.policy_gradient_agent import PolicyGradient



class ActorCriticRNNDiscrete(PolicyGradient):
    # no need to vmap
    def sample_action(self, x, carry, done, actor, key):
        carry, pred_logits = actor.apply_fn({'params': actor.params}, x, done, carry)
        action_id = jax.random.categorical(key, pred_logits, -1)
        action_prob = jax.nn.softmax(pred_logits, -1)[jnp.arange(action_id.shape[0]), action_id]

        return carry, (action_id, action_prob)

    def suggest_action(self, x, carry, done, actor):
        carry, pred = actor.apply_fn({'params': actor.params}, x, done, carry)
        action_id = jnp.argmax(pred, axis=-1)

        return carry, action_id

    @partial(jit, static_argnums=(0,))
    def calculate_loss_components(self, actor, critic, actor_params, critic_params, obs, action_id, done):
        # action_id = jnp.astype(jnp.squeeze(action_id), jnp.int32)
        action_id = jnp.squeeze(action_id).astype(jnp.int32)
        
        _, action_logits = actor.apply_fn({'params': actor_params}, obs, done)
        _, pred_return = critic.apply_fn({'params': critic_params}, obs, done)

        action_prob = jax.nn.softmax(action_logits, -1)
        action_prob = jax.vmap(lambda x, id: x[jnp.arange(id.shape[0]), id],
                               in_axes = (0, 0))(
                                   jax.nn.softmax(action_logits, -1), action_id
                                   )
        action_prob = jnp.expand_dims(action_prob, -1)

        return pred_return, jnp.log(action_prob + 1e-10)



def normal_pdf(x, mean, std):
    return jnp.exp(-jnp.square((x - mean) / std) / 2) / jnp.sqrt(2 * jnp.pi) / std

class ActorCriticRNNContinuous(PolicyGradient):
    # no need to vmap
    def sample_action(self, x, carry, done, actor, key):
        carry, (mean, std) = actor.apply_fn({'params': actor.params}, x, done, carry)

        z = jax.random.normal(key)
        action = mean + z * std
        action_prob = normal_pdf(action, mean, std)

        return carry, (action, action_prob)

    def suggest_action(self, x, carry, done, actor):
        carry, (mean, std) = actor.apply_fn({'params': actor.params}, x, done, carry)

        return carry, mean

    @partial(jit, static_argnums=(0,))
    def calculate_loss_components(self, actor, critic, actor_params, critic_params, obs, action, done):
        _, (action_mean, action_std) = actor.apply_fn({'params': actor_params}, obs, done)
        _, pred_return = critic.apply_fn({'params': critic_params}, obs, done)

        log_action_prob = jax.scipy.stats.norm.logpdf(action, action_mean, action_std)
        
        return pred_return, log_action_prob