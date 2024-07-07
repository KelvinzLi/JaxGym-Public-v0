import jax.numpy as jnp
from jax import jit
import jax

from functools import partial

class PolicyGradient:
    def __init__(self, advantage_estimator):
        self.advantage_estimator = advantage_estimator

    # no need to vmap
    def sample_action(self, x, actor, key):
        raise NotImplementedError()

    def suggest_action(self, x, actor):
        raise NotImplementedError()

    @partial(jit, static_argnums=(0,))
    def calculate_loss_components(self, actor, critic, actor_params, critic_params, obs, action_id):
        raise NotImplementedError()

    @partial(jit, static_argnums=(0,))
    def train_one_step(self, actor, critic, obs, reward, action, done):
        raise NotImplementedError()



class ActorCriticDiscrete(PolicyGradient):
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
    def calculate_loss_components(self, actor, critic, actor_params, critic_params, obs, action_id):
        action_id = jnp.astype(jnp.squeeze(action_id), jnp.int32)
        
        action_logits = actor.apply_fn({'params': actor_params}, obs)
        pred_return = critic.apply_fn({'params': critic_params}, obs)

        action_prob = jax.nn.softmax(action_logits, -1)
        action_prob = jax.vmap(lambda x, id: x[jnp.arange(id.shape[0]), id],
                               in_axes = (0, 0))(
                                   jax.nn.softmax(action_logits, -1), action_id
                                   )
        action_prob = jnp.expand_dims(action_prob, -1)

        return pred_return, jnp.log(action_prob + 1e-10)



def normal_pdf(x, mean, std):
    return jnp.exp(-jnp.square((x - mean) / std) / 2) / jnp.sqrt(2 * jnp.pi) / std

class ActorCriticContinuous(PolicyGradient):
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
    def calculate_loss_components(self, actor, critic, actor_params, critic_params, obs, action):
        action_mean, action_std = actor.apply_fn({'params': actor_params}, obs)
        pred_return = critic.apply_fn({'params': critic_params}, obs)

        log_action_prob = jax.scipy.stats.norm.logpdf(action, action_mean[:, :-1], action_std[:, :-1])
        return pred_return, log_action_prob