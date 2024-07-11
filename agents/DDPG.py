import jax.numpy as jnp
from jax import jit
import jax

from functools import partial

def update_target_params(old_params, new_params, polyak_coef):
    def update_param_dict(old_dict, new_dict):
        return {key: update_target_params(old_dict[key], new_dict[key], polyak_coef) for key in old_dict.keys()}

    if isinstance(old_params, dict):
        return update_param_dict(old_params, new_params)
    else:
        return old_params * polyak_coef + new_params * (1 - polyak_coef)

class DDPG:
    def __init__(self, discount, polyak_coef, noise_scale):
        self.discount = discount
        self.polyak_coef = polyak_coef
        self.noise_scale = noise_scale

    def sample_action(self, x, actor, key):
        action = actor.apply_fn({'params': actor.params}, x)
        noise = self.noise_scale * jax.random.normal(key, action.shape)
        return jnp.clip(action + noise, -2, 2)
    
    def suggest_action(self, x, actor):
        return actor.apply_fn({'params': actor.params}, x)

    @partial(jit, static_argnums=(0,))
    def train_one_step(self, actor, critic, target_actor_params, target_critic_params, obs, next_obs, reward, action, done):

        def loss_func(params):
            actor_params, critic_params = params

            pred_action = actor.apply_fn({'params': actor_params}, obs)
            pred_next_action = actor.apply_fn({'params': target_actor_params}, next_obs)

            pred_og_return = critic.apply_fn({'params': critic_params}, obs, action)
            pred_return = critic.apply_fn({'params': jax.lax.stop_gradient(critic_params)}, obs, pred_action)
            pred_next_return = critic.apply_fn({'params': target_critic_params}, next_obs, pred_next_action)

            td_error = pred_og_return - (reward + self.discount * (1 - done) * pred_next_return)

            print(pred_return.shape)

            critic_loss = jnp.square(td_error).mean()

            actor_loss = -(pred_return).mean()

            return actor_loss + critic_loss, actor_loss

        grad_fn = jax.value_and_grad(loss_func, has_aux = True)
        (loss, aux), (actor_grads, critic_grads) = grad_fn((actor.params, critic.params))

        aux = actor_grads['layers_0']['layers']['layers_0']['kernel'][0, 0]

        actor = actor.apply_gradients(grads=actor_grads)
        critic = critic.apply_gradients(grads=critic_grads)

        target_actor_params = update_target_params(target_actor_params, actor.params, self.polyak_coef)
        target_critic_params = update_target_params(target_critic_params, critic.params, self.polyak_coef)

        return actor, critic, target_actor_params, target_critic_params, loss, aux