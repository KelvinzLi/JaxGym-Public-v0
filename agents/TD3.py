import jax.numpy as jnp
from jax import jit
import jax

from functools import partial

### Based on the DDPG implementation

def update_target_params(old_params, new_params, polyak_coef):
    def update_param_dict(old_dict, new_dict):
        return {key: update_target_params(old_dict[key], new_dict[key], polyak_coef) for key in old_dict.keys()}

    if isinstance(old_params, dict):
        return update_param_dict(old_params, new_params)
    else:
        return old_params * polyak_coef + new_params * (1 - polyak_coef)

class TD3:
    def __init__(self, discount, polyak_coef, action_noise_scale, target_noise_scale, noise_clip, policy_delay = 1, action_limits = (None, None)):
        self.discount = discount
        self.polyak_coef = polyak_coef
        self.action_noise_scale = action_noise_scale
        self.target_noise_scale = target_noise_scale
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.action_limits = action_limits

    def sample_action(self, x, actor, key):
        action = actor.apply_fn({'params': actor.params}, x)
        noise = self.action_noise_scale * jax.random.normal(key, action.shape)
        return jnp.clip(action + noise, self.action_limits[0], self.action_limits[1])
    
    def suggest_action(self, x, actor):
        return actor.apply_fn({'params': actor.params}, x)

    @partial(jit, static_argnums=(0,))
    def train_one_step(self, actor, critic, target_actor_params, target_critic_params, obs, next_obs, reward, action, done, 
                       step_idx, key, *args, **kwargs):

        critic_1, critic_2 = critic
        target_critic_params_1, target_critic_params_2 = target_critic_params

        def loss_func(params):
            actor_params, critic_params_1, critic_params_2 = params

            ########## Critic (Q-function) loss calculation

            pred_og_return_1 = critic_1.apply_fn({'params': critic_params_1}, obs, action)
            pred_og_return_2 = critic_2.apply_fn({'params': critic_params_2}, obs, action)
            
            pred_next_action = actor.apply_fn({'params': target_actor_params}, next_obs)
            target_noise = jnp.clip(self.target_noise_scale * jax.random.normal(key, action.shape), -self.noise_clip, self.noise_clip)
            pred_next_action = jnp.clip(pred_next_action + target_noise, self.action_limits[0], self.action_limits[1])
            
            pred_next_return_1 = critic_1.apply_fn({'params': target_critic_params_1}, next_obs, pred_next_action)
            pred_next_return_2 = critic_2.apply_fn({'params': target_critic_params_2}, next_obs, pred_next_action)
            pred_next_return = jnp.where(pred_next_return_1 < pred_next_return_2, pred_next_return_1, pred_next_return_2)

            target_q = reward + self.discount * (1 - done) * pred_next_return

            td_error_1 = pred_og_return_1 - target_q
            td_error_2 = pred_og_return_2 - target_q

            critic_loss_1 = jnp.square(td_error_1).mean()
            critic_loss_2 = jnp.square(td_error_2).mean()

            ########## Actor (policy) loss calculation

            pred_action = actor.apply_fn({'params': actor_params}, obs)
            pred_return = critic_1.apply_fn({'params': jax.lax.stop_gradient(critic_params_1)}, obs, pred_action)

            actor_loss = -(pred_return).mean()

            return actor_loss + critic_loss_1 + critic_loss_2, actor_loss

        grad_fn = jax.value_and_grad(loss_func, has_aux = True)
        (loss, aux), (actor_grads, critic_grads_1, critic_grads_2) = grad_fn((actor.params, critic_1.params, critic_2.params))

        aux = actor_grads['layers_0']['layers']['layers_0']['kernel'][0, 0]

        actor = jax.lax.cond(jnp.remainder(step_idx, self.policy_delay) == 0, lambda: actor.apply_gradients(grads=actor_grads), lambda: actor)
        critic_1 = critic_1.apply_gradients(grads=critic_grads_1)
        critic_2 = critic_2.apply_gradients(grads=critic_grads_2)

        target_actor_params = update_target_params(target_actor_params, actor.params, self.polyak_coef)
        target_critic_params_1 = update_target_params(target_critic_params_1, critic_1.params, self.polyak_coef)
        target_critic_params_2 = update_target_params(target_critic_params_2, critic_2.params, self.polyak_coef)

        critic = (critic_1, critic_2)
        target_critic_params = (target_critic_params_1, target_critic_params_2)

        return actor, critic, target_actor_params, target_critic_params, loss, aux