import jax.numpy as jnp
from jax import jit
import jax

from functools import partial

### Based on the TD3 implementation

def update_target_params(old_params, new_params, polyak_coef):
    def update_param_dict(old_dict, new_dict):
        return {key: update_target_params(old_dict[key], new_dict[key], polyak_coef) for key in old_dict.keys()}

    if isinstance(old_params, dict):
        return update_param_dict(old_params, new_params)
    else:
        return old_params * polyak_coef + new_params * (1 - polyak_coef)

def normal_pdf(x, mean, std):
    return jnp.exp(-jnp.square((x - mean) / (std + 1e-10)) / 2) / jnp.sqrt(2 * jnp.pi) / (std + 1e-10)

class SAC:
    def __init__(self, discount, polyak_coef, entropy_alpha, action_limits = (None, None)):
        self.discount = discount
        self.polyak_coef = polyak_coef
        self.entropy_alpha = entropy_alpha
        
        self.action_limits = action_limits

        self.action_cm = (action_limits[0] + action_limits[1]) / 2
        self.action_halved_diff = (action_limits[1] - action_limits[0]) / 2

    def tanhSquashAction(self, x):
        return self.action_cm + self.action_halved_diff * jnp.tanh((x - self.action_cm) / self.action_halved_diff)

    def sample_action(self, x, actor, key):
        mean, std = actor.apply_fn({'params': actor.params}, x)

        z = jax.random.normal(key)
        action = mean + z * std
        action_prob = normal_pdf(z, 0, 1)

        return self.tanhSquashAction(action)

    def suggest_action(self, x, actor):
        mean, std = actor.apply_fn({'params': actor.params}, x)

        return mean

    @partial(jit, static_argnums=(0,))
    def train_one_step(self, actor, critic, target_actor_params, target_critic_params, obs, next_obs, reward, action, done, 
                       key, *args, **kwargs):

        critic_1, critic_2 = critic
        target_critic_params_1, target_critic_params_2 = target_critic_params

        action_key, next_action_key = jax.random.split(key, 2)

        def loss_func(params):
            actor_params, critic_params_1, critic_params_2 = params

            ########## Critic (Q-function) loss calculation

            pred_og_return_1 = critic_1.apply_fn({'params': critic_params_1}, obs, action)
            pred_og_return_2 = critic_2.apply_fn({'params': critic_params_2}, obs, action)
            
            pred_next_action_mean, pred_next_action_std = actor.apply_fn({'params': jax.lax.stop_gradient(actor_params)}, next_obs)
            z = jax.random.normal(next_action_key)
            pred_next_action = self.tanhSquashAction(pred_next_action_mean + z * pred_next_action_std)
            pred_next_action_prob = normal_pdf(pred_next_action, pred_next_action_mean, pred_next_action_std)

            pred_next_return_1 = critic_1.apply_fn({'params': target_critic_params_1}, next_obs, pred_next_action)
            pred_next_return_2 = critic_2.apply_fn({'params': target_critic_params_2}, next_obs, pred_next_action)
            pred_next_return = jnp.where(pred_next_return_1 < pred_next_return_2, pred_next_return_1, pred_next_return_2)

            next_entropy = -jnp.log(pred_next_action_prob + 1e-10)

            target_q = reward + self.discount * (1 - done) * (pred_next_return + self.entropy_alpha * next_entropy)

            td_error_1 = pred_og_return_1 - target_q
            td_error_2 = pred_og_return_2 - target_q

            critic_loss_1 = jnp.square(td_error_1).mean()
            critic_loss_2 = jnp.square(td_error_2).mean()

            ########## Actor (policy) loss calculation

            pred_action_mean, pred_action_std = actor.apply_fn({'params': actor_params}, obs)
            z = jax.random.normal(action_key)
            pred_action = self.tanhSquashAction(pred_action_mean + z * pred_action_std)
            pred_action_prob = normal_pdf(pred_action, pred_action_mean, pred_action_std)

            pred_return_1 = critic_1.apply_fn({'params': jax.lax.stop_gradient(critic_params_1)}, obs, pred_action)
            pred_return_2 = critic_2.apply_fn({'params': jax.lax.stop_gradient(critic_params_2)}, obs, pred_action)
            pred_return = jnp.where(pred_return_1 < pred_return_2, pred_return_1, pred_return_2)

            entropy = -jnp.log(pred_action_prob + 1e-10)

            actor_loss = -(pred_return + self.entropy_alpha * entropy).mean()

            return actor_loss + critic_loss_1 + critic_loss_2, actor_loss

        grad_fn = jax.value_and_grad(loss_func, has_aux = True)
        (loss, aux), (actor_grads, critic_grads_1, critic_grads_2) = grad_fn((actor.params, critic_1.params, critic_2.params))

        aux = actor_grads['layers_0']['layers']['layers_0']['kernel'][0, 0]

        actor = actor.apply_gradients(grads=actor_grads)
        critic_1 = critic_1.apply_gradients(grads=critic_grads_1)
        critic_2 = critic_2.apply_gradients(grads=critic_grads_2)

        target_critic_params_1 = update_target_params(target_critic_params_1, critic_1.params, self.polyak_coef)
        target_critic_params_2 = update_target_params(target_critic_params_2, critic_2.params, self.polyak_coef)

        critic = (critic_1, critic_2)
        target_critic_params = (target_critic_params_1, target_critic_params_2)

        return actor, critic, target_actor_params, target_critic_params, loss, aux