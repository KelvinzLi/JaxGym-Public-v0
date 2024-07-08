import jax.numpy as jnp
from jax import jit
import jax

from functools import partial

from agents.policy_gradient_agent import PolicyGradient, ActorCriticDiscrete, ActorCriticContinuous

class PPO(PolicyGradient):

    def __init__(self, advantage_estimator, clip_ratio, ppo_steps):
        super().__init__(advantage_estimator)
        self.clip_ratio = clip_ratio
        self.ppo_steps = ppo_steps
    
    @partial(jit, static_argnums=(0,))
    def train_one_step(self, actor, critic, obs, reward, action, done):

        old_pred_return, old_log_action_prob = self.calculate_loss_components(actor, critic, actor.params, critic.params, obs, action)

        old_advantage = jax.vmap(self.advantage_estimator, in_axes = (0, 0, 0))(old_pred_return, reward, done)

        def ppo_update_step(ii, carry):
            def loss_func(params):
                actor_params, critic_params = params

                pred_return, log_action_prob = self.calculate_loss_components(actor, critic, actor_params, critic_params, obs, action)

                advantage = jax.vmap(self.advantage_estimator, in_axes = (0, 0, 0))(pred_return, reward, done)

                critic_loss = jnp.square(advantage).mean() / 2.0

                threshold = jnp.where(old_advantage > 0, 1 + self.clip_ratio, 1 - self.clip_ratio)
                ratio = jnp.exp(log_action_prob - old_log_action_prob) # action_prob / old_action_prob

                surrogate_loss = -jnp.min(jnp.concat([ratio * old_advantage, threshold * old_advantage], axis = -1), axis = -1, keepdims = True)

                actor_loss = surrogate_loss.mean()

                return actor_loss + critic_loss, actor_loss

            actor, critic, cumulative_loss, cumulative_aux = carry

            grad_fn = jax.value_and_grad(loss_func, has_aux = True)
            (loss, aux), (actor_grads, critic_grads) = grad_fn((actor.params, critic.params))

            aux = actor_grads['layers_0']['layers']['layers_0']['kernel'][0, 0]

            actor = actor.apply_gradients(grads=actor_grads)
            critic = critic.apply_gradients(grads=critic_grads)

            carry = (actor, critic, cumulative_loss + loss, cumulative_aux + aux)

            return carry

        carry = (actor, critic, 0, 0)
        carry = jax.lax.fori_loop(0, self.ppo_steps, ppo_update_step, carry)
        actor, critic, loss, aux = carry

        return actor, critic, loss, aux

class PPODiscrete(PPO, ActorCriticDiscrete):
    pass

class PPOContinuous(PPO, ActorCriticContinuous):
    pass