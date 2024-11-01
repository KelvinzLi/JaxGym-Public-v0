import jax.numpy as jnp
from jax import jit
import jax

from functools import partial

from agents.policy_gradient_agent import PolicyGradient, ActorCriticDiscrete, ActorCriticContinuous

class PPO(PolicyGradient):

    def __init__(self, advantage_estimator, return_estimator, clip_ratio, ppo_steps, target_kl = 0.01, entropy_coef = 0):
        super().__init__(advantage_estimator)

        self.return_estimator = return_estimator
        
        self.clip_ratio = clip_ratio
        self.ppo_steps = ppo_steps
        self.target_kl = target_kl

        if type(entropy_coef) in (int, float):
            self.entropy_scheduler = lambda: entropy_coef
        else:
            self.entropy_scheduler = entropy_coef
    
    @partial(jit, static_argnums=(0,))
    def train_one_step(self, actor, critic, obs, reward, action, done, update_stamp):

        old_pred_return, old_log_action_prob = self.calculate_loss_components(actor, critic, actor.params, critic.params, obs, action, done = done)

        old_advantage = jax.vmap(self.advantage_estimator, in_axes = (0, 0, 0))(old_pred_return, reward, done)

        old_advantage = (old_advantage - old_advantage.mean()) / old_advantage.std()

        def ppo_update_step(carry):
            def loss_func(params):
                actor_params, critic_params = params

                pred_return, log_action_prob = self.calculate_loss_components(actor, critic, actor_params, critic_params, obs, action, done = done)

                expected_return = jax.vmap(self.return_estimator, in_axes = (0, 0, 0))(pred_return, reward, done)
                expected_return = jax.lax.stop_gradient(expected_return)

                critic_loss = jnp.square(pred_return - expected_return).mean() / 2.0

                threshold = jnp.where(old_advantage > 0, 1 + self.clip_ratio, 1 - self.clip_ratio)
                ratio = jnp.exp(log_action_prob - old_log_action_prob) # action_prob / old_action_prob

                surrogate_loss = -jnp.min(jnp.concatenate([ratio * old_advantage, threshold * old_advantage], axis = -1), axis = -1, keepdims = True)

                entropy = -log_action_prob.mean()

                actor_loss = surrogate_loss.mean() - self.entropy_scheduler() * entropy

                log_r = log_action_prob - old_log_action_prob
                approx_kl = (jnp.exp(log_r) - 1 - log_r).mean()
                # approx_kl = (old_log_action_prob - log_action_prob).mean()

                return actor_loss + critic_loss, (actor_loss, critic_loss, approx_kl, entropy)

            actor, critic, metrics, t = carry
            cumulative_loss, cumulative_actor_loss, cumulative_critic_loss, approx_kl, cumulative_entropy, flag = metrics

            grad_fn = jax.value_and_grad(loss_func, has_aux = True)
            (loss, aux_metrics), (actor_grads, critic_grads) = grad_fn((actor.params, critic.params))

            actor_loss, critic_loss, approx_kl, entropy = aux_metrics

            actor, flag = jax.lax.cond(jnp.greater(1.5 * self.target_kl, approx_kl), 
                                             lambda: (actor.apply_gradients(grads=actor_grads), True),
                                             lambda: (actor, False), 
                                            )
            critic = critic.apply_gradients(grads=critic_grads)

            # aux = actor_grads['layers_0']['layers']['layers_0']['kernel'][0, 0]
            # aux = approx_kl

            flag = flag & jnp.greater(self.ppo_steps, t + 1)

            metrics = (
                cumulative_loss + loss, 
                cumulative_actor_loss + actor_loss, 
                cumulative_critic_loss + critic_loss, 
                approx_kl,
                cumulative_entropy + entropy,
                flag,
            )

            carry = (actor, critic, metrics, t + 1)

            return carry

        metrics = (0, 0, 0, 0, 0, True)
        carry = (actor, critic, metrics, 0)
        carry = jax.lax.while_loop(lambda carry: carry[2][-1], 
                                   ppo_update_step, 
                                   carry,
                                  )
        actor, critic, metrics, t = carry

        cumulative_loss, cumulative_actor_loss, cumulative_critic_loss, approx_kl, cumulative_entropy, flag = metrics

        loss = cumulative_loss / t
        actor_loss = cumulative_actor_loss / t
        critic_loss = cumulative_critic_loss / t
        entropy = cumulative_entropy / t

        train_log = {"loss": loss.squeeze(), 
                     "actor_loss": actor_loss.squeeze(), 
                     "critic_loss": critic_loss.squeeze(), 
                     "kl": approx_kl.squeeze(), 
                     "entropy": entropy.squeeze(), 
                     "ppo_steps": t
                    }
        
        return actor, critic, train_log

class PPODiscrete(PPO, ActorCriticDiscrete):
    pass

class PPOContinuous(PPO, ActorCriticContinuous):
    pass