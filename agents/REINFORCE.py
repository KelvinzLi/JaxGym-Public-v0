import jax.numpy as jnp
from jax import jit
import jax

from functools import partial

from agents.policy_gradient_agent import PolicyGradient, ActorCriticDiscrete, ActorCriticContinuous

class REINFORCE(PolicyGradient):
    
    @partial(jit, static_argnums=(0,))
    def train_one_step(self, actor, critic, obs, reward, action, done, update_stamp):

        def loss_func(params):
            actor_params, critic_params = params

            pred_return, log_action_prob = self.calculate_loss_components(actor, critic, actor_params, critic_params, obs, action)

            advantage = jax.vmap(self.advantage_estimator, in_axes = (0, 0, 0))(pred_return, reward, done)

            critic_loss = jnp.square(advantage).mean() / 2.0

            actor_loss = -(jax.lax.stop_gradient(advantage) * log_action_prob).mean()

            return actor_loss + critic_loss, (actor_loss, critic_loss)

        grad_fn = jax.value_and_grad(loss_func, has_aux = True)
        (loss, aux_metrics), (actor_grads, critic_grads) = grad_fn((actor.params, critic.params))

        # aux = actor_grads['layers_0']['Dense_0']['kernel'][0]

        actor = actor.apply_gradients(grads=actor_grads)
        critic = critic.apply_gradients(grads=critic_grads)

        actor_loss, critic_loss = aux_metrics

        train_log = {"loss": loss.squeeze(), 
                     "actor_loss": actor_loss.squeeze(), 
                     "critic_loss": critic_loss.squeeze(), 
                    }

        return actor, critic, train_log

class REINFORCEDiscrete(REINFORCE, ActorCriticDiscrete):
    pass

class REINFORCEContinuous(REINFORCE, ActorCriticContinuous):
    pass