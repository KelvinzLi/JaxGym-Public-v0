import jax
import jax.numpy as jnp

import flax
from flax import linen as nn  # Linen API
from flax import struct                # Flax dataclasses

@struct.dataclass
class History:
    obs: jax.Array
    reward: jax.Array
    action_id: jax.Array
    mask: jax.Array

@struct.dataclass
class Transition:
    obs: jax.Array

def build_trainer(agent, env, env_params, obs_size, max_episode_steps):
    def episode_body(carry):
        actor, critic, env_state, transition, history, key, t, flag = carry
    
        sample_key, env_key, key = jax.random.split(key, 3)
    
        action_id, action_prob = agent.sample_action(jnp.expand_dims(transition.obs, 0), actor, sample_key)
    
        action_id, action_prob = jnp.squeeze(action_id), jnp.squeeze(action_prob)
    
        obs, env_state, reward, done, _ = env.step(env_key, env_state, action_id, env_params)
    
        next_transition = Transition(obs)
    
        history = History(history.obs.at[t, :].set(transition.obs),
                          history.reward.at[t, :].set(reward),
                          history.action_id.at[t, :].set(action_id),
                          history.mask.at[t, :].set(1),
                          )
    
        t += 1
        flag = jnp.logical_not(jnp.logical_or(t == max_episode_steps, done))
    
        return actor, critic, env_state, next_transition, history, key, t, flag
    
    def fori_body(ii, carry):
        actor, critic, all_rewards, key = carry
    
        reset_key, episode_key, key = jax.random.split(key, 3)
    
        obs, env_state = env.reset(reset_key, env_params)
    
        transition = Transition(obs)
    
        history = History(jnp.zeros((max_episode_steps, obs_size)),
                          jnp.zeros((max_episode_steps, 1)),
                          jnp.zeros((max_episode_steps, 1)),
                          jnp.zeros((max_episode_steps, 1)),
                          )
    
        episode_carry = (actor, critic, env_state, transition, history, key, 0, True)
    
        episode_carry = jax.lax.while_loop(lambda carry: carry[-1], episode_body, episode_carry)
    
        history = episode_carry[4]
    
        _, expected_return = jax.lax.scan(agent.scan_discounted_reward, 0, history.reward, reverse = True)
    
        actor, critic, loss, aux = agent.train_one_step(actor, critic, history.obs, expected_return, history.action_id, history.mask)
    
        all_rewards = all_rewards.at[ii, :].set(jnp.sum(history.reward))
    
        return actor, critic, all_rewards, key

    return fori_body