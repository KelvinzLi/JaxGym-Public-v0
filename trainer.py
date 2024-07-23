import jax
import jax.numpy as jnp

import flax
from flax import linen as nn  # Linen API
from flax import struct       # Flax dataclasses

from structs import History, Transition

def build_trainer(agent, env, env_params, num_envs, obs_size, action_size, max_episode_steps, callback = None, 
                  use_rnn_agent = False, rnn_carry_initializer = None):

    vmap_env_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    
    def episode_body(t, carry):
        actor, critic, env_state, transition, history, key = carry
    
        sample_key, env_key, key = jax.random.split(key, 3)

        if not use_rnn_agent:
            action, action_prob = agent.sample_action(transition.obs, actor, sample_key)
        else:
            carry, (action, action_prob) = agent.sample_action(jnp.expand_dims(transition.obs, axis = 1), transition.carry, jnp.expand_dims(transition.done, axis = 1), actor, sample_key)

        action = jnp.squeeze(action, axis = -1) # otherwise gives error when action size is 1
        
        vmap_env_key = jax.random.split(env_key, num_envs)
        obs, env_state, reward, done, _ = vmap_env_step(vmap_env_key, env_state, action, env_params)
        reward, action, done = jnp.reshape(reward, (num_envs, 1)), jnp.reshape(action, (num_envs, action_size)), jnp.reshape(done, (num_envs, 1))
    
        next_transition = Transition(obs)

        if use_rnn_agent:
            next_transition = next_transition.replace(done = done, carry = carry)

        history = History(history.obs.at[:, t, :].set(transition.obs),
                          history.reward.at[:, t, :].set(reward),
                          history.action.at[:, t, :].set(action),
                          history.done.at[:, t, :].set(done),
                          )
    
        return actor, critic, env_state, next_transition, history, key
    
    def fori_body(ii, carry):
        actor, critic, logger, key = carry
    
        reset_key, episode_key, key = jax.random.split(key, 3)

        vmap_reset_key = jax.random.split(reset_key, num_envs)
        obs, env_state = vmap_env_reset(vmap_reset_key, env_params)
    
        transition = Transition(obs)

        if use_rnn_agent:
            assert rnn_carry_initializer is not None
            
            transition = transition.replace(done = jnp.ones((num_envs, 1)).astype(bool), # force to reset at the start
                                            carry = rnn_carry_initializer(reset_key), # random carry shape to initiate the process
                                           )
        
        history = History(jnp.zeros((num_envs, max_episode_steps, obs_size)),
                          jnp.zeros((num_envs, max_episode_steps, 1)),
                          jnp.zeros((num_envs, max_episode_steps, action_size)),
                          jnp.zeros((num_envs, max_episode_steps, 1)),
                          )
    
        episode_carry = (actor, critic, env_state, transition, history, key)

        episode_carry = jax.lax.fori_loop(0, max_episode_steps, episode_body, episode_carry)
    
        history = episode_carry[4]

        actor, critic, loss, aux = agent.train_one_step(actor, critic, history.obs, history.reward, history.action, history.done)
    
        logger = logger.at[ii, :].set(history.reward.sum() / (history.done).sum())
        # logger = logger.at[ii, :].set(aux)

        if callback is not None:
            jit_unique = jax.jit(jnp.unique, static_argnames=['size'])
            info_dict = {"Reward": history.reward.sum() / history.done.sum(), "Total reward": history.reward.sum(), "actions": jit_unique(history.action, size=4, fill_value=-1)}
            jax.debug.callback(callback, info_dict)
    
        return actor, critic, logger, key

    return fori_body