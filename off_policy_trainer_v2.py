import jax
import jax.numpy as jnp

import flax
from flax import linen as nn  # Linen API
from flax import struct       # Flax dataclasses

from structs import History, Transition

def build_trainer(
    agent, sampler, 
    env, env_params, num_envs, 
    obs_size, action_size, 
    max_episode_steps, 
    buffer_size_before_training, 
    update_every, steps_per_update, 
    eval_every, 
    callback = None):

    vmap_env_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    
    def episode_body(t, carry):
        actor, critic, env_state, transition, history, key = carry
    
        sample_key, env_key, key = jax.random.split(key, 3)

        action = agent.sample_action(transition.obs, actor, sample_key)

        vmap_env_key = jax.random.split(env_key, num_envs)
        obs, env_state, reward, done, _ = vmap_env_step(vmap_env_key, env_state, action, env_params)
        reward, action, done = jnp.reshape(reward, (num_envs, 1)), jnp.reshape(action, (num_envs, action_size)), jnp.reshape(done, (num_envs, 1))
    
        next_transition = Transition(obs)

        history = History(history.obs.at[:, t, :].set(transition.obs),
                          history.reward.at[:, t, :].set(reward),
                          history.action.at[:, t, :].set(action),
                          history.done.at[:, t, :].set(done),
                          )
    
        return actor, critic, env_state, next_transition, history, key

    def update_body(ii, carry):
        actor, critic, target_actor_params, target_critic_params, buffer, accumulative_loss, accumulative_aux, key = carry

        sample_key, train_key, key = jax.random.split(key, 3)

        sample = sampler.sample_batch(buffer, sample_key)

        training_output = jax.lax.cond(buffer.mask.sum() > buffer_size_before_training, 
                                       lambda: agent.train_one_step(actor, critic, target_actor_params, target_critic_params, 
                                                                    sample.obs, sample.next_obs, sample.reward, sample.action, sample.done, 
                                                                    step_idx = ii, key = train_key
                                                                   ), 
                                       lambda: (actor, critic, target_actor_params, target_critic_params, 0.0, 0.0)
                                      )
        actor, critic, target_actor_params, target_critic_params, loss, aux = training_output

        accumulative_loss += loss
        accumulative_aux += aux

        return actor, critic, target_actor_params, target_critic_params, buffer, accumulative_loss, accumulative_aux, key

    def eval_rollout(actor, critic, key):
        reset_key, key = jax.random.split(key, 2)
        
        vmap_reset_key = jax.random.split(reset_key, num_envs)
        obs, env_state = vmap_env_reset(vmap_reset_key, env_params)
    
        transition = Transition(obs)
        
        history = History(jnp.zeros((num_envs, max_episode_steps, *obs_size)),
                          jnp.zeros((num_envs, max_episode_steps, 1)),
                          jnp.zeros((num_envs, max_episode_steps, action_size)),
                          jnp.zeros((num_envs, max_episode_steps, 1)),
                          )

        episode_carry = (actor, critic, env_state, transition, history, key)
        
        episode_carry = jax.lax.fori_loop(0, max_episode_steps, episode_body, episode_carry)
    
        history = episode_carry[4]

        return history
    
    def fori_body(ii, carry):
        actor, critic, target_actor_params, target_critic_params, buffer, env_state, transition, logger, key = carry
    
        episode_key, update_key, eval_key, key = jax.random.split(key, 4)
        
        history = History(jnp.zeros((num_envs, 1, *obs_size)),
                          jnp.zeros((num_envs, 1, 1)),
                          jnp.zeros((num_envs, 1, action_size)),
                          jnp.zeros((num_envs, 1, 1)),
                          )
    
        episode_carry = (actor, critic, env_state, transition, history, key)
        
        episode_carry = episode_body(0, episode_carry)
    
        _, _, env_state, transition, history, _ = episode_carry

        buffer = sampler.update_buffer(buffer, history)

        train_carry = (actor, critic, target_actor_params, target_critic_params, buffer, 0.0, 0.0, update_key)
        train_carry = jax.lax.cond(jnp.remainder(ii, update_every) == 0, 
                                   lambda: jax.lax.fori_loop(0, steps_per_update, update_body, train_carry), 
                                   lambda: train_carry)
        actor, critic, target_actor_params, target_critic_params, buffer, loss, aux, _ = train_carry

        def eval(logger):
            history = eval_rollout(actor, critic, eval_key)
            logger = logger.at[ii // eval_every, :].set(history.reward.sum() / (history.done).sum())

            if callback is not None:
                info_dict = {"Reward": history.reward.sum() / (history.done.sum())}
                # info_dict = {"Reward": buffer.mask.sum()}
                jax.debug.callback(callback, info_dict)

            return logger

        logger = jax.lax.cond(ii % eval_every == 0, eval, lambda l: l, logger)
    
        return actor, critic, target_actor_params, target_critic_params, buffer, env_state, transition, logger, key

    def train(iters, actor, critic, target_actor_params, target_critic_params, buffer, logger, key):
        reset_key, key = jax.random.split(key, 2)
        vmap_reset_key = jax.random.split(reset_key, num_envs)
        obs, env_state = vmap_env_reset(vmap_reset_key, env_params)

        transition = Transition(obs)
        
        carry = (actor, critic, target_actor_params, target_critic_params, buffer, env_state, transition, logger, key)
        return jax.lax.fori_loop(0, iters, fori_body, carry)

    return train