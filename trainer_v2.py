from functools import partial

import jax
import jax.numpy as jnp

import flax
from flax import linen as nn  # Linen API
from flax import struct       # Flax dataclasses

from structs import History, Transition

# from utils.envpool import envpool_to_gymnax_interface

def build_trainer(agent, env, env_params, num_envs, obs_size, action_size, 
                  train_rollout_steps, eval_rollout_steps, eval_every,
                  callback = None, 
                  use_rnn_agent = False, rnn_carry_initializer = None,
                  envpool_format = False,
                 ):

    eval_freq = eval_every * train_rollout_steps

    if not envpool_format:
        vmap_env_reset = jax.vmap(env.reset, in_axes=(0, None))
        vmap_env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    # else:
    #     vmap_env_reset, vmap_env_step = envpool_to_gymnax_interface(env)

    def init_history(rollout_steps):
        history = History(jnp.zeros((num_envs, rollout_steps, *obs_size)),
                          jnp.zeros((num_envs, rollout_steps, 1)),
                          jnp.zeros((num_envs, rollout_steps, action_size)),
                          jnp.zeros((num_envs, rollout_steps, 1)),
                          )
        return history

    @partial(jax.jit, static_argnames=['rollout_steps', 'eval_mode'])
    def _episode_body(t, carry, rollout_steps, eval_mode = False):
        actor, critic, env_state, transition, history, key = carry
    
        sample_key, env_key, key = jax.random.split(key, 3)

        if not use_rnn_agent:
            if not eval_mode:
                action, action_prob = agent.sample_action(transition.obs, actor, sample_key)
            else:
                action = agent.suggest_action(transition.obs, actor)
        else:
            carry, (action, action_prob) = agent.sample_action(jnp.expand_dims(transition.obs, axis = 1), transition.carry, jnp.expand_dims(transition.done, axis = 1), actor, sample_key)

        if action.shape[-1] == 1:
            action = jnp.squeeze(action, axis = -1) # otherwise gives error when action size is 1
        
        vmap_env_key = jax.random.split(env_key, num_envs)
        obs, env_state, reward, done, _ = vmap_env_step(vmap_env_key, env_state, action, env_params)
        reward, action, done = jnp.reshape(reward, (num_envs, 1)), jnp.reshape(action, (num_envs, action_size)), jnp.reshape(done, (num_envs, 1))
    
        next_transition = Transition(obs)

        if use_rnn_agent:
            next_transition = next_transition.replace(done = done, carry = carry)

        history = History(history.obs.at[:, t % rollout_steps, :].set(transition.obs),
                          history.reward.at[:, t % rollout_steps, :].set(reward),
                          history.action.at[:, t % rollout_steps, :].set(action),
                          history.done.at[:, t % rollout_steps, :].set(done),
                          )
    
        return actor, critic, env_state, next_transition, history, key

    def train_episode_body(t, carry):
        carry = _episode_body(t, carry, train_rollout_steps)

        actor, critic, env_state, transition, history, key = carry

        eval_key, key = jax.random.split(key, 2)

        actor, critic, history = jax.lax.cond((t + 1) % train_rollout_steps == 0,
                                             on_policy_update,
                                             lambda _a,_b,_c,_d: (actor, critic, history),
                                             actor, critic, history, t // train_rollout_steps,
                                             )

        jax.lax.cond((t + 1) % eval_freq == 0, 
                     eval,
                     lambda x,y,z: None,
                     actor, critic, key,
                    )

        return actor, critic, env_state, transition, history, key

    def eval_episode_body(t, carry):
        carry = _episode_body(t, carry, eval_rollout_steps, eval_mode = True)
        return carry
    
    def on_policy_update(actor, critic, history, update_stamp):

        actor, critic, train_log = agent.train_one_step(actor, critic, history.obs, history.reward, history.action, history.done, update_stamp = update_stamp)

        if callback is not None:
            jax.debug.callback(callback, train_log)

        history = init_history(train_rollout_steps)
    
        return actor, critic, history

    def init_run(rollout_steps, key):
        reset_key, key = jax.random.split(key, 2)
        
        vmap_reset_key = jax.random.split(reset_key, num_envs)
        obs, env_state = vmap_env_reset(vmap_reset_key, env_params)
    
        transition = Transition(obs)

        if use_rnn_agent:
            assert rnn_carry_initializer is not None
            
            transition = transition.replace(done = jnp.ones((num_envs, 1)).astype(bool), # force to reset at the start
                                            carry = rnn_carry_initializer(reset_key), # random carry shape to initiate the process
                                           )

        history = init_history(rollout_steps)

        return env_state, transition, history

    def _run(iters, actor, critic, key, eval_mode = False):
        init_key, key = jax.random.split(key, 2)

        if not eval_mode:
            rollout_steps, episode_body = train_rollout_steps, train_episode_body
        else:
            rollout_steps, episode_body = eval_rollout_steps, eval_episode_body
            
        env_state, transition, history = init_run(rollout_steps, init_key)
            
        carry = (actor, critic, env_state, transition, history, key)
        
        carry = jax.lax.fori_loop(0, iters, episode_body, carry)

        return carry

    def eval(actor, critic, key):
        carry = _run(eval_rollout_steps, actor, critic, key, eval_mode = True)

        history = carry[4]

        eval_dict = {}

        if callback is not None:
            eval_dict["episode_reward"] = (history.reward.sum() / history.done.sum()).squeeze()

            jax.debug.callback(callback, eval_dict, from_eval = True)

    def train(update_iters, actor, critic, key):
        carry = _run(update_iters * train_rollout_steps, actor, critic, key)

        actor, critic = carry[0], carry[1]

        return actor, critic

    return train