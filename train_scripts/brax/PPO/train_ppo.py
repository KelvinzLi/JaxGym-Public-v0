import sys

sys.path.append("./")

import jax
import jax.numpy as jnp

import flax
from flax import linen as nn  # Linen API
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
import optax                           # Common loss functions and optimizers

from flax.training.train_state import TrainState

import gymnax
import brax
from brax import envs

import matplotlib.pyplot as plt

from models.base_model import BaseModel, NormalDistPredictor

# from agents.REINFORCE_continuous import ActorCriticContinuous
from agents.PPO_v2 import PPOContinuous
from agents.advantage_estimator import expected_return_estimator, gae_estimator

from trainer_v2 import build_trainer

from utils.callbacks import versatile_callback_v2

# https://github.com/luchris429/purejaxrl/blob/5343613b07b3bc543c49695df601fc40f5ec3062/purejaxrl/wrappers.py#L117

from gymnax.environments import environment, spaces
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper

class BraxGymnaxWrapper:
    def __init__(self, env_name, backend="positional"):
        env = envs.get_environment(env_name=env_name, backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


def train(config):
    actor_lr = config["actor_lr"]
    critic_lr = config["critic_lr"]

    discount = config["discount"]

    clip_ratio = config["clip_ratio"]
    ppo_steps = config["ppo_steps"]

    num_envs = config["num_envs"]

    update_iters = config["update_iters"]

    train_rollout_steps = config["train_rollout_steps"]
    eval_rollout_steps = config["eval_rollout_steps"]
    eval_every = config["eval_every"]

    env_name = config["env_name"]

    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]

    gae_factor = config["gae_factor"]

    #########################################################

    backend = "positional"

    env, env_params = BraxGymnaxWrapper(env_name), None

    obs_size = env.observation_space(env_params).shape
    action_size = env.action_space(env_params).shape[0]

    #########################################################

    key = jax.random.PRNGKey(64)

    dummy = jnp.ones([1, *obs_size])
    actor_init_key, critic_init_key, key = jax.random.split(key, 3)

    actor_model = nn.Sequential([BaseModel(hidden_size = hidden_size, num_layers = num_layers), NormalDistPredictor(output_size = action_size, logvar_init_value = 0, limits = (-1, 1))])
    actor_params = actor_model.init(actor_init_key, dummy)['params']
    actor_tx = optax.chain(
       optax.clip_by_global_norm(0.5),
       optax.adam(actor_lr),
    )
    actor = TrainState.create(apply_fn=actor_model.apply,
                                params=actor_params,
                                tx=actor_tx,
                                )

    critic_model = nn.Sequential([BaseModel(hidden_size = hidden_size, num_layers = num_layers), nn.Dense(features = 1)])
    critic_params = critic_model.init(critic_init_key, dummy)['params']
    critic_tx = optax.chain(
       optax.clip_by_global_norm(0.5),
       optax.adam(critic_lr),
    )
    critic = TrainState.create(apply_fn=critic_model.apply,
                                params=critic_params,
                                tx=critic_tx,
                                )

    agent = PPOContinuous(gae_estimator(discount, gae_factor),
                            expected_return_estimator(discount),
                            clip_ratio, ppo_steps)

    callback = versatile_callback_v2(update_iters, tqdm_keys = ["episode_reward"], split_train_eval = True)

    train = build_trainer(agent, env, env_params, num_envs, obs_size, action_size, 
                            train_rollout_steps, eval_rollout_steps, eval_every, 
                            callback, 
                            )

    actor, critic = train(update_iters, actor, critic, key)

    return callback.eval_history