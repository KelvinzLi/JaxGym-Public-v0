import jax.numpy as jnp
from jax import jit
import jax

from functools import partial

from agents.policy_gradient_agent_rnn import ActorCriticRNNDiscrete, ActorCriticRNNContinuous
from agents.PPO import PPO

class PPORNNDiscrete(PPO, ActorCriticRNNDiscrete):
    pass

class PPORNNContinuous(PPO, ActorCriticRNNContinuous):
    pass