import jax
import jax.numpy as jnp

import flax
from flax import linen as nn  # Linen API

class ResetLSTMCell(nn.Module):
    hidden_size: int = 128

    def setup(self):
        self.init_carry_0 = self.param('init_carry_0', lambda rng, shape: jnp.zeros(shape), (1, self.hidden_size))
        self.init_carry_1 = self.param('init_carry_1', lambda rng, shape: jnp.zeros(shape), (1, self.hidden_size))

        self.lstm_cell = nn.OptimizedLSTMCell(features = self.hidden_size)

    def __call__(self, carry, x, done):
        # expected shape:
        # carry / init_carry: tuple of (num_envs or 1, hidden_size)
        # done: (num_envs, 1)
        
        carry_0 = jnp.where(done, self.init_carry_0, carry[0])
        carry_1 = jnp.where(done, self.init_carry_1, carry[1])
        carry = (carry_0, carry_1)

        carry, x = self.lstm_cell(carry, x)
        
        return carry, x

    def initialize_carry(self, batch_size):
        return (jnp.ones((batch_size, 1)) * self.init_carry_0, jnp.ones((batch_size, 1)) * self.init_carry_1)

class ResetLSTMLayer(nn.Module):
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x, done, carry = None):
        lstm = nn.scan(ResetLSTMCell,
                       variable_broadcast="params",
                       split_rngs={"params": False},
                       in_axes=1,
                       out_axes=1)(self.hidden_size)

        if carry is None:
            carry = lstm.initialize_carry(x.shape[0])

        carry, x = lstm(carry, x, done)

        return carry, x

class BaseLSTMModel(nn.Module):
    hidden_size: int = 128
    num_early_layers: int = 1
    num_layers: int = 1

    def setup(self):
        early_layers = []
        for ii in range(self.num_early_layers):
            early_layers.append(nn.Dense(self.hidden_size))
            early_layers.append(nn.relu)
            
        self.early_layers = nn.Sequential(early_layers)
        
        self.lstm = ResetLSTMLayer(self.hidden_size)
        
        layers = []
        for ii in range(self.num_layers):
            layers.append(nn.Dense(self.hidden_size))
            layers.append(nn.relu)

        self.layers = nn.Sequential(layers)

    def __call__(self, x, done, carry = None):
        x = self.early_layers(x)
        carry, x = self.lstm(x, done, carry)
        return carry, self.layers(x)

class RNNRouter(nn.Module):
    route: nn.Module

    @nn.compact
    def __call__(self, carry, x):

        return carry, self.route(x)