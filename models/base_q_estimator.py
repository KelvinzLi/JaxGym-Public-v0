import jax.numpy as jnp

import flax
from flax import linen as nn  # Linen API

class BaseQEstimator(nn.Module):
    hidden_size: int = 128
    num_layers: int = 1

    def setup(self):
        layers = []
        for ii in range(self.num_layers):
            layers.append(nn.Dense(self.hidden_size))
            layers.append(nn.relu)

        self.layers = nn.Sequential(layers)

    def __call__(self, x, action):
        x = jnp.concatenate([x, action], axis = -1)
        return self.layers(x)