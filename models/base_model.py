import jax.numpy as jnp

import flax
from flax import linen as nn  # Linen API

class BaseModel(nn.Module):
    hidden_size: int = 128
    num_layers: int = 1

    def setup(self):
        layers = []
        for ii in range(self.num_layers):
            layers.append(nn.Dense(self.hidden_size))
            layers.append(nn.relu)

        self.layers = nn.Sequential(layers)

    def __call__(self, x):
        return self.layers(x)

class NormalDistPredictor(nn.Module):
    output_size: int
    logvar_init_value: float = 0

    @nn.compact
    def __call__(self, x):
        mean = nn.Dense(features = self.output_size)(x)

        logvar_param = self.param('logvar_param', lambda rng, shape: self.logvar_init_value * jnp.ones(shape), (1,))
        std = jnp.exp(logvar_param / 2) * jnp.ones_like(mean)
        std = jnp.clip(std, 1e-10, 50)

        return mean, std