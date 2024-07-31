import jax
import jax.numpy as jnp

import flax
from flax import linen as nn  # Linen API

from models.rnn_model import ResetLSTMCell, ResetLSTMLayer, RNNRouter

class BaseLSTMModel(nn.Module):
    hidden_size: int = 128
    num_early_layers: int = 1
    num_layers: int = 1

    def setup(self):
        
        self.conv = nn.Conv(features = 32, kernel_size = (3, 3))
        
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
        N, T, H, W, C = x.shape

        x = jnp.reshape(x, (N * T, H, W, C))
        x = self.conv(x)
        x = jnp.reshape(x, (N, T, -1))

        x = self.early_layers(x)
        carry, x = self.lstm(x, done, carry)
        return carry, self.layers(x)