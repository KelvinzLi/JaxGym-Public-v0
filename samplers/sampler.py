import jax
import jax.numpy as jnp

from structs import Buffer, Sample

class Sampler:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample_batch(self, buffer, key):

        num_envs, buffer_size = buffer.obs.shape[:2]
        
        vmap_key = jax.random.split(key, num_envs)

        batch_idx = jax.vmap(lambda key, m: jax.random.choice(key, jnp.arange(buffer_size), shape=(self.batch_size,), p = m), 
                             in_axes = (0, None))(vmap_key, buffer.mask)
                
        aux_idx = jnp.repeat(jnp.expand_dims(jnp.arange(num_envs), axis = 1), self.batch_size, axis = 1)

        sample = Sample(
            obs = buffer.obs[aux_idx, batch_idx],
            next_obs = buffer.obs[aux_idx, batch_idx + 1],
    
            reward = buffer.reward[aux_idx, batch_idx],
            action = buffer.action[aux_idx, batch_idx],
            done = buffer.done[aux_idx, batch_idx],
        )

        return sample

    def update_buffer(self, buffer, sample):
        def update_buffer_fori_body(ii, carry):
            buffer, sample = carry

            buffer_size = buffer.shape[0]

            shifted_ii = jax.lax.cond(shift + ii < buffer_size, lambda: shift + ii, lambda: shift + ii - buffer_size)

            buffer = buffer.at[shifted_ii].set(sample[ii])

            return buffer, sample

        shift = buffer.head

        buffer_size = buffer.obs.shape[1]
        sample_size = sample.obs.shape[1]

        mask = jnp.ones((sample_size,)).at[-1].set(0)

        update_func = lambda b, s: jax.lax.fori_loop(0, sample_size, update_buffer_fori_body, (b, s))[0]
        vmap_update_func = jax.vmap(update_func, in_axes = (0, 0))

        buffer = Buffer(
            vmap_update_func(buffer.obs, sample.obs),
            vmap_update_func(buffer.reward, sample.reward),
            vmap_update_func(buffer.action, sample.action),
            vmap_update_func(buffer.done, sample.done),
            update_func(buffer.mask, mask).at[-1].set(0),
            jax.lax.cond(shift + sample_size < buffer_size, lambda: shift + sample_size, lambda: shift + sample_size - buffer_size)
        )

        return buffer