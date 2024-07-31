from typing import Any, Dict, Optional, Tuple, Union

from flax import struct

import chex

import jax
from jax import jit, lax
import jax.numpy as jnp

from functools import partial

from gymnax.environments import environment
from gymnax.environments import spaces

@struct.dataclass
class GridWorldState(environment.EnvState):
    map: jax.Array
    pos: jax.Array
    target: jax.Array
    direction: jax.Array
    
    time: int
    
@struct.dataclass
class GridWorldParams(environment.EnvParams):
    noise_thres: float = 0.75
    max_steps_in_episode: int = 200

class GridWorld(environment.Environment[GridWorldState, GridWorldParams]):
    STEP_REWARD = -0.1
    END_REWARD = 1.0
    VOID = 0
    WALL = 1

    ACTIONS = jnp.array([[1, 0], 
                         [-1, 0], 
                         [0, 1], 
                         [0, -1],], 
                        dtype = jnp.int32
                       )

    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        camera_distance: int = 1,
        view_size: Tuple[int, int] = (3, 3),
        noise_thres: float = 1,
        max_steps_in_episode: int = 200,
    ):
        super().__init__()

        self.grid_size = grid_size
    
        self.camera_distance = camera_distance
        self.view_size = view_size
    
        self.noise_thres = noise_thres
    
        self.max_steps_in_episode = max_steps_in_episode

    @property
    def name(self) -> str:
        return "JAXGym-GridWorld"

    @property
    def default_params(self):
        return GridWorldParams(
            noise_thres = self.noise_thres, 
            max_steps_in_episode = self.max_steps_in_episode, 
        )

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params) -> spaces.Discrete:
        return spaces.Discrete(4)

    def observation_space(self, params) -> spaces.Box:
        return spaces.Box(0, 1, self.view_size + (3,), dtype=jnp.int32)

    ########################################################################################################

    @partial(jit, static_argnums=(0,))
    def get_reward_done(self, state, params):
        reward, done = jax.lax.cond(jnp.array_equal(state.pos, state.target),
                                    lambda: (1 - 0.9 * state.time / params.max_steps_in_episode, True),
                                    lambda: (0.0, False))

        done = jax.lax.cond(state.time == params.max_steps_in_episode, 
                            lambda: True, 
                            lambda: done)

        return reward, done

    @partial(jit, static_argnums=(0,))
    def check_pos_validity(self, pos, map, params):
        return (pos >= 0).all() & (pos - jnp.array(self.grid_size) < 0).all() & (map[pos[0], pos[1]] == self.VOID)

    @partial(jit, static_argnums=(0,))
    def get_obs(self, state, params, key):

        pos, map, direction, target = state.pos, state.map, state.direction, state.target

        grid_size = self.grid_size
        view_size = self.view_size
        camera_distance = self.camera_distance

        # view_size = jax.lax.cond(direction[0] != 0, lambda: view_size, lambda: (view_size[1], view_size[0]))
        
        view_side_size = (view_size[0] // 2, 
                          view_size[1] // 2)

        canvas_size = (grid_size[0] + 2 * view_size[0], 
                       grid_size[1] + 2 * view_size[1])
        
        canvas = jnp.full(canvas_size, self.WALL)
        canvas = canvas.at[view_size[0]: -view_size[0], view_size[1]: -view_size[1]].set(map)

        target_canvas = jnp.zeros_like(canvas)
        target_canvas = target_canvas.at[view_size[0] + target[0], view_size[1] + target[1]].set(1)

        pos_canvas = jnp.zeros_like(canvas)
        pos_canvas = pos_canvas.at[view_size[0] + pos[0], view_size[1] + pos[1]].set(1)

        canvas = jnp.stack([canvas, pos_canvas, target_canvas], axis = -1)

        camera_pos = (view_size[0] + pos[0] + direction[0] * camera_distance, 
                      view_size[1] + pos[1] + direction[1] * camera_distance)
        
        view_lower_pin = (camera_pos[0] - view_side_size[0], 
                          camera_pos[1] - view_side_size[1],
                          0
                         )

        view = jax.lax.dynamic_slice(canvas, view_lower_pin, view_size + (3,))

        return view

    @partial(jit, static_argnums=(0,))
    def reset_env(self, key, params):
        map_key, pos_key, target_key, direction_key, obs_key = jax.random.split(key, 5)

        print("reset", type(self.grid_size[0]))

        map = jnp.where(
            jnp.abs(jax.random.normal(map_key, self.grid_size)) > params.noise_thres, 
            self.WALL, self.VOID
        )

        target_key1, target_key2 = jax.random.split(key, 2)
        pos = jax.random.randint(pos_key, (2,), 0, jnp.array(self.grid_size))
        # target = jax.random.randint(target_key, (2,), 0, jnp.array(self.grid_size))
        target = jnp.array([
            jax.random.choice(target_key1, a = jnp.arange(self.grid_size[0]), p = jnp.ones((self.grid_size[0],)).at[pos[0]].set(0)),
            jax.random.choice(target_key2, a = jnp.arange(self.grid_size[1]), p = jnp.ones((self.grid_size[1],)).at[pos[1]].set(0)),
        ])
        
        map = map.at[pos[0], pos[1]].set(self.VOID)
        map = map.at[target[0], target[1]].set(self.VOID)

        direction_idx = jax.random.randint(direction_key, (1,), 0, len(self.ACTIONS))[0]
        direction = self.ACTIONS[direction_idx]

        state = GridWorldState(
            map = map, 
            pos = pos,
            target = target,
            direction = direction,
            time = 0,
        )

        return self.get_obs(state, params, obs_key), state

    def step_env(
        self,
        key, 
        state: GridWorldState, 
        action, 
        params: GridWorldParams,
        ):

        # outputs: obs_st, state_st, reward, done, info

        direction = self.ACTIONS[action]
        next_pos = state.pos + self.ACTIONS[action]
        
        valid_flag = self.check_pos_validity(next_pos, state.map, params)
        next_pos = jax.lax.cond(valid_flag, lambda: next_pos, lambda: state.pos)
        
        next_state = state.replace(pos = next_pos, direction = direction, time = state.time + 1)

        obs = self.get_obs(next_state, params, key)

        reward, done = self.get_reward_done(next_state, params)

        return obs, next_state, reward, done, {}


    ########################################################################################################


    def visualize_state(
        self, 
        state,
        wall_color = jnp.array([0.01, 0.01, 0.01]),
        pos_color = jnp.array([1, 0, 0]),
        target_color = jnp.array([0, 1, 0]), 
        void_color = jnp.array([1, 1, 1]) # colors expressed in RGB
    ):
        map, pos, target = state.map, state.pos, state.target

        map = jnp.expand_dims(map, axis = -1)
        
        wall_color = jnp.expand_dims(wall_color, axis = (0, 1))
        void_color = jnp.expand_dims(void_color, axis = (0, 1))

        vis = jnp.where(map == self.WALL, wall_color, void_color)
        
        vis = vis.at[pos[0], pos[1]].set(pos_color)
        vis = vis.at[target[0], target[1]].set(target_color)

        return (vis * 255).astype(int)

    def visualize_obs(
        self, 
        obs,
        wall_color = jnp.array([0.01, 0.01, 0.01]),
        pos_color = jnp.array([1, 0, 0]),
        target_color = jnp.array([0, 1, 0]), 
        void_color = jnp.array([1, 1, 1]) # colors expressed in RGB
    ):
        
        wall_color = jnp.expand_dims(wall_color, axis = (0, 1))
        pos_color = jnp.expand_dims(pos_color, axis = (0, 1))
        target_color = jnp.expand_dims(target_color, axis = (0, 1))
        void_color = jnp.expand_dims(void_color, axis = (0, 1))

        vis = jnp.where(obs[:, :, [0]] == self.WALL, wall_color, void_color)
        vis = jnp.where(obs[:, :, [1]] == 0, vis, pos_color)
        vis = jnp.where(obs[:, :, [2]] == 0, vis, target_color)

        return (vis * 255).astype(int)