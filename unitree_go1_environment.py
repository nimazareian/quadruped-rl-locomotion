import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Similar examples:
# LocoMujoco: https://github.com/robfiras/loco-mujoco/blob/master/loco_mujoco/environments/quadrupeds/unitreeA1.py#L838
# PyBullet: https://github.com/arthurchevalley/Legged-Robots-Projects/blob/main/Quadruped%20robot%20walking/Code/env/quadruped_gym_env.py
# PyChrono sim: https://github.com/projectchrono/gym-chrono/blob/86db0c15f909de22906229e38d8806efd4f57a5d/gym_chrono/envs/legged/quadruped_walk.py#L52
class UnitreeGo1Env(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}
    ACTION_SPACE_SIZE = 12

    def __init__(self):
        super().__init__()
        # Box: Supports continuous (and discrete) vectors or matrices, used for vector observations, images, etc
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.ACTION_SPACE_SIZE,), dtype=np.float32)

        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-1, high=255,
                                            shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

    def step(self, action):
        ...
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        ...
        return observation, info

    def render(self):
        ...

    def close(self):
        ...