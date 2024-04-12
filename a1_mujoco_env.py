# TODO: Add custom Tensorboard calls for individual reward functions to get a better
#   sense of the contribution of each reward function

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco_viewer
import numpy as np
from path import Path
from reward import RewardCalculator


class A1MujocoEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30, }
    model_path = Path("unitree_a1/unitree_a1_torque.xml")

    def __init__(self, max_ep_len=1000):
        super().__init__()
        
        # For more information on Mujoco's Python bindings, refer to:
        # https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html
        # https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.ep_timestep = 0
        
        self.reward_calculator = RewardCalculator()
        
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)
        
        # Extra attributes
        self.last_render_time = -1.0
        

    def step(self, action):
        # TODO: How do you apply torque to a joint in mujoco
        # Update data with global (x, y, z) values
        self.ep_timestep += 1
        
        info = None
        terminated = False
        truncated = False
        if self.ep_timestep >= self.max_ep_len:
            terminated = True
        
        if self.__has_fallen():
            terminated = True

        # Step the 
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        obs = self.__get_observations()
        action
        next_obs
        reward = self.reward_calculator.calculate_reward(obs, action, next_obs)
        
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset the simulation to frame 0
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.last_render_time = -1.0
        self.ep_timestep = 0
        
        return observation, info

    def render(self):
        # TODO: For native viewer, refer to: https://colab.research.google.com/drive/1d7r7_TFXLBsnUfsYz77vye4jKhSEXA0t?usp=sharing#scrollTo=40vvXA4DdT9i
        # Render at render_fps
        if self.data.time - self.last_render_time >= 1.0 / self.metadata["render_fps"]:
            self.last_render_time = self.data.time
            mujoco_viewer.render(self.model, self.data)

    def close(self):
        pass
    
    def __get_observations(self):
        pass
    
    def __has_fallen(self):
        return False