# TODO: Add custom Tensorboard calls for individual reward functions to get a better
#   sense of the contribution of each reward function

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
import mujoco
import mujoco_viewer
import numpy as np
from pathlib import Path
from reward import RewardCalculator


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class Go1MujocoEnv(MujocoEnv):
    """Custom Environment that follows gym interface."""
    
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }
    model_path = Path("./unitree_go1/scene.xml")

    def __init__(self, **kwargs):
        self._desired_velocity = np.array([1.5, 0.0])
        
        self._max_xy_vel_tracking_reward = 1
        self._tracking_velocity_sigma = 0.6
        self._ctrl_cost_weight = 0.05
        self._contact_cost_weight = 5e-4

        self._healthy_reward = 1
        self._healthy_z_range = (0.195, 0.75)
        self._healthy_pitch_range = (-np.deg2rad(15), np.deg2rad(15))
        self._healthy_roll_range = (-np.deg2rad(15), np.deg2rad(15))
        
        self._contact_force_range = (-1.0, 1.0)
        
        self._main_body = 1

        self._reset_noise_scale = 0.1
        
        MujocoEnv.__init__(
            self,
            model_path=self.model_path.absolute().as_posix(),
            frame_skip=25, # Perform an action every 25 frames (=0.05 seconds)
            observation_space=None, # Manually defined afterwards
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # Update metadata to include the render FPS
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        observation_size = self.data.qpos.size + self.data.qvel.size
        observation_size += self.data.cfrc_ext[1:].size

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float64
        )

    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(xy_velocity, action)
        terminated = not self.is_healthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": xy_velocity[0],
            "y_velocity": xy_velocity[1],
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def close(self):
        pass

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        # TODO:
        #  - Measure step duration using contact forces
        #  - Model should be using motors/torque
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        
        min_roll, max_roll = self._healthy_roll_range
        is_healthy = is_healthy and min_roll <= state[4] <= max_roll
        
        min_pitch, max_pitch = self._healthy_pitch_range
        is_healthy = is_healthy and min_pitch <= state[5] <= max_pitch
        
        return is_healthy

    def _get_rew(self, xy_velocity, action):
        vel_sqr_error = np.sum(np.square(self._desired_velocity - xy_velocity))
        vel_tracking_reward = np.exp(-vel_sqr_error / self._tracking_velocity_sigma) * self._max_xy_vel_tracking_reward
        
        healthy_reward = self.healthy_reward
        rewards = vel_tracking_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        reward_info = {
            "vel_tracking_reward": vel_tracking_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        contact_force = self.contact_forces[1:].flatten()

        return np.concatenate((position, velocity, contact_force))

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
