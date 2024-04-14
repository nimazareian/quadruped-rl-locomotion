import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

import mujoco

import numpy as np
from pathlib import Path


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
    model_path = Path("./unitree_go1/scene_torque.xml")

    def __init__(self, **kwargs):
        self.reward_weights = {
            "vel_tracking": 2.0,
            "healthy": 0.05,
            "feet_airtime": 2.0,
        }
        self.cost_weights = {
            "action_rate": 0.25,
            "contact": 0.001,
        }

        # vx (m/s), vy (m/s), wz (rad/s)
        self._desired_velocity = np.array([1.0, 0.0, 0.0])
        self._desired_velocity_min = np.array([-0.5, -0.6, -0.6])
        self._desired_velocity_max = np.array([1.5, 0.6, 0.6])

        self._max_xy_vel_tracking_reward = 1
        self._tracking_velocity_sigma = 0.25
        self._ctrl_cost_weight = 0.05
        self._contact_cost_weight = 5e-4

        self._healthy_reward = 1
        self._healthy_z_range = (0.195, 0.75)
        self._healthy_pitch_range = (-np.deg2rad(15), np.deg2rad(15))
        self._healthy_roll_range = (-np.deg2rad(15), np.deg2rad(15))

        self._contact_force_range = (-1.0, 1.0)

        self._feet_contacting_ground_threshold = (
            0.006  # When idle, feet sites are at 0.0053
        )
        self._feet_air_time = np.zeros(4)

        self._main_body = 1

        self._reset_noise_scale = 0.1

        # Action: 12 torque values
        self._last_action = np.zeros(12)

        MujocoEnv.__init__(
            self,
            model_path=self.model_path.absolute().as_posix(),
            frame_skip=2,  # Perform an action every 2 frames (dt(=0.002) * 2 = 0.004 seconds -> 250hz action rate)
            observation_space=None,  # Manually defined afterwards
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
            "render_fps": 60,
        }

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._get_obs().shape, dtype=np.float64
        )

        print(f"{self.action_space=}")

        # In the updated go1_torque.xml, the motors control is clamped to -1.0 and 1.0
        # But geared up to be equal to the torque of the actual Unitree Go1 motors

        # Feet site name to index mapping
        # https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-site
        feet_site = [
            "FR",
            "FL",
            "RR",
            "RL",
        ]
        self.feet_site_name_to_id = {
            f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        }

    def step(self, action):
        self._last_action = action

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
        # cfrc_ext forces are stored in the following indices
        # 4:fromright
        # 7:frontleft
        # 10:backright
        # 13:backleft
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
    def feet_air_time_reward(self):
        feet_heights = self.data.site_xpos[list(self.feet_site_name_to_id.values()), 2]
        feet_contact = feet_heights < self._feet_contacting_ground_threshold

        self._feet_air_time[feet_contact] = 0
        self._feet_air_time[~feet_contact] += self.dt

        return np.sum(
            self._feet_air_time - 0.5
        )  # TODO: Does it make sense to subtract 0.5 from all 4?

    def velocity_tracking_reward(self, xy_velocity):
        vel_sqr_error = np.sum(np.square(self._desired_velocity[:2] - xy_velocity))
        return (
            np.exp(-vel_sqr_error / self._tracking_velocity_sigma)
            * self._max_xy_vel_tracking_reward
        )

    def vertical_velocity_cost(self, z_velocity):
        return z_velocity**2

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z

        min_roll, max_roll = self._healthy_roll_range
        is_healthy = is_healthy and min_roll <= state[4] <= max_roll

        min_pitch, max_pitch = self._healthy_pitch_range
        is_healthy = is_healthy and min_pitch <= state[5] <= max_pitch

        return is_healthy

    def _get_rew(self, xy_velocity, action):
        # TODO: Add custom Tensorboard calls for individual reward functions to get a better
        #   sense of the contribution of each reward function
        # TODO:
        #  - Measure step duration using contact forces
        #  - Give reward for the orientation of the robot
        #  - RCL GPU paper gives a reward and a penalty for lin + ang velocity tracking!!
        vel_tracking_reward = (
            self.velocity_tracking_reward(xy_velocity)
            * self.reward_weights["vel_tracking"]
        )
        healthy_reward = self.healthy_reward * self.reward_weights["healthy"]
        feet_air_time_reward = 0  # TODO: Tune self.feet_air_time_reward * self.reward_weights["feet_airtime"]
        rewards = vel_tracking_reward + healthy_reward + feet_air_time_reward

        ctrl_cost = self.control_cost(action) * self.cost_weights["action_rate"]
        contact_cost = self.contact_cost * self.cost_weights["contact"]
        costs = ctrl_cost + contact_cost

        # self.dt coefficient does not seem to have an effect on the result
        reward = rewards  # TODO: (rewards - costs) - self.dt might make the gradient small and learning slow

        # TODO: Reward info isnt accurate as it doesn't include the weights
        reward_info = {
            "vel_tracking_reward": vel_tracking_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def _get_obs(self):
        # The first three indices are the global x and y positions of the robot
        # The second four are the quaternion representing the orientation of the robot
        # The remaining 12 values are the joint positions
        position = self.data.qpos[7:].flatten()

        # The first three values are the global linear velocity of the robot
        # The second three are the angular velocity of the robot
        # The remaining 12 values are the joint velocities
        velocity = self.data.qvel.flatten()

        desired_vel = self._desired_velocity
        last_action = self._last_action
        # contact_force = self.contact_forces[1:].flatten()

        return np.concatenate((position, velocity, desired_vel, last_action))

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

    def _sample_desired_vel(self):
        return np.random.default_rng().uniform(
            low=self._desired_velocity_min, high=self._desired_velocity_max
        )
