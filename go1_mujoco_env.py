import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

import mujoco

import numpy as np
from pathlib import Path


DEFAULT_CAMERA_CONFIG = {
    "distance": 2.0,
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
    model_path = Path("./unitree_go1/scene_position.xml")

    def __init__(self, **kwargs):
        MujocoEnv.__init__(
            self,
            model_path=self.model_path.absolute().as_posix(),
            frame_skip=10,  # Perform an action every 2 frames (dt(=0.002) * 2 = 0.004 seconds -> 250hz action rate)
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
        self._last_render_time = -1.0
        self._max_episode_time_sec = 15.0
        self._step = 0

        # Weights for the reward and cost functions
        self.reward_weights = {
            "linear_vel_tracking": 2.0,  # Was 1.0
            "angular_vel_tracking": 0.5,
            "healthy": 0.0,  # was 0.05
            "feet_airtime": 1.0,
        }
        self.cost_weights = {
            "torque": 0.0001,  # Was 0.0002
            "vertical_vel": 0.4,  # Was 1.0
            "xy_angular_vel": 0.02,  # Was 0.05
            "action_rate": 0.01,
            "joint_limit": 10.0,
        }

        # vx (m/s), vy (m/s), wz (rad/s)
        self._desired_velocity_min = np.array([-0.5, 0, 0.0])  # y and wz -0.6, 0.6
        self._desired_velocity_max = np.array([1.5, 0, 0.0])
        self._desired_velocity = self._sample_desired_vel()
        self._velocity_scale = np.array([2.0, 2.0, 0.25])
        self._tracking_velocity_sigma = 0.25

        # Metrics used to determine if the episode should be terminated
        self._healthy_z_range = (0.19, 0.65)
        self._healthy_pitch_range = (-np.deg2rad(15), np.deg2rad(15))
        self._healthy_roll_range = (-np.deg2rad(15), np.deg2rad(15))

        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._cfrc_ext_feet_indices = [4, 7, 10, 13]  # 4:FR, 7:FL, 10:RR, 13:RL

        # Non-penalized degrees of freedom range of the control joints
        dof_position_limit_multiplier = 0.9  # The % of the range that is not penalized
        ctrl_range_offset = (
            0.5
            * (1 - dof_position_limit_multiplier)
            * (
                self.model.actuator_ctrlrange[:, 1]
                - self.model.actuator_ctrlrange[:, 0]
            )
        )
        self._soft_ctrl_range = np.copy(self.model.actuator_ctrlrange)
        self._soft_ctrl_range[:, 0] += ctrl_range_offset
        self._soft_ctrl_range[:, 1] -= ctrl_range_offset

        self._reset_noise_scale = 0.1

        # Action: 12 torque values
        self._last_action = np.zeros(12)

        self._clip_obs = 100.0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._get_obs().shape, dtype=np.float64
        )

        print(f"{self.action_space=}")

        # In the updated go1_torque.xml, the motors control is clamped to -1.0 and 1.0
        # But geared up to be equal to the torque of the actual Unitree Go1 motors

        # Feet site name to index mapping
        # https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-site
        # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtobj
        feet_site = [
            "FR",
            "FL",
            "RR",
            "RL",
        ]
        self._feet_site_name_to_id = {
            f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        }

        self._main_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY.value, "trunk"
        )

    def step(self, action):
        self._step += 1
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._calc_reward(action)
        # TODO: Consider terminating if knees touch the ground
        terminated = not self.is_healthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            **reward_info,
        }

        if self.render_mode == "human" and (self.data.time - self._last_render_time) > (
            1.0 / self.metadata["render_fps"]
        ):
            self.render()
            self._last_render_time = self.data.time
            # print(f"{self.data.time=}")

        self._last_action = action
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        truncated = self._step >= (self._max_episode_time_sec / self.dt)
        return observation, reward, terminated, truncated, info

    @property
    def feet_contact_forces(self):
        feet_contact_forces = self.data.cfrc_ext[self._cfrc_ext_feet_indices]
        return np.linalg.norm(feet_contact_forces, axis=1)

    ######### Positive Reward functions #########
    @property
    def linear_velocity_tracking_reward(self):
        vel_sqr_error = np.sum(
            np.square(self._desired_velocity[:2] - self.data.qvel[:2])
        )
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    @property
    def angular_velocity_tracking_reward(self):
        vel_sqr_error = (self._desired_velocity[2] - self.data.qvel[5]) ** 2
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    @property
    def heading_tracking_reward(self):
        # TODO: Unsure how to calculate the robots heading. qpos[3:7] are the quaternion values
        pass

    @property
    def feet_air_time_reward(self):
        """Award strides depending on their duration only when the feet makes contact with the ground"""
        # TODO: For this to work as expected, do we need to pass a history of observations?
        feet_contact_force_mag = self.feet_contact_forces
        curr_contact = feet_contact_force_mag > 0.1
        contact_filter = np.logical_or(curr_contact, self._last_contacts)
        self._last_contacts = curr_contact

        # if feet_air_time is > 0 (feet was in the air) and contact_filter detects a contact with the ground
        # then it is the first contact of this stride
        first_contact = (self._feet_air_time > 0.0) * contact_filter
        self._feet_air_time += self.dt

        # Award the feets that have just finished their stride (first step with contact)
        air_time_reward = np.sum((self._feet_air_time - 0.5) * first_contact)
        # No award if the desired velocity is very low (i.e. robot should remain stationary and feet shouldn't move)
        air_time_reward *= np.linalg.norm(self._desired_velocity[:2]) > 0.1

        # zero-out the air time for the feet that have just made contact (i.e. contact_filter==1)
        self._feet_air_time *= ~contact_filter

        # TODO: Could penalize for foot sliding like Colab
        # contacting_feet = self.data.site_xvel[list(self._feet_site_name_to_id.values()), 2] * curr_contact
        # print(f"{contacting_feet=}\n")

        return air_time_reward

    @property
    def healthy_reward(self):
        return self.is_healthy

    ######### Negative Reward functions #########
    @property  # TODO: Not used
    def feet_contact_forces_cost(self):
        return np.sum(
            (self.feet_contact_forces - self._max_contact_force).clip(min=0.0)
        )

    @property  # TODO: Not actually used by RSL. Also reading quaternion values...
    def non_flat_base_cost(self):
        # Penalize the robot for not being flat on the ground
        return np.sum(np.square(self.data.qpos[4:6]))

    @property
    def joint_limit_cost(self):
        # Penalize the robot for having the joints outside of the soft control range
        out_of_range = (self._soft_ctrl_range[:, 0] - self.data.qpos[7:]).clip(
            min=0.0
        ) + (self.data.qpos[7:] - self._soft_ctrl_range[:, 1]).clip(min=0.0)
        return np.sum(out_of_range)

    @property
    def torque_cost(self):
        # Last 12 values are the motor torques
        return np.sum(np.square(self.data.qfrc_actuator[-12:]))

    @property
    def vertical_velocity_cost(self):
        return self.data.qvel[2] ** 2

    @property
    def xy_angular_velocity_cost(self):
        return np.sum(np.square(self.data.qvel[3:5]))

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

    def action_rate_cost(self, action):
        return np.sum(np.square(self._last_action - action))

    def _calc_reward(self, action):
        # TODO: Add custom Tensorboard calls for individual reward functions to get a better
        #   sense of the contribution of each reward function
        # TODO:
        #  - Cost for thigh or calf contact with the ground
        #  - Cost for reaching limit of joint angles and motor velocities
        #  - Give reward/cost based on the height of the robot being at (0.25?!) NOT ACTUALLY USED BY RSL!!!
        linear_vel_tracking_reward = (
            self.linear_velocity_tracking_reward
            * self.reward_weights["linear_vel_tracking"]
        )
        angular_vel_tracking_reward = (
            self.angular_velocity_tracking_reward
            * self.reward_weights["angular_vel_tracking"]
        )
        healthy_reward = self.healthy_reward * self.reward_weights["healthy"]
        feet_air_time_reward = (
            self.feet_air_time_reward * self.reward_weights["feet_airtime"]
        )
        rewards = (
            linear_vel_tracking_reward
            + angular_vel_tracking_reward
            + healthy_reward
            + feet_air_time_reward
        )

        ctrl_cost = self.torque_cost * self.cost_weights["torque"]
        action_rate_cost = (
            self.action_rate_cost(action) * self.cost_weights["action_rate"]
        )
        vertical_vel_cost = (
            self.vertical_velocity_cost * self.cost_weights["vertical_vel"]
        )
        xy_angular_vel_cost = (
            self.xy_angular_velocity_cost * self.cost_weights["xy_angular_vel"]
        )
        joint_limit_cost = self.joint_limit_cost * self.cost_weights["joint_limit"]
        costs = (
            ctrl_cost
            + action_rate_cost
            + vertical_vel_cost
            + xy_angular_vel_cost
            + joint_limit_cost
        )

        # self.dt coefficient does not seem to have an effect on the result
        reward = max(0.0, rewards - costs)

        # print(f"sum_{reward=:.3f}  {rewards=:.3f}  {costs=:.3f}  {joint_limit_cost=:.3f}")
        # print(f"sum_{reward=:.3f}  {rewards=:.3f}  {costs=:.3f}  {ctrl_cost=:.3f}  {action_rate_cost=:.3f}  {vertical_vel_cost=:.3f}  {xy_angular_vel_cost=:.3f}")
        # print(f"{self.feet_contact_forces=}")
        # print(f"{linear_vel_tracking_reward=:.3f}   {angular_vel_tracking_reward=:.3f}   {self._desired_velocity=}")

        reward_info = {
            "linear_vel_tracking_reward": linear_vel_tracking_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def _get_obs(self):
        # The first three indices are the global x,y,z position of the trunk of the robot
        # The second four are the quaternion representing the orientation of the robot
        # The remaining 12 values are the joint positions
        # The joint positions are relative to the starting position
        position = self.data.qpos[7:].flatten() - self.model.key_qpos[0, 7:]

        # The first three values are the global linear velocity of the robot
        # The second three are the angular velocity of the robot
        # The remaining 12 values are the joint velocities
        velocity = self.data.qvel.flatten()
        velocity[:3] *= self._velocity_scale

        desired_vel = self._desired_velocity * self._velocity_scale
        last_action = self._last_action

        curr_obs = np.concatenate((position, velocity, desired_vel, last_action)).clip(
            -self._clip_obs, self._clip_obs
        )

        return curr_obs

    def reset_model(self):
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # mujoco.mj_resetData(self.model, self.data)
        # self.set_state(self.init_qpos, self.init_qvel)

        self.data.qpos[:] = self.model.key_qpos[0] + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nq,
        )
        self.data.ctrl[:] = self.model.key_ctrl[
            0
        ] + self._reset_noise_scale * self.np_random.standard_normal(
            *self.data.ctrl.shape
        )

        # Reset the variables and sample a new desired velocity
        self._desired_velocity = self._sample_desired_vel()
        self._step = 0
        self._last_action = np.zeros(12)
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._last_render_time = -1.0

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }

    def _sample_desired_vel(self):
        desired_vel = np.random.default_rng().uniform(
            low=self._desired_velocity_min, high=self._desired_velocity_max
        )
        # print(f"Desired velocity: {desired_vel}")
        return np.array([1.0, 0, 0.0])  # TODO: randomize desired_vel
