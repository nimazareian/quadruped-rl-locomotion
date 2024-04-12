import numpy as np
from constants import ActionIndex, ObservationIndex


class RewardHandler:
    """Class used to calculate the reward for the Unitree A1 environment."""
    
    desired_velocity = np.array([0.5, 0.0])
    
    tracking_velocity_sigma = 0.25
    max_xy_vel_tracking_reward = 0.8
    max_standing_reward = 0.1
    max_minimal_action_reward = 0.5
    
    def __init__(self):
        self.prev_action = np.zeros(len(ActionIndex))

    def calculate_reward(self, state, action, next_state):
        vel_tracking_rew = self.__reward_tracking_velocity(state, action, next_state)
        standing_rew = self.__reward_standing(state, action, next_state)
        minimal_action_rew = self.__reward_minimal_action(state, action, next_state)
        # print(f"{vel_tracking_rew=}, {standing_rew=}, {minimal_action_rew=}")

        # Update the previous action
        self.prev_action = action

        return np.sum([standing_rew, vel_tracking_rew, minimal_action_rew])

    def __reward_tracking_velocity(self, state, action, next_state):
        curr_global_vel = np.array(
            [state[ObservationIndex.trunk_tx_vel], state[ObservationIndex.trunk_ty_vel]]
        )
        
        # Penalize the agent for not tracking the desired velocity
        # using a gaussian
        vel_sqr_error = np.sum(np.square(self.desired_velocity - curr_global_vel))
        return np.exp(-vel_sqr_error / self.tracking_velocity_sigma) * self.max_xy_vel_tracking_reward

    def __reward_standing(self, state, action, next_state):
        # Penalize the agent for not being at 0 height
        return np.exp(-state[ObservationIndex.trunk_tz_pos]**2) * self.max_standing_reward

    def __reward_minimal_action(self, state, action, next_state):
        # Penalize the agent for moving
        l2_norm = np.sum(np.square(action - self.prev_action))
        return np.exp(-l2_norm / 10.0) * self.max_minimal_action_reward
