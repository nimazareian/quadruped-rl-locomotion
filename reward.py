import numpy as np
from constants import ObservationIndex


def combined_reward_function(state, action, next_state):
    vel_tracking_rew = reward_tracking_velocity(state, action, next_state)
    standing_rew = reward_standing(state, action, next_state)
    return np.exp(np.sum([standing_rew]))


def reward_tracking_velocity(state, action, next_state):
    desired_vel = np.array([0.5, 0.0])
    curr_global_vel = np.array(
        [state[ObservationIndex.trunk_tx_vel], state[ObservationIndex.trunk_ty_vel]]
    )

    # L2 norm on the velocity difference
    # TODO: Reward should be non-negative, and increase the closer we are to the desired velocity
    return -np.linalg.norm(desired_vel - curr_global_vel)


def reward_standing(state, action, next_state):
    # Penalize the agent for not being at 0 height
    return -abs(state[ObservationIndex.trunk_tz_pos])
