from enum import IntEnum, unique

@unique
class ObservationIndex(IntEnum):
    """Associated indices for each joint position and velocity
    From: https://loco-mujoco.readthedocs.io/en/latest/source/loco_mujoco.environments.quadrupeds.html#observation-space
    
    Note that trunk_tx_pos and trunk_ty_pos are ommitted by LocoMujoco
    """
    
    # Joint Positions
    trunk_tz_pos = 0
    trunk_list_pos = 1
    trunk_tilt_pos = 2
    trunk_rotation_pos = 3
    
    FR_hip_joint_pos = 4
    FR_thigh_joint_pos = 5
    FR_calf_joint_pos = 6
    
    FL_hip_joint_pos = 7
    FL_thigh_joint_pos = 8
    FL_calf_joint_pos = 9
    
    RR_hip_joint_pos = 10
    RR_thigh_joint_pos = 11
    RR_calf_joint_pos = 12
    
    RL_hip_joint_pos = 13
    RL_thigh_joint_pos = 14
    RL_calf_joint_pos = 15

    # Joint Velocities
    trunk_tx_vel = 16
    trunk_ty_vel = 17
    trunk_tz_vel = 18
    trunk_list_vel = 19
    trunk_tilt_vel = 20
    trunk_rotation_vel = 21

    FR_hip_joint_vel = 22
    FR_thigh_joint_vel = 23
    FR_calf_joint_vel = 24

    FL_hip_joint_vel = 25
    FL_thigh_joint_vel = 26
    FL_calf_joint_vel = 27

    RR_hip_joint_vel = 28
    RR_thigh_joint_vel = 29
    RR_calf_joint_vel = 30

    RL_hip_joint_vel = 31
    RL_thigh_joint_vel = 32
    RL_calf_joint_vel = 33

    desired_sin_cos_vel = 34
    
    desired_vel = 36


@unique
class ActionIndex(IntEnum):
    FR_hip = 0
    FR_thigh = 1
    FR_calf = 2
    
    FL_hip = 3
    FL_thigh = 4
    FL_calf = 5
    
    RR_hip = 6
    RR_thigh = 7
    RR_calf = 8
    
    RL_hip = 9
    RL_thigh = 10
    RL_calf = 11


def reward_config():
    return dict(
        # TODO
    )
