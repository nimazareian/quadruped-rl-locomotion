import numpy as np
from loco_mujoco import LocoEnv


np.random.seed(0)
mdp = LocoEnv.make("UnitreeA1.simple.perfect")

mdp.play_trajectory(n_steps_per_episode=500)