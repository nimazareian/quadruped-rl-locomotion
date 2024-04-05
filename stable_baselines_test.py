import time
import gymnasium as gym

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy


import numpy as np
from loco_mujoco import LocoEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

from tqdm import tqdm


# Create environment
env = gym.make("LocoMujoco", 
                env_name="UnitreeA1.simple.perfect")
# env = gym.make("LunarLander-v2", render_mode="rgb_array")

# Instantiate the agent
# model = DQN("MlpPolicy", env, verbose=1)
# # Train the agent and display a progress bar
# model.learn(total_timesteps=int(2e5), progress_bar=True)
# # Save the agent
# model.save("dqn_lunar")
# del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = PPO.load("models/2024-04-04_17-54-02/ppo_loco_mujoco_1300000.zip", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
dones = False
total_reward = 0
while not dones:
    action, _states = model.predict(obs, deterministic=True)
    print(vec_env.step(action))
    obs, rewards, dones, info = vec_env.step(action)
    print(rewards)
    total_reward += rewards[0]
    vec_env.render()
    time.sleep(0.01)

print(f"Total reward: {total_reward}")