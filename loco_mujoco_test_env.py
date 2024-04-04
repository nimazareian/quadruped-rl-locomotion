import argparse
import os
import time

import numpy as np
from loco_mujoco import LocoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from tqdm import tqdm

ENV_NAME = "UnitreeA1.simple.perfect"
TIME_STEPS_PER_SAVE = 25_000
NUM_PARALLEL_ENVS = 4
SEED = 0

MODEL_DIR = "models"
LOG_DIR = "logs"

# define what ever reward function you want
def my_reward_function(state, action, next_state):
    return -np.mean(action)  # here we just return the negative mean of the action


def train():
    # Can pass a custom reward function using:
    # reward_type="custom", reward_params=dict(reward_callback=my_reward_function)

    # https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html
    # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
    # https://pythonprogramming.net/custom-environment-reinforcement-learning-stable-baselines-3-tutorial/?completed=/saving-and-loading-reinforcement-learning-stable-baselines-3-tutorial/
    # vec_env = make_vec_env(
    #     "LocoMujoco",
    #     env_name=ENV_NAME,
    #     n_envs=NUM_PARALLEL_ENVS,
    #     seed=SEED,
    #     vec_env_cls=SubprocVecEnv,
    # )

    vec_env = gym.make("LocoMujoco", 
                       env_name=ENV_NAME)


    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR)

    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    time_steps = TIME_STEPS_PER_SAVE
    while True:
        model.learn(total_timesteps=TIME_STEPS_PER_SAVE, reset_num_timesteps=False)
        model.save(f"{MODEL_DIR}/{train_time}/ppo_loco_mujoco_{time_steps}")
        time_steps += TIME_STEPS_PER_SAVE

    # obs, info = vec_env.reset()
    # vec_env.render()
    # terminated = False
    # i = 0
    # for _ in tqdm(range(10_000)):
    #     if i == 1000 or terminated:
    #         vec_env.reset()
    #         i = 0

    #     action, _ = model.predict(obs)
    #     nstate, reward, terminated, truncated, info = vec_env.step(action)
    #     # nstate is comprised of 37 floats, and action is comprised of 12 floats (Unclear what the units are)
    #     # More detail: https://loco-mujoco.readthedocs.io/en/latest/source/loco_mujoco.environments.quadrupeds.html

    #     vec_env.render()
    #     i += 1

def test(model_path):
    env = gym.make("LocoMujoco", 
                    env_name=ENV_NAME)
    obs, _ = env.reset()

    model = PPO.load(model_path, env=env)
    episode_reward = 0
    while True:
        action, _ = model.predict(obs)
        nstate, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        env.render()

        if terminated:
            break

    print(f"Total episode reward: {episode_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", help="Path to the model to test")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.train:
        train()
    elif args.test:
        test(args.test)