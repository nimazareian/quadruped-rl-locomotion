import argparse
import os
import time
from pathlib import Path

import numpy as np
from loco_mujoco import LocoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from tqdm import tqdm

ENV_NAME = "UnitreeA1.simple.perfect"
TIME_STEPS_PER_SAVE = 100_000
NUM_PARALLEL_ENVS = 4
SEED = 0

MODEL_DIR = "models"
LOG_DIR = "logs"

# define what ever reward function you want
def my_reward_function(state, action, next_state):
    return -np.mean(action)  # here we just return the negative mean of the action


def train(starting_model=None):
    # Can pass a custom reward function using:
    # reward_type="custom", reward_params=dict(reward_callback=my_reward_function)

    # TODO: Vectorize the envrionment so it can train in parallel multiple instances
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

    # TODO: What is the reward function used by LocoMujoco? 
    #       How does it detect termination?
    vec_env = gym.make("LocoMujoco", 
                       env_name=ENV_NAME)

    if starting_model:
        model = PPO.load(path=starting_model, env=vec_env, verbose=1, tensorboard_log=LOG_DIR)
        time_steps = int(Path(starting_model).stem.split("_")[-1]) + TIME_STEPS_PER_SAVE
    else:
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR)
        time_steps = TIME_STEPS_PER_SAVE
    
    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    while True:
        model.learn(total_timesteps=TIME_STEPS_PER_SAVE, reset_num_timesteps=False, progress_bar=True)
        # TODO: Could use CheckpointCallback and EvalCallback to save the best model automatically.
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

    model = PPO.load(path=model_path, env=env, verbose=1)

    NUM_EPISODES = 10
    episode_reward = 0
    episode_length = 0
    for _ in range(NUM_EPISODES):
        vec_env = model.get_env()
        obs = vec_env.reset()
        vec_env.render()
        extra = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            # print(action)
            nstate, reward, dones, info = vec_env.step(action)
            episode_reward += reward
            vec_env.render()
            time.sleep(0.01)
            episode_length += 1

            if dones:
                extra -= 1
                if extra <= 0:
                    break

    print(f"Total episode reward: {episode_reward}, avg episode length: {episode_length / NUM_EPISODES}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--existing_model_path", help="Path to the model to continue training", default=None)
    parser.add_argument("--test", help="Path to the model to test")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    test("models/2024-04-04_17-54-02/ppo_loco_mujoco_1300000.zip")
    
    # if args.train:
    #     train(args.existing_model_path)
    # elif args.test:
    #     test(args.test)