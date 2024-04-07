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
import time


MODEL_DIR = "models"
LOG_DIR = "logs"

# define what ever reward function you want
def my_reward_function(state, action, next_state):
    return -np.mean(action)  # here we just return the negative mean of the action


def train(args):
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

    # vec_env = gym.make("LocoMujoco", 
    #                    env_name=ENV_NAME)
    vec_env = make_vec_env(args.env, n_envs=4)

    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR)
    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    model.learn(total_timesteps=args.total_timesteps)
    model.save(f"{MODEL_DIR}/{train_time}/ppo_loco_mujoco_{args.total_timesteps}")
    del model
    model = PPO.load(f"{MODEL_DIR}/{train_time}/ppo_loco_mujoco_{args.total_timesteps}")


def test(args):
    env = make_vec_env(args.env, n_envs=1)
    # env = make_vec_env(args.env, n_envs=4)
    
    model = PPO.load(path=args.model_path)

    NUM_EPISODES = 10
    episode_reward = 0
    episode_length = 0
    for _ in range(NUM_EPISODES):
        obs = env.reset()
        env.render()
        while True:
            action, _ = model.predict(obs)
            obs, reward, terminated, info = env.step(action)
            episode_reward += reward
            env.render("human")
            # time.sleep(0.01)
            episode_length += 1

            if terminated[0]:
                break

    print(f"Total episode reward: {episode_reward}, avg episode length: {episode_length / NUM_EPISODES}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="train", help="test or train")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment to train/test on")
    parser.add_argument("--seed", type=float, default=0)
    parser.add_argument("--num_parallel_envs", type=int, default=4)
    parser.add_argument("--total_timesteps", type= int, default=100000)
    parser.add_argument("--existing_model_path", type= float, default=None, help="Path to the model to continue training")
    parser.add_argument("--model_path", type= str, default=None, help="Path to the model to test")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.run == "train":
        train(args)
    elif args.run == "test":
        test(args)