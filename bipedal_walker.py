import argparse
import os
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
from tqdm import tqdm

TIME_STEPS_PER_SAVE = 100_000
NUM_PARALLEL_ENVS = 4
SEED = 0

MODEL_DIR = "models"
LOG_DIR = "logs"

def train(starting_model=None):
    # TODO: Make this a vectorized environment
    env = gym.make("BipedalWalker-v3")

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(env, best_model_save_path=MODEL_DIR + "/ppo_bipedal_walker",
                                log_path=LOG_DIR, eval_freq=20_000,
                                deterministic=True, render=False)
    
    if starting_model:
        model = PPO.load(
            path=starting_model, env=env, verbose=1, tensorboard_log=LOG_DIR
        )
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)


    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    model.learn(
        total_timesteps=100_000_000,
        reset_num_timesteps=False,
        progress_bar=True,
        tb_log_name=f"ppo_bipedal_walker_{train_time}",
        callback=eval_callback
    )


def test(model_path):
    env = gym.make("BipedalWalker-v3") # render_mode="human"

    print(f"Testing model: {model_path}")
    model = PPO.load(path=model_path, env=env, verbose=1)

    NUM_EPISODES = 5
    episode_reward = 0
    episode_length = 0
    for _ in tqdm(range(NUM_EPISODES)):
        vec_env = model.get_env()
        obs = vec_env.reset()
        extra = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            nstate, reward, dones, info = vec_env.step(action)
            episode_reward += reward
            episode_length += 1

            if np.all(dones):
                extra -= 1
                if extra <= 0:
                    break

    print(
        f"Avg episode reward: {episode_reward / NUM_EPISODES}, avg episode length: {episode_length / NUM_EPISODES}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--existing_model_path",
        help="Path to the model to continue training",
        default=None,
    )
    parser.add_argument("--test", help="Path to the model to test")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.train:
        train(args.existing_model_path)
    elif args.test:
        test(args.test)
