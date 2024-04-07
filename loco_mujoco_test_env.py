import argparse
import os
import time
from pathlib import Path

import numpy as np
from loco_mujoco import LocoEnv
from loco_mujoco.utils.reward import VelocityVectorReward

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
from tqdm import tqdm

ENV_NAME = "UnitreeA1.simple.perfect"
TOTAL_TRAINING_TIMESTEPS = 1_000_000
EVAL_FREQUENCY = 25_000
NUM_PARALLEL_ENVS = 4
SEED = 0

MODEL_DIR = "models"
LOG_DIR = "logs"


# LocoMujoco reward helper functions: https://loco-mujoco.readthedocs.io/en/latest/source/loco_mujoco.utils.html#module-loco_mujoco.utils.reward
def my_reward_function(state, action, next_state):
    # TODO: Print the state, action, and next_state to understand the data structure
    # Could use VelocityVectorReward: https://github.com/robfiras/loco-mujoco/blob/c4f0e546725d5681a3ec865d3427ce5fdbb7526e/loco_mujoco/environments/quadrupeds/unitreeA1.py#L491
    # Power = Torque * Angular Velocity -> Minimize power/energy usage (i.e. reward -= power)

    return -np.mean(action)  # here we just return the negative mean of the action


def train(starting_model=None):
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
    env = gym.make(
        "LocoMujoco",
        env_name=ENV_NAME,
        # Can pass a custom reward function using:
        # reward_type="custom",
        # reward_params=dict(reward_callback=my_reward_function),
    )

    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"ppo_loco_mujoco_{train_time}"

    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"{MODEL_DIR}/{run_name}",
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQUENCY,
        deterministic=True,
        render=True,
    )

    if starting_model is not None:
        model = PPO.load(
            path=starting_model, env=env, verbose=1, tensorboard_log=LOG_DIR
        )
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

    model.learn(
        total_timesteps=TOTAL_TRAINING_TIMESTEPS,
        reset_num_timesteps=False,
        progress_bar=True,
        tb_log_name=run_name,
        callback=eval_callback,
    )


def test(model_path):
    env = gym.make("LocoMujoco", env_name=ENV_NAME, render_mode="human")

    model = PPO.load(path=model_path, env=env, verbose=1)

    NUM_EPISODES = 5
    NUM_EXTRA_STEPS_AFTER_TERMINATION = 0

    episode_reward = 0
    episode_length = 0
    for _ in tqdm(range(NUM_EPISODES)):
        obs, _ = env.reset()
        env.render()
        extra = NUM_EXTRA_STEPS_AFTER_TERMINATION

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Render the environment at ~100fps
            env.render()
            time.sleep(0.01)

            if terminated or truncated:
                extra -= 1
                if extra <= 0:
                    break
            else:
                episode_length += 1

    print(
        f"Total episode reward: {episode_reward}, avg episode length: {episode_length / NUM_EPISODES}"
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
