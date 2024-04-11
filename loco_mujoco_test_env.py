import argparse
import os
import time

import numpy as np
from loco_mujoco import LocoEnv

from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
from tqdm import tqdm

from reward import combined_reward_function


NUM_PARALLEL_ENVS = 4
SEED = 0

MODEL_DIR = "models"
LOG_DIR = "logs"



def train(args):
    # TODO: Vectorize the envrionment so it can train in parallel multiple instances
    # TODO: What is the reward function used by LocoMujoco?
    #       How does it detect termination?
    env = gym.make(
        "LocoMujoco",
        env_name=args.env,
        # The next 3 arguments can not change for the "perfect" environment
        action_mode="torque",
        use_foot_forces=False,
        default_target_velocity=0.5,
        setup_random_rot=False,
        render_mode="human",
        reward_type="custom",
        reward_params=dict(reward_callback=combined_reward_function),
    )

    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"ppo_loco_mujoco_{train_time}"

    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"{MODEL_DIR}/{run_name}",
        log_path=LOG_DIR,
        eval_freq=args.eval_frequency,
        deterministic=True,
        render=False,
    )

    if args.model_path is not None:
        model = PPO.load(
            path=args.model_path, env=env, verbose=1, tensorboard_log=LOG_DIR
        )
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

    model.learn(
        total_timesteps=args.total_timesteps,
        reset_num_timesteps=False,
        progress_bar=True,
        tb_log_name=run_name,
        callback=eval_callback,
    )


def test(args):
    env = gym.make("LocoMujoco", env_name=args.env, render_mode="human")

    model = PPO.load(path=args.model_path, env=env, verbose=1)

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
    parser.add_argument("--env", type= str, default="UnitreeA1.simple.perfect")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval_frequency", type=int, default=25_000)
    parser.add_argument("--model_path", type= str, default=None, help="Path to the model to continue training")
    parser.add_argument("--run", type=str, default="train", choices=["train", "test"])
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.run == "train":
        train(args)
    elif args.run == "test":
        test(args)
