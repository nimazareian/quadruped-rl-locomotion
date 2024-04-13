import argparse
import os
import time

import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from go1_mujoco_env import Go1MujocoEnv
from tqdm import tqdm


NUM_LOGICAL_CPU_CORES = 12
SEED = 0

MODEL_DIR = "models"
LOG_DIR = "logs"


def train(args):
    vec_env = make_vec_env(
        Go1MujocoEnv,
        n_envs=NUM_LOGICAL_CPU_CORES,
        seed=SEED,
        vec_env_cls=SubprocVecEnv,
    )

    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"ppo_loco_mujoco_{train_time}"

    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=f"{MODEL_DIR}/{run_name}",
        log_path=LOG_DIR,
        eval_freq=args.eval_frequency,
        deterministic=True,
        render=False,
    )

    if args.model_path is not None:
        model = PPO.load(
            path=args.model_path, env=vec_env, verbose=1, tensorboard_log=LOG_DIR
        )
    else:
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR)

    model.learn(
        total_timesteps=args.total_timesteps,
        reset_num_timesteps=False,
        progress_bar=True,
        tb_log_name=run_name,
        callback=eval_callback,
    )


def test(args):
    env = Go1MujocoEnv(
        render_mode="human",
    )

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
            # env.render()
            time.sleep(1.0 / 60.0)

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
    parser.add_argument("--env", type=str, default="UnitreeA1.simple.perfect")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval_frequency", type=int, default=10_000)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model to continue training",
    )
    parser.add_argument("--run", type=str, default="train", choices=["train", "test"])
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.run == "train":
        train(args)
    elif args.run == "test":
        test(args)
