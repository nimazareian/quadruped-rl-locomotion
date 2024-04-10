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

# ENV_NAME = "UnitreeA1.simple.perfect"
# TOTAL_TRAINING_TIMESTEPS = 1_000_000
# EVAL_FREQUENCY = 25_000
NUM_PARALLEL_ENVS = 4
SEED = 0

MODEL_DIR = "models"
LOG_DIR = "logs"


# LocoMujoco reward helper functions: https://loco-mujoco.readthedocs.io/en/latest/source/loco_mujoco.utils.html#module-loco_mujoco.utils.reward
def my_reward_function(state, action, next_state):
    # TODO: Print the state, action, and next_state to understand the data structure
    # Could use VelocityVectorReward: https://github.com/robfiras/loco-mujoco/blob/c4f0e546725d5681a3ec865d3427ce5fdbb7526e/loco_mujoco/environments/quadrupeds/unitreeA1.py#L491
    # Power = Torque * Angular Velocity -> Minimize power/energy usage (i.e. reward -= power)
    
    # Observation & Action spaces: https://loco-mujoco.readthedocs.io/en/latest/source/loco_mujoco.environments.quadrupeds.html#unitree-a1
    # Fields and indices: https://github.com/robfiras/loco-mujoco/blob/4a9e87563e112b8da48a27cbe3df13d743efd830/loco_mujoco/environments/quadrupeds/unitreeA1.py#L48
    # Print the length of the variables and the variables values
    # len(action) = 12
    # len(state) = 44
    # len(next_state) = 44
    # Positive rewards, get standing.
    return -np.mean(action)  # here we just return the negative mean of the action


def train(args):
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
        env_name=args.env,
        action_mode="torque",
        setup_random_rot=False,
        default_target_velocity=0.5,
        reward_type="custom",
        reward_params=dict(reward_callback=my_reward_function),
    )

    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"ppo_loco_mujoco_{train_time}"

    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"{MODEL_DIR}/{run_name}",
        log_path=LOG_DIR,
        eval_freq=args.eval_frequency,
        deterministic=True,
        render=True,
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
    parser.add_argument("--total_timesteps", type=int, default=100_000)
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
