# Training Quadruped Locomotion using Reinforcement Learning in Mujoco

https://github.com/nimazareian/quadruped-rl-locomotion/assets/28585597/262b7812-0b8f-4758-aedd-a429f743fb69


A custom gymnasium environment for training quadruped locomotion using reinforcement learning in the Mujoco simulator. The environment has been set up for the Unitree Go1 robot, however, it can be easily extended to train other robots as well. 

There are two MJCF models provided for the Go1 robot. One tuned for position control with a proportional controller, and one model which directly takes in torque values for end-to-end training.

## Setup
```bash
python -m pip install -r requirements.txt
```

## Train
```bash
python train.py --run train
```

## Displaying Trained Models 

```bash
python train.py --run test --model_path <path to model zip file>
```
For example, to run a pretrained model which outputs motor torques and has the robot desired velocity set to <x=1, y=0>, you can run:
```bash
python train.py --run test --model_path .\models\2024-04-16_10-11-57-x=1_torque_ctrl_fixed_joint_range_5mill_iter_working_well\final_model.zip
```

<details>
  <summary>Additional arguments for customizing training and testing</summary>

    usage: train.py [-h] --run {train,test} [--run_name RUN_NAME] [--num_parallel_envs NUM_PARALLEL_ENVS]
                    [--num_test_episodes NUM_TEST_EPISODES] [--record_test_episodes] [--total_timesteps TOTAL_TIMESTEPS]      
                    [--eval_frequency EVAL_FREQUENCY] [--model_path MODEL_PATH] [--seed SEED]

    optional arguments:
    -h, --help            show this help message and exit
    --run {train,test}
    --run_name RUN_NAME   Custom name of the run. Note that all runs are saved in the 'models' directory and have the       
                            training time prefixed.
    --num_parallel_envs NUM_PARALLEL_ENVS
                            Number of parallel environments while training
    --num_test_episodes NUM_TEST_EPISODES
                            Number of episodes to test the model
    --record_test_episodes
                            Whether to record the test episodes or not. If false, the episodes are rendered in the window.    
    --total_timesteps TOTAL_TIMESTEPS
                            Number of timesteps to train the model for
    --eval_frequency EVAL_FREQUENCY
                            The frequency of evaluating the models while training
    --model_path MODEL_PATH
                            Path to the model (.zip)
    --seed SEED

</details>

## Note

This repository serves for education purposes and is by no mean finalized!
