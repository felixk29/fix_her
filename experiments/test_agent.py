import torch
import os
import numpy as np
import gymnasium as gym
import wandb
import stable_baselines3 as sb3 

print("current path: ", (os.getcwd()))
print("Done importing!")

from four_room.env import FourRoomsEnv

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)


import dill
from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from adapted_vec_env import AdaptedVecEnv

from stable_baselines3.common.callbacks import EvalCallback
from wandb.integration.sb3 import WandbCallback
base_log = "./experiments/logs/"
os.makedirs(base_log, exist_ok=True)

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from tpdqn import tpDQN
from doubledqn import DoubleDQN


with open('./experiments/four_room/configs/fourrooms_train_config.pl', 'rb') as file:
    train_config = dill.load(file)

with open('./experiments/four_room/configs/fourrooms_test_0_config.pl', 'rb') as file:
    test_0_config = dill.load(file)

with open('./experiments/four_room/configs/fourrooms_test_100_config.pl', 'rb') as file:
    test_100_config = dill.load(file)

def make_env_fn(config, seed: int= 0, rank: int = 0):
    def _init():
        env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1', 
                    agent_pos=config['agent positions'], 
                    goal_pos=config['goal positions'], 
                    doors_pos=config['topologies'], 
                    agent_dir=config['agent directions']))
        env.reset(seed=seed+rank)
        return Monitor(env)
    return _init

#######  

num_envs=8

env=make_env_fn(train_config)()

train_env = DummyVecEnv([make_env_fn(train_config, seed=0, rank=i) for i in range(num_envs)])
train_env_tp = AdaptedVecEnv([make_env_fn(train_config, seed=0, rank=i) for i in range(num_envs)])

tr_eval_env = DummyVecEnv([make_env_fn(train_config, seed=0, rank=i) for i in range(num_envs)])
test_0_env = DummyVecEnv([make_env_fn(test_0_config, seed=0, rank=i) for i in range(num_envs)])
test_100_env = DummyVecEnv([make_env_fn(test_100_config, seed=0, rank=i) for i in range(num_envs)])


    ### Creating Agents ###

lr_schedule = lambda x: 0.00009*x+0.00001

class Baseline_CNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(Baseline_CNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    

wandb.tensorboard.patch(root_logdir="./experiments/logs/")


run=wandb.init(
    # set the wandb project where this run will be logged
    project="thesis",
    name="baseline",
    monitor_gym=True,
    sync_tensorboard=True,
    # track hyperparameters and run metadata
)

baseline_policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[512,256,128], 
                              features_extractor_class=Baseline_CNN, features_extractor_kwargs={'features_dim': 512},
                              optimizer_class=torch.optim.Adam, optimizer_kwargs={'weight_decay': 1e-5},
                              normalize_images=False)


baseline_model = DoubleDQN('CnnPolicy', train_env, buffer_size=100000, batch_size=256, gamma=0.99, learning_starts=256, 
                     gradient_steps=1, train_freq=(10, 'step'), target_update_interval=10, tau=0.01, exploration_fraction=0.2,
                     exploration_initial_eps=1.0, exploration_final_eps=0.01, max_grad_norm=1.0, learning_rate=1e-4,
                     verbose=0, tensorboard_log=f"runs/{run.id}/", policy_kwargs=baseline_policy_kwargs ,device='cuda')


eval_tr_callback = EvalCallback(tr_eval_env, log_path=base_log+"log_baseline/tr/", eval_freq=max(10000 // num_envs, 1),
                              n_eval_episodes=50, deterministic=True, render=False, verbose=0)

eval_0_callback = EvalCallback(test_0_env, log_path=base_log+"log_baseline/0/", eval_freq=max(10000 // num_envs, 1),
                              n_eval_episodes=50, deterministic=True, render=False, verbose=0)

eval_100_callback = EvalCallback(test_100_env, log_path=base_log+"log_baseline/100/", eval_freq=max(10000 // num_envs, 1),
                              n_eval_episodes=50, deterministic=True, render=False, verbose=0)

baseline_wandb_callback=WandbCallback(log='all', gradient_save_freq=1000)

baseline_model.learn(total_timesteps=500000,progress_bar=True, log_interval=10, callback=[eval_tr_callback,eval_0_callback,eval_100_callback,baseline_wandb_callback])

wandb.finish()

wandb.tensorboard.patch(root_logdir="./experiments/logs/")

run=wandb.init(
    # set the wandb project where this run will be logged
    project="thesis",
    name="tpdqn",
    monitor_gym=True,
    sync_tensorboard=True,
    # track hyperparameters and run metadata
)

tp_policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[512,256,128], 
                              features_extractor_class=Baseline_CNN, features_extractor_kwargs={'features_dim': 512},
                              optimizer_class=torch.optim.Adam, optimizer_kwargs={'weight_decay': 1e-5},
                              normalize_images=False)


tp_model = tpDQN('CnnPolicy', train_env_tp, buffer_size=100000, batch_size=256, gamma=0.99, learning_starts=256,
                     gradient_steps=1, train_freq=(10, 'step'), target_update_interval=10, tau=0.01, exploration_fraction=0.2,
                     exploration_initial_eps=1.0, exploration_final_eps=0.01, max_grad_norm=1.0, learning_rate=1e-4,
                     verbose=0, tensorboard_log=f"runs/{run.id}/", policy_kwargs=tp_policy_kwargs ,device='cuda')


eval_tr_callback = EvalCallback(tr_eval_env, log_path=base_log+"log_tp/tr/", eval_freq=max(10000 // num_envs, 1),
                              n_eval_episodes=50, deterministic=True, render=False, verbose=0)

eval_0_callback = EvalCallback(test_0_env, log_path=base_log+"log_tp/0/", eval_freq=max(10000 // num_envs, 1),
                              n_eval_episodes=50, deterministic=True, render=False, verbose=0)

eval_100_callback = EvalCallback(test_100_env, log_path=base_log+"log_tp/100/", eval_freq=max(10000 // num_envs, 1),
                              n_eval_episodes=50, deterministic=True, render=False, verbose=0)

tp_wandb_callback=WandbCallback(log='all', gradient_save_freq=1000)


tp_model.learn(total_timesteps=500000, progress_bar=True,  log_interval=10, callback=[eval_tr_callback,eval_0_callback,eval_100_callback, tp_wandb_callback])
print(tp_model.refilled)

