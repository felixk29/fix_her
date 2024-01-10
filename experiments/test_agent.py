import torch
import os
import numpy as np
import gymnasium as gym

print("current path: ", (os.getcwd()))
print("Done importing!")

base_path=os.getcwd()+'/experiments/'


from four_room.env import FourRoomsEnv

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

## Creating Environments 2##
import dill
from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper

from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv


with open(f'{base_path}four_room/configs/fourrooms_train_config.pl', 'rb') as file:
    train_config = dill.load(file)

with open(f'{base_path}four_room/configs/fourrooms_test_0_config.pl', 'rb') as file:
    test_0_config = dill.load(file)

with open(f'{base_path}four_room/configs/fourrooms_test_100_config.pl', 'rb') as file:
    test_100_config = dill.load(file)

def make_env_fn(config, seed: int= 0, rank: int = 0):
    def _init():
        env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1', 
                    agent_pos=config['agent positions'], 
                    goal_pos=config['goal positions'], 
                    doors_pos=config['topologies'], 
                    agent_dir=config['agent directions']))
        env.reset(seed=seed+rank)
        return env
    return _init

#######  

num_envs=8

env=make_env_fn(train_config)()

train_env = DummyVecEnv([make_env_fn(train_config, seed=0, rank=i) for i in range(num_envs)])
train_env_tp = DummyVecEnv([make_env_fn(train_config, seed=0, rank=i) for i in range(num_envs)])

tr_ev_env = DummyVecEnv([make_env_fn(train_config, seed=0, rank=i) for i in range(num_envs)])
test_0_env = DummyVecEnv([make_env_fn(test_0_config, seed=0, rank=i) for i in range(num_envs)])
test_100_env = DummyVecEnv([make_env_fn(test_100_config, seed=0, rank=i) for i in range(num_envs)])

### Creating Agents ###

## Baseline 
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3 import DQN, tpDQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


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

baseline_policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[512,256,128], 
                              features_extractor_class=Baseline_CNN, features_extractor_kwargs={'features_dim': 512},
                              optimizer_class=torch.optim.Adam, 
                              normalize_images=False)

baseline_model = DQN('CnnPolicy', train_env, buffer_size=500000, batch_size=256, gamma=0.99, 
                     
                    learning_starts=1000, #idk what to set this at, for testing set at 100

                     gradient_steps=1, train_freq=(10, 'step'), target_update_interval=10, tau=0.01,
                     exploration_initial_eps=1.0, exploration_final_eps=0.01, max_grad_norm=1.0, learning_rate=lr_schedule,
                     verbose=1, tensorboard_log="./four_room/tensorboard/", policy_kwargs=baseline_policy_kwargs ,device='cuda')

tp_model = tpDQN('CnnPolicy', train_env_tp, buffer_size=500000, batch_size=256, gamma=0.99, 
                 
                    learning_starts=1000, #idk what to set this at, for testing set at 100

                     gradient_steps=1, train_freq=(10, 'step'), target_update_interval=10, tau=0.01,
                     exploration_initial_eps=1.0, exploration_final_eps=0.01, max_grad_norm=1.0, learning_rate=lr_schedule,
                     verbose=1, tensorboard_log="./four_room/tensorboard/", policy_kwargs=baseline_policy_kwargs ,device='cuda')

### Training Agents ###
from stable_baselines3.common.callbacks import EvalCallback
base_log = base_path +"logs/"
os.makedirs(base_log, exist_ok=True)

eval_tr_callback = EvalCallback(tr_ev_env, log_path=base_log+"log_baseline/tr/", eval_freq=max(1000 // num_envs, 1),
                              n_eval_episodes=20, deterministic=True, render=False)

eval_0_callback = EvalCallback(test_0_env, log_path=base_log+"log_baseline/0/", eval_freq=max(1000 // num_envs, 1),
                              n_eval_episodes=20, deterministic=True, render=False)

eval_100_callback = EvalCallback(test_100_env, log_path=base_log+"log_baseline/100/", eval_freq=max(1000 // num_envs, 1),
                              n_eval_episodes=20, deterministic=True, render=False)

#total_timesteps = 500000 should be used, will use 200000 for testing
#baseline_model.learn(total_timesteps=500000,progress_bar=True, log_interval=10, callback=[eval_tr_callback,eval_0_callback,eval_100_callback])

eval_tr_callback = EvalCallback(tr_ev_env, log_path=base_log+"log_tp/tr/", eval_freq=max(1000 // num_envs, 1),
                              n_eval_episodes=20, deterministic=True, render=False)

eval_0_callback = EvalCallback(test_0_env, log_path=base_log+"log_tp/0/", eval_freq=max(1000 // num_envs, 1),
                              n_eval_episodes=20, deterministic=True, render=False)

eval_100_callback = EvalCallback(test_100_env, log_path=base_log+"log_tp/100/", eval_freq=max(1000 // num_envs, 1),
                              n_eval_episodes=20, deterministic=True, render=False)

#total_timesteps = 500000 should be used, will use 200000 for testing
tp_model.learn(total_timesteps=500000, progress_bar=True,  log_interval=10, callback=[eval_tr_callback,eval_0_callback,eval_100_callback])

### Visualizing Results ###

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

results={}
base_log='./logs/'
def plot_reward(ax,dt, title):
    sns.lineplot(x=dt['timesteps'], y=dt['results'], ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Reward')
    ax.set_xlabel('Timesteps')
    ax.grid(True)

def plot_episode_length(ax,dt, title):
    sns.lineplot(x=dt['timesteps'], y=dt['ep_lengths'], ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Episode Length')
    ax.set_xlabel('Timesteps')
    ax.grid(True)

fig, ax = plt.subplots(3,2, figsize=(10,10))

for name in ['tp','baseline']:
    for env in ['tr', '0', '100']:
        tmp=np.load(base_log+'log_'+name+'/'+env+'/evaluations.npz')
        results[env]={}
        results[env]['results']=np.mean(tmp['results'],axis=1)
        results[env]['ep_lengths']=np.mean(tmp['ep_lengths'],axis=1)
        results[env]['timesteps']=tmp['timesteps']

    plot_reward(ax[0,0],results['tr'], 'Training Reward')
    plot_episode_length(ax[0,1],results['tr'], 'Training Episode Length')

    plot_reward(ax[1,0],results['100'], 'Test 100 Reward')
    plot_episode_length(ax[1,1],results['100'], 'Test 100 Episode Length')

    plot_reward(ax[2,0],results['0'], 'Test 0 Reward')
    plot_episode_length(ax[2,1],results['0'], 'Test 0 Episode Length')

plt.tight_layout()