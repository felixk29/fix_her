# Stops depcreated warnings
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
from four_room.wrappers import gym_wrapper
from four_room.utils import obs_to_state
from four_room.old_env import FourRoomsEnv
import dill
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure
from copy import deepcopy
from gymnasium.wrappers import TransformObservation
from gymnasium import ObservationWrapper
from stable_baselines3 import DQN
from stable_baselines3 import HerReplayBuffer
from RND import RND
from doubledqn import DoubleDQN
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from torch.nn import functional as F
from gymnasium import spaces
import torch as th
import numpy as np
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.type_aliases import MaybeCallback
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.buffers import ReplayBuffer
from fixedHER import FixedHerBuffer

import warnings
warnings.filterwarnings("ignore")

from utils import state_to_obs, heatmapCallback

# https://github.com/g1910/HyperNetworks
# https://github.com/JJGO/hyperlight
# https://github.com/shyamsn97/hyper-nn
class MultiInput_CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512, device=th.device("cuda")):
        super(MultiInput_CNN, self).__init__(observation_space, features_dim)
        self.device = device
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space['observation'].shape[0]*2
        self.cnn= nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            #

            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.ones(1, n_input_channels, *observation_space['observation'].shape[1:])).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: spaces.Dict) -> torch.Tensor:
        if len(observations['desired_goal'].shape) < 4:
            obs = observations['observation'].unsqueeze(0).to(self.device)
            goal = observations['desired_goal'].unsqueeze(0).to(self.device)
        else:
            obs = observations['observation'].to(self.device)
            goal = observations['desired_goal'].to(self.device)

        t=self.linear(self.cnn(th.cat([obs,goal],dim=1)))
        return t

class MultiInput_Hyper(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512, device=th.device("cuda")):
        super(MultiInput_Hyper, self).__init__(observation_space, features_dim)
        self.device = device
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space['observation'].shape[0]
        self.cnn_obs = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.cnn_goal = nn.Sequential(
            nn.Conv2d(n_input_channels,64, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn_obs(torch.ones(1, n_input_channels, *observation_space['observation'].shape[1:])).shape[1]
            n_flatten_goal = self.cnn_goal(torch.ones(1, n_input_channels, *observation_space['observation'].shape[1:])).shape[1]

        self.linear_obs = nn.Sequential(nn.Linear(n_flatten, 1024), nn.ReLU(),
                                        nn.Linear(1024, features_dim), nn.ReLU())
        
        
        self.linear_goal= nn.Sequential(nn.Linear(n_flatten_goal, features_dim*features_dim+features_dim), nn.ReLU())

    def forward(self, observations: spaces.Dict) -> torch.Tensor:

        if len(observations['desired_goal'].shape) < 4:
            obs = observations['observation'].unsqueeze(0).to(self.device)
            goal = observations['desired_goal'].unsqueeze(0).to(self.device)
        else:
            obs = observations['observation'].to(self.device)
            goal = observations['desired_goal'].to(self.device)

        obs=self.linear_obs(self.cnn_obs(obs))
        goal=self.linear_goal(self.cnn_goal(goal))

        weights=goal[:,:-self.features_dim].view(goal.shape[0],self.features_dim,self.features_dim)
        bias=goal[:,-self.features_dim:]
        
        #goal provides the weights last layer of obs
        combined=th.matmul(obs.unsqueeze(1),weights).squeeze(1)+bias

        return F.relu(combined)


class UVFWrapper(ObservationWrapper):
    time_spent=0.0

    def __init__(self, env, mode='train', direct_goal=False):
        assert mode in ['train', 'test0', 'test100'], 'mode should be either train or test0 or test100'

        super().__init__(env)

        self.direct_goal = direct_goal
        if self.direct_goal:
            self.observation_space = spaces.Dict(spaces={'observation': env.observation_space, 'achieved_goal': spaces.Box(
                low=0, high=1, shape=(2, 8,)), 'desired_goal': spaces.Box(low=0, high=1, shape=(2, 8,))})
        else:
            self.observation_space = spaces.Dict(
                spaces={'observation': env.observation_space, 'achieved_goal': env.observation_space, 'desired_goal': env.observation_space})
        self.goal = None
        self.goal_state = None
        self.goal_one_hot = None  # either (x,y) or (x*8+y)
        self.mode = mode

        self.train_env_starting_pos = [(1, 2), (1, 3), (1, 6), (1, 7), (2, 1), (2, 2), (3, 1), (3, 2), (
            3, 3), (3, 7), (5, 2), (5, 5), (5, 6), (5, 7), (6, 1), (6, 5), (6, 7), (7, 1), (7, 2), (7, 6)]
        self.train_context = [(7, 6, 0, 1, 2, 1), (2, 7, 1, 0, 0, 1), (2, 2, 0, 0, 1, 2), (2, 1, 0, 1, 1, 1), (6, 6, 0, 2, 0, 1), (6, 3, 2, 1, 2, 1), (3, 2, 2, 0, 1, 0), (3, 3, 2, 1, 0, 2),
                              (1, 6, 0, 2, 0, 2), (6, 1, 1, 1, 1, 0), (5, 5, 1, 1, 1, 1), (7, 6, 2, 0, 2, 1), (6, 1, 1, 1, 2, 1), (5, 3, 0, 0, 1, 1), (3, 7, 2, 2, 0, 0), (3, 6, 1, 2, 1, 1),
                              (2, 3, 1, 2, 0, 1), (2, 3, 1, 1, 0, 2), (3, 5, 1, 0, 1, 0), (2, 6, 2, 1, 1, 2), (7, 2, 1, 2, 0, 0), (1, 2, 1, 0, 2, 2), (6, 5, 1, 2, 2, 2), (5, 6, 0, 2, 0, 0),
                              (2, 3, 1, 0, 2, 0), (1, 3, 2, 2, 1, 2), (7, 5, 1, 0, 0, 0), (1, 5, 2, 0, 2, 2), (7, 5, 0, 1, 2, 2), (2, 5, 0, 2, 1, 0), (3, 1, 2, 2, 2, 1), (3, 5, 1, 1, 0, 0),
                              (2, 7, 0, 2, 1, 1), (7, 3, 0, 1, 1, 0), (3, 7, 1, 0, 1, 2), (6, 6, 2, 0, 0, 0), (3, 7, 2, 1, 0, 1), (1, 5, 2, 0, 1, 1), (6, 2, 0, 0, 2, 0), (1, 3, 1, 2, 2, 0)]
        return

    def observation(self, observation):
        if self.goal is None:
            goal = np.zeros_like(observation)
            dict_obs = {'observation': observation,
                        'achieved_goal': observation, 'desired_goal': goal}
        else:
            dict_obs = {'observation': observation,
                        'achieved_goal': observation, 'desired_goal': self.goal}

        return dict_obs

    def reset(self, **kwargs):
        obs, i = self.env.reset(**kwargs)
        t = obs_to_state(obs)

        # print(t)
        # usable numbers: 1,2,3 5,6,7
        # starting pos of train config:

        if self.mode == 'train':
            self.goal_state = self.generate_goal(t, False)

        elif self.mode == 'test0':
            self.goal_state = self.generate_goal(t, True)

        self.goal = state_to_obs(self.goal_state)

        return self.observation(obs), i
    
    def render(self):
        return self.env.render()

    def generate_goal(self, state, test=False):
        g_x = np.random.choice([1, 2, 3, 5, 6, 7])
        g_y = np.random.choice([1, 2, 3, 5, 6, 7])
        while g_x == state[0] and g_y == state[1]:
            g_x = np.random.choice([1, 2, 3, 5, 6, 7])
            g_y = np.random.choice([1, 2, 3, 5, 6, 7])

        if not test:
            goal_state = (g_x, g_y, *state[2:])
        else:
            og_x = np.random.choice([1, 2, 3, 5, 6, 7])
            og_y = np.random.choice([1, 2, 3, 5, 6, 7])
            dir = np.random.choice([0, 1, 2, 3])
            w = tuple([np.random.choice([0, 1, 2]) for i in range(4)])

            goal_state = (g_x, g_y, dir, og_x, og_y, *w)
            while goal_state[3:] in self.train_context:
                goal_state = self.generate_goal(state, test)
        return goal_state

    def step(self, action):
        obs, r, d, t, i = self.env.step(action)

        if self.agent_pos == self.goal_state[:2]:
            #r=1
            r = (0==np.random.randint(20)) if self.mode == 'train' else 1
            d = True
        else:
            if d==True:
                t=True
                d=False
            r = 0

        # assert d == (self.goal_state[:2] == self.agent_pos), f"somethings vewy wrong here, {self.mode}"
        # assert r == d

        return self.observation(obs), r, d, t, i

    def compute_reward(self, achieved_goal, desired_goal, info):
        temp = np.all(achieved_goal == desired_goal, axis=(1, 2, 3)).astype(float)
        # temp= np.array([samePos(achieved_goal[i],desired_goal[i]) for i in range(achieved_goal.shape[0])]).astype(float)
        return temp

from stable_baselines3.common.vec_env import VecVideoRecorder
import imageio.v3 as iio

class renderEpisodeCallback(BaseCallback):
    def __init__(self, save_dir: str = './experiments/logs/episodes', env = None, log_freq: int = 100000, verbose=0):
        super(renderEpisodeCallback, self).__init__(verbose)
        self.env=env
        self.log_freq = log_freq
        self.save_dir = save_dir
        self.once=True

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0  or self.once:
            self.once=False

            env=self.env

            imgs=[]
            obs,_ = env.reset()

            goal=obs_to_state(obs['desired_goal'])

            img=env.render()
            #
            done,trunc = False,False
            imgs.append(img)
            while not done and not trunc:
                action, _ = self.model.predict(obs)
                obs, _, done, trunc , _ = env.step(action)
                img=env.render()

                #marking goal
                img[(goal[1]*32)+1:(goal[1]+1)*32,(goal[0]*32)+1:(goal[0]+1)*32]=[0,0,255]

                imgs.append(img)
            iio.imwrite(f"behaviour{self.num_timesteps//self.log_freq}.gif",imgs,duration=100,loop=0)
        return True


if __name__ == "__main__":
    # callbacks can maybe be made in the on_step() just have to figue out when exactly its called
    from torchsummary import summary
    from four_room.env import FourRoomsEnv
    gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)
    from stable_baselines3.common.monitor import Monitor
    import wandb
    from stable_baselines3.common.callbacks import EvalCallback
    from wandb.integration.sb3 import WandbCallback
    import os
    base_log = "./experiments/logs/"
    os.makedirs(base_log, exist_ok=True)

    # TODO:
    # - check (64,) one hot encoding
    # - check compute_reward()
    # - check current behaviour of agent
    # - check different agents (TD3 or SAC or DDPG)
    # - run RandomWalk Experiment to get data (min 5 seeds)

    ###### CONFIGS ######

    num_envs = 10
    discrete_goal = False

    with open('./experiments/four_room/configs/fourrooms_train_config.pl', 'rb') as file:
        train_config = dill.load(file)

    with open('./experiments/four_room/configs/fourrooms_test_0_config.pl', 'rb') as file:
        test_0_config = dill.load(file)

    with open('./experiments/four_room/configs/fourrooms_test_100_config.pl', 'rb') as file:
        test_100_config = dill.load(file)

    def make_env_fn(config, seed: int = 0, rank: int = 0, mode='train', render=False):
        def _init():
            env = UVFWrapper(gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                                                  agent_pos=config['agent positions'],
                                                  goal_pos=config['goal positions'],
                                                  doors_pos=config['topologies'],
                                                  agent_dir=config['agent directions'],
                                                  render_mode='rgb_array' if render else None
                                                  )), mode=mode, direct_goal=discrete_goal)
            if seed == 0:
                env.reset()
            else:
                env.reset(seed=seed+rank)
            return Monitor(env)
        return _init

    # wandb.tensorboard.patch(root_logdir="./experiments/logs/")
    for rn in range(5):
        for eps in [-2]:#,0.9]:
            for bs in [500]:

                print("-------------------------------------------------------------------")
                print(f"Starting run {rn} with epsilon {eps} and buffer size {bs}k")
                print("-------------------------------------------------------------------")

                experiment = f"pure_uvf_b{bs}k/"

                path = base_log+f"{experiment}/{round(eps*100)}"

                dqn_cf={'buffer_size': bs*1000,
                    'batch_size': 256,
                    'learning_starts':512,
                    'gamma': 0.99,
                    'max_grad_norm': 1.0,
                    'gradient_steps': 1,
                    'train_freq': (10//num_envs, 'step'),
                    'target_update_interval': 10,
                    'tau': 0.01,
                    'exploration_fraction': 1.0,
                    'exploration_initial_eps': 1.0,
                    'exploration_final_eps': 0.01,
                    'learning_rate': 1e-3,
                    'verbose': 0,
                    'device': 'cuda',
                    'replay_buffer_class': FixedHerBuffer,
                    'replay_buffer_kwargs': {
                        'n_sampled_goal': 48,
                        'goal_selection_strategy': 'final', #future, episode and final #final best results 
                    },
                    'policy_kwargs':{
                        'activation_fn': torch.nn.ReLU,
                        'net_arch': [256,64],
                        'features_extractor_class': MultiInput_CNN,
                        'features_extractor_kwargs':{'features_dim': 512},
                        'optimizer_class':torch.optim.Adam,
                        'optimizer_kwargs':{'weight_decay': 1e-5},
                        'normalize_images':False
                    },
                }

                best_cf={'buffer_size': bs*1000,
                    'batch_size': 256,
                    'learning_starts':512,
                    'gamma': 0.99,
                    'max_grad_norm': 1.0,
                    'gradient_steps': 1,
                    'train_freq': (10//num_envs, 'step'),
                    'target_update_interval': 10,
                    'tau': 0.01,
                    'exploration_fraction': 0.5,
                    'exploration_initial_eps': 1.0,
                    'exploration_final_eps': 0.01,
                    'learning_rate': 1e-3,
                    'verbose': 0,
                    'device': 'cuda',
                    'replay_buffer_class': FixedHerBuffer,
                    'replay_buffer_kwargs': {
                        'n_sampled_goal': 48,
                        'goal_selection_strategy': 'final', #future, episode and final #final best results 
                    },
                    'policy_kwargs':{
                        'activation_fn': torch.nn.ReLU,
                        'net_arch': [256,64],
                        'features_extractor_class': MultiInput_CNN,
                        'features_extractor_kwargs':{'features_dim': 512},
                        'optimizer_class':torch.optim.Adam,
                        'optimizer_kwargs':{'weight_decay': 1e-5},
                        'normalize_images':False
                    },
                }
                # import cProfile, pstats
                # profiler = cProfile.Profile()
                # profiler.enable()


                # run=wandb.init(
                #     project="thesis",
                #     entity='felix-kaubek',
                #     name=f"pure_uve_b{bs}k_{rn}",
                #     config=dqn_cf,
                #     monitor_gym=True,
                #     sync_tensorboard=True,
                # )

                # TODO: Implement test case where both walking environment as well as target environment are different
                # for non discrete goals, change to train_config on all, to show that complicated goal representation works
                # (or actually just keep it train_config, as the uvf will only work on those door positions anyway (as its only used during training)
                train_env = DummyVecEnv([make_env_fn(train_config, seed=0, rank=i) for i in range(num_envs)])
                tr_eval_env = DummyVecEnv([make_env_fn(train_config, seed=0, rank=i)for i in range(1)])
                test_0_env = DummyVecEnv([make_env_fn(train_config, seed=0, rank=i, mode='test0') for i in range(1)])
                test_100_env = DummyVecEnv([make_env_fn(train_config, seed=0, rank=i) for i in range(1)])

                # wandb_callback=WandbCallback(log='all', gradient_save_freq=1000)

                eval_tr_callback = EvalCallback(tr_eval_env, log_path=f"{path}/tr/{rn}/", eval_freq=(25000//num_envs),
                                                n_eval_episodes=20, deterministic=True, render=False, verbose=0)

                eval_0_callback = EvalCallback(test_0_env, log_path=f"{path}/0/{rn}/", eval_freq=(25000//num_envs),
                                               n_eval_episodes=20, deterministic=True, render=False, verbose=0)

                eval_100_callback = EvalCallback(test_100_env, log_path=f"{path}/100/{rn}/", eval_freq=(25000//num_envs),
                                                 n_eval_episodes=4, deterministic=True, render=False, verbose=0)

                render_env=make_env_fn(train_config, seed=0, rank=0, render=True)()

                renderCallback=renderEpisodeCallback(log_freq=100000,env=render_env)

                heatmapping=heatmapCallback(log_freq=100000,id=f"bs{bs}ep{round(eps*10)}rn{rn}")

                model = DoubleDQN('MultiInputPolicy', train_env, **dqn_cf)#, tensorboard_log=f"runs/{run.id}/")

                model.learn(total_timesteps=500000, progress_bar=True,  log_interval=10, callback=[
                            #renderCallback])
                            eval_tr_callback, eval_0_callback, eval_100_callback,renderCallback, heatmapping])

                # run.finish()
                # adapt rollout to make sure that reaching env goal doesnt give a reward, but reaching the uvf goal does
                # profiler.disable()
                # stats = pstats.Stats(profiler)
                # stats.dump_stats(f"{path}/profile_{rn}.prof")