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
import warnings
warnings.filterwarnings("ignore")
from PPO_HER import HerRolloutBuffer

def state_to_obs(state):
    """
    Turn a state tuple back into a numpy observation array.
    """
    # Create an empty observation array
    obs = np.zeros((4, 9, 9))

    # Unpack the state tuple
    player_loc_x, player_loc_y, player_dir, goal_loc_x, goal_loc_y, door_pos_up, door_pos_down, door_pos_left, door_pos_right = state

    # Center the player location
    center_x, center_y = 4, 4  # Center of the 9x9 grid
    player_loc_x = center_x + (player_loc_x - center_x)
    player_loc_y = center_y + (player_loc_y - center_y)

    # Set the player location and direction
    obs[0, player_loc_y, player_loc_x] = 1
    if player_dir == 0:  # right
        obs[1, player_loc_y, player_loc_x+1] = 1
    elif player_dir == 1:  # down
        obs[1, player_loc_y+1, player_loc_x] = 1
    elif player_dir == 2:  # left
        obs[1, player_loc_y, player_loc_x-1] = 1
    elif player_dir == 3:  # up
        obs[1, player_loc_y-1, player_loc_x] = 1

    # Set the goal location
    obs[3, goal_loc_y, goal_loc_x] = 1

    # Set the walls and doors
    obs[2] = np.zeros((9, 9))
    obs[2, 0, :] = 1
    obs[2, 8, :] = 1
    obs[2, :, 0] = 1
    obs[2, :, 8] = 1
    obs[2, 4, :] = 1
    obs[2, :, 4] = 1

    obs[2, door_pos_up+1, 4] = 0
    obs[2, door_pos_down+5, 4] = 0
    obs[2, 4, door_pos_left+1] = 0
    obs[2, 4, door_pos_right+5] = 0

    # Shit so player is in center
    obs = np.roll(obs, (4-player_loc_x, 4-player_loc_y), axis=(2, 1))

    return obs


class MultiInput_CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512, device=th.device("cuda")):
        super(MultiInput_CNN, self).__init__(observation_space, features_dim)
        self.device = device
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space['observation'].shape[0]*2
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3,
                      stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,
                      padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,
                      padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():

            n_flatten = self.cnn(torch.ones(
                1, n_input_channels, *observation_space['observation'].shape[1:])).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), nn.ReLU())
        # self.linear = nn.Sequential(nn.Linear(n_flatten *2, features_dim), nn.ReLU())

    def forward(self, observations: spaces.Dict) -> torch.Tensor:
        if len(observations['desired_goal'].shape) < 4:
            obs = observations['observation'].unsqueeze(0).to(self.device)
            goal = observations['desired_goal'].unsqueeze(0).to(self.device)
        else:
            obs = observations['observation'].to(self.device)
            goal = observations['desired_goal'].to(self.device)

        # obs_cnn=self.cnn(obs)
        # goal_cnn=self.cnn(goal)
        # combined=torch.cat([obs_cnn,goal_cnn],axis=1)

        combined = self.cnn(torch.cat([obs, goal], axis=1))

        return self.linear(combined)


class UVFWrapper(ObservationWrapper):
    def __init__(self, env, mode='train', direct_goal=False):
        assert mode in [
            'train', 'test0', 'test100'], 'mode should be either train or test0 or test100'

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
            if self.direct_goal:
                goal = (np.zeros(8), np.zeros(8))
            else:
                goal = np.zeros_like(observation)
            dict_obs = {'observation': observation,
                        'achieved_goal': observation, 'desired_goal': goal}
        else:
            dict_obs = {'observation': observation,
                        'achieved_goal': observation, 'desired_goal': self.goal}
        if self.direct_goal:
            state = obs_to_state(observation)
            x_hot = np.zeros(8)
            y_hot = np.zeros(8)
            x_hot[state[0]] = 1
            y_hot[state[1]] = 1
            dict_obs['achieved_goal'] = [x_hot, y_hot]
            dict_obs['desired_goal'] = self.goal_one_hot

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

        if self.direct_goal:
            self.goal = [self.goal_state[0], self.goal_state[1]]
            x_hot = np.zeros(8)
            y_hot = np.zeros(8)
            x_hot[self.goal[0]] = 1
            y_hot[self.goal[1]] = 1
            self.goal_one_hot = [x_hot, y_hot]
        else:
            self.goal = state_to_obs(self.goal_state)

        return self.observation(obs), i

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
        if self.direct_goal:
            state = obs_to_state(obs)
            if state[0] == self.goal_state[0] and state[1] == self.goal_state[1]:
                r = 1
                d = True
                i['is_success'] = np.array([1.0])
            else:
                r = 0
                i['is_success'] = np.array([0.0])

        elif self.mode == 'train':
            if np.array_equal(obs, self.goal):
                r = 1
                d = True
                i['is_success'] = np.array([1.0])
            else:
                r = 0  # 1/(np.linalg.norm(self.goal-obs)**2)
                i['is_success'] = np.array([0.0])
        elif self.mode == 'test100':
            if np.array_equal(obs, self.goal):
                r = 1
                d = True
                i['is_success'] = np.array([1.0])
            else:
                r = 0
                i['is_success'] = np.array([0.0])
        elif self.mode == 'test0':
            if obs_to_state(obs)[:3] == self.goal_state[:3]:
                r = 1
                d = True
                i['is_success'] = np.array([1.0])
            else:
                r = 0
                i['is_success'] = np.array([0.0])

        return self.observation(obs), r, d, t, i

    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.direct_goal:
            temp = np.all(achieved_goal == desired_goal,
                          axis=(1, 2)).astype(float)
        else:
            temp = np.all(achieved_goal == desired_goal,
                          axis=(1, 2, 3)).astype(float)

        # temp= np.array([samePos(achieved_goal[i],desired_goal[i]) for i in range(achieved_goal.shape[0])]).astype(float)
        return temp


class heatmapCallback(BaseCallback):
    def __init__(self, save_dir: str = './experiments/logs/heatmaps', log_freq: int = 100000, verbose=0, id: str = ""):
        super(heatmapCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.save_dir = save_dir
        self.id = id
        self.dir_dict = {0: '→', 1: '↓', 2: '←',
                         3: '↑', -1: 'G', -2: 'X', -3: ' '}

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            tik = time()
            nr = self.num_timesteps//self.log_freq
            heatmap, annot = self.calculateHeatmap()

            max_val = np.max(heatmap)*1.05
            min_val = np.min(heatmap[heatmap > -90])
            min_val = min_val/2 if min_val > 0 else min_val
            sns.heatmap(heatmap, annot=annot, fmt='',
                        vmin=min_val, vmax=max_val)
            plt.savefig(f"{self.save_dir}/heatmap{self.id}_{nr}.png")
            plt.clf()
            print(f"Heatmap generated. Took {round(time()-tik,2)}s")

        return True

    def calculateHeatmap(self):
        nn = self.locals['self'].q_net
        goal = self.locals['self']._last_obs['desired_goal'][0]

        env = obs_to_state(goal)
        walls = env[-4:]  # walls are up down left right range= 0,1,2

        heatmap = np.zeros((9, 9, 4))-100

        for y in range(1, 8):
            for x in range(1, 8):
                if (x == 4 or y == 4) and not ((x, y) in [(4, 1+walls[0]), (4, 5+walls[1]), (1+walls[2], 4), (5+walls[3], 4)]):
                    continue  # checks so only doorways are calculated

                for dir in range(4):  # dir 0-3 : right down left up
                    obs = state_to_obs((x, y, dir, *env[3:]))
                    dic = {'observation': th.Tensor(obs), 'achieved_goal': th.Tensor(
                        obs), 'desired_goal': th.Tensor(goal)}

                    heatmap[y, x, dir] = th.max(nn(dic)).cpu().detach().item()

        values_2d = np.max(heatmap, axis=2)
        annot = np.argmax(heatmap, axis=2)
        annot[values_2d == -100] = -3
        annot[env[4], env[3]] = -2
        annot[env[1], env[0]] = -1
        annot = np.array([self.dir_dict[i]
                         for i in annot.reshape((81))]).reshape((9, 9))

        return values_2d, annot


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
    from PPO_HER import HER_PPO

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

    def make_env_fn(config, seed: int = 0, rank: int = 0, mode='train'):
        def _init():
            env = UVFWrapper(gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                                                  agent_pos=config['agent positions'],
                                                  goal_pos=config['goal positions'],
                                                  doors_pos=config['topologies'],
                                                  agent_dir=config['agent directions'])), mode=mode, direct_goal=discrete_goal)
            if seed == 0:
                env.reset()
            else:
                env.reset(seed=seed+rank)
            return Monitor(env)
        return _init

    # wandb.tensorboard.patch(root_logdir="./experiments/logs/")
    for rn in range(10,15):
        for eps in [0.2,0.4]:#,0.9]:
            for bs in [50]:

                print("-------------------------------------------------------------------")
                print(f"Starting run {rn} with epsilon {eps} and buffer size {bs}k")
                print("-------------------------------------------------------------------")

                experiment = f"pure_uvf_b{bs}k/"

                path = base_log+f"{experiment}/{round(eps*100)}"

                # dqn_cf={'buffer_size': bs*1000,
                #     'batch_size': 256,
                #     'learning_starts':256*2,
                #     'gamma': 0.99,
                #     'max_grad_norm': 1.0,
                #     'gradient_steps': 1,
                #     'train_freq': (10//num_envs, 'step'),
                #     'target_update_interval': 10,
                #     'tau': 0.01,
                #     'exploration_fraction': 0.5,
                #     'exploration_initial_eps': 1.0,
                #     'exploration_final_eps': 0.1,
                #     'learning_rate': 3e-4,
                #     'verbose': 0,
                #     'device': 'cuda',
                #     'replay_buffer_class': HerReplayBuffer,
                #     'replay_buffer_kwargs': {
                #         'n_sampled_goal': 16 if eps==0.5 else 8,
                #         'goal_selection_strategy': 'future',
                #     },
                #     'policy_kwargs':{
                #         'activation_fn': torch.nn.ReLU,
                #         'net_arch': [512 ,64],
                #         'features_extractor_class': MultiInput_CNN,
                #         'features_extractor_kwargs':{'features_dim': 1024},
                #         'optimizer_class':torch.optim.Adam,
                #         'optimizer_kwargs':{'weight_decay': 1e-5},
                #         'normalize_images':False
                #     },
                # }


                ppo_cf = {
                    'her_val':eps,
                    'learning_rate': 3e-4,
                    'n_steps': 256, ## will be multiplied with num_envs
                    'batch_size': 256,
                    'n_epochs': 10,
                    'clip_range': 0.15,

                    # # TODO add HER replay buffer as extension of DictRolloutBuffer
                    # 'rollout_buffer_class': HerRolloutBuffer,
                    # 'rollout_buffer_kwargs': {
                    #     'her_ratio':eps
                    #     },
                    'policy_kwargs': {
                        'activation_fn': torch.nn.ReLU,
                        'net_arch': [512, 64],
                        'features_extractor_class': MultiInput_CNN,
                        'features_extractor_kwargs': {'features_dim': 1024},
                        'optimizer_class': torch.optim.Adam,
                        'optimizer_kwargs': {'weight_decay': 1e-5},
                        'normalize_images': False
                    },
                }

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
                train_env = DummyVecEnv(
                    [make_env_fn(train_config, seed=0, rank=i) for i in range(num_envs)])
                tr_eval_env = DummyVecEnv(
                    [make_env_fn(train_config, seed=0, rank=i)for i in range(1)])
                test_0_env = DummyVecEnv(
                    [make_env_fn(train_config, seed=0, rank=i, mode='test0') for i in range(1)])
                test_100_env = DummyVecEnv(
                    [make_env_fn(train_config, seed=0, rank=i) for i in range(1)])

                # wandb_callback=WandbCallback(log='all', gradient_save_freq=1000)

                eval_tr_callback = EvalCallback(tr_eval_env, log_path=f"{path}/tr/{rn}/", eval_freq=(10000//num_envs),
                                                n_eval_episodes=20, deterministic=True, render=False, verbose=0)

                eval_0_callback = EvalCallback(test_0_env, log_path=f"{path}/0/{rn}/", eval_freq=(10000//num_envs),
                                               n_eval_episodes=20, deterministic=True, render=False, verbose=0)

                eval_100_callback = EvalCallback(test_100_env, log_path=f"{path}/100/{rn}/", eval_freq=(5000//num_envs),
                                                 n_eval_episodes=4, deterministic=True, render=False, verbose=0)

                # heatmapping=heatmapCallback(log_freq=100000,id=f"bs{bs}ep{round(eps*10)}rn{rn}")

                #model = DoubleDQN('MultiInputPolicy', train_env, **dqn_cf)#, tensorboard_log=f"runs/{run.id}/")
                model = HER_PPO('MultiInputPolicy', train_env,**ppo_cf)

                model.learn(total_timesteps=200000, progress_bar=True,  log_interval=10, callback=[
                            eval_tr_callback, eval_0_callback, eval_100_callback])

                # run.finish()
                # adapt rollout to make sure that reaching env goal doesnt give a reward, but reaching the uvf goal does
