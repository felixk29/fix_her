from stable_baselines3.common.buffers import ReplayBuffer
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.dqn.policies import DQNPolicy
from doubledqn import DoubleDQN
from RND import RND
from stable_baselines3 import HerReplayBuffer
from stable_baselines3 import DQN
from gymnasium import ObservationWrapper
from gymnasium.wrappers import TransformObservation
from copy import deepcopy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
from stable_baselines3.common.vec_env import DummyVecEnv
import dill
from four_room.old_env import FourRoomsEnv
from four_room.utils import obs_to_state
from four_room.wrappers import gym_wrapper
import gymnasium as gym

SelfUVF = TypeVar("SelfUVF", bound="UVF")

class UVF(DoubleDQN):

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = HerReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        #self added
        uvf_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0

        if _init_setup_model:
            self._setup_model()
    
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

    #Shit so player is in center
    obs=np.roll(obs, (4-player_loc_x,4-player_loc_y), axis=(2,1))

    return obs

class MultiInput_CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512, device=th.device("cuda")):
        super(MultiInput_CNN, self).__init__(observation_space, features_dim)
        self.device=device
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space['observation'].shape[0]*2
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad(): 
            
            n_flatten = self.cnn(torch.ones(1,n_input_channels ,*observation_space['observation'].shape[1:])).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten , features_dim), nn.ReLU())


    def forward(self, observations: spaces.Dict) -> torch.Tensor:
        if len(observations['desired_goal'].shape)<4:
            obs=observations['observation'].unsqueeze(0).to(self.device)
            goal=observations['desired_goal'].unsqueeze(0).to(self.device)
        else:
            obs=observations['observation'].to(self.device)
            goal=observations['desired_goal'].to(self.device)

        # obs_cnn=self.cnn(obs)
        # goal_cnn=self.cnn(goal)
        # combined=torch.cat([obs_cnn,goal_cnn],axis=1)
        
        combined=self.cnn(torch.cat([obs,goal],axis=1))

        return self.linear(combined)

#TODO update to one hot encoded goal (either 2x8, or 64 array)
class MultiInput_CNN_Goal(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512, device=th.device("cuda")):
        super(MultiInput_CNN_Goal, self).__init__(observation_space, features_dim)
        self.device=device
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        
        # input= observation(4,9,9)
        n_input_channels = observation_space['observation'].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Flatten(),
        )        
        # input= goal(2,) -> goal(16,) as both x and y one hot encoded
        self.lin_goal=nn.Sequential(nn.Linear(16, 64), nn.ReLU())

        with torch.no_grad():
            n_flatten = self.cnn(torch.ones(1,n_input_channels,*observation_space['observation'].shape[1:])).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten + 64, features_dim), nn.ReLU())

    def forward(self, observations: spaces.Dict) -> torch.Tensor:
        obs=observations['observation']
        if len(observations['observation'].shape)<4:
            obs=obs.unsqueeze(0)

        goal=observations['desired_goal']
        goal_stack=torch.cat([goal[:,0,:],goal[:,1,:]],axis=1)

        processed_obs=self.cnn(obs)
        processed_goal=self.lin_goal(goal_stack)

        return self.linear(torch.cat([processed_obs, processed_goal], axis=1))


class UVFWrapper(ObservationWrapper):
    def __init__(self, env, mode='train',direct_goal=False):
        assert mode in ['train', 'test0', 'test100'], 'mode should be either train or test0 or test100'

        super().__init__(env)

        self.direct_goal=direct_goal
        if self.direct_goal:
            self.observation_space= spaces.Dict(spaces={'observation':env.observation_space, 'achieved_goal':spaces.Box(low=0, high=1, shape=(2,8,)), 'desired_goal':spaces.Box(low=0, high=1, shape=(2,8,))})
        else:
            self.observation_space= spaces.Dict(spaces={'observation':env.observation_space, 'achieved_goal':env.observation_space, 'desired_goal':env.observation_space})
        self.goal=None
        self.goal_state=None
        self.goal_one_hot=None #either (x,y) or (x*8+y)
        self.mode=mode

        self.train_env_starting_pos=[(1, 2), (1, 3), (1, 6), (1, 7), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (3, 7), (5, 2), (5, 5), (5, 6), (5, 7), (6, 1), (6, 5), (6, 7), (7, 1), (7, 2), (7, 6)]

        return 

    def observation(self, observation):
        if self.goal is None:
            if self.direct_goal:
                goal=(np.zeros(8),np.zeros(8))
            else:
                goal=np.zeros_like(observation)
            dict_obs={'observation':observation, 'achieved_goal':observation, 'desired_goal':goal}
        else:
            dict_obs={'observation':observation, 'achieved_goal':observation, 'desired_goal':self.goal}
        if self.direct_goal:
            state=obs_to_state(observation)
            x_hot=np.zeros(8)
            y_hot=np.zeros(8)
            x_hot[state[0]]=1
            y_hot[state[1]]=1
            dict_obs['achieved_goal']=[x_hot,y_hot]
            dict_obs['desired_goal']=self.goal_one_hot

        return dict_obs

    def reset(self, **kwargs):
        obs, i =self.env.reset(**kwargs)
        t=obs_to_state(obs)

        #print(t)
        #usable numbers: 1,2,3 5,6,7
        #starting pos of train config: 

        if self.mode=='train':
            s = np.random.choice(4)
            g_x , g_y=[(3,6),(5,3),(7,4),(1,1)][s]
            self.goal_state=(g_x,g_y,*t[2:])

        elif self.mode=='test100':
            s = np.random.choice(4)
            g_x , g_y=[(3,5),(1,4),(7,3),(5,1)][s]
            self.goal_state=(g_x,g_y,*t[2:])

        elif self.mode=='test0':
            
            g_x=np.random.choice([1,2,3,5,6,7])
            g_y=np.random.choice([1,2,3,5,6,7])
            if g_x==t[0] and g_y==t[1]:
                g_x=np.random.choice([1,2,3,5,6,7])
                g_y=np.random.choice([1,2,3,5,6,7])

            w=tuple([np.random.choice([0,1,2]) for i in range(4)])
            self.goal_state=(g_x,g_y,*t[2:5],*w)

        if self.direct_goal: 
            self.goal=[self.goal_state[0],self.goal_state[1]]
            x_hot=np.zeros(8)
            y_hot=np.zeros(8)
            x_hot[self.goal[0]] = 1
            y_hot[self.goal[1]] = 1
            self.goal_one_hot=[x_hot,y_hot]
        else:
            self.goal=state_to_obs(self.goal_state)

        return self.observation(obs), i

    def step(self, action):
        obs, r, d, t, i = self.env.step(action)
        if self.direct_goal:
            state=obs_to_state(obs)
            if state[0]==self.goal_state[0] and state[1]==self.goal_state[1]:
                r=1
                d=True
                i['is_success']=np.array([1.0])
            else:
                r=0
                i['is_success']=np.array([0.0])

        elif self.mode=='train':
            if np.array_equal(obs, self.goal):
                r=1
                d=True
                i['is_success']=np.array([1.0])
            else:
                r=0#1/(np.linalg.norm(self.goal-obs)**2)
                i['is_success']=np.array([0.0])
        elif self.mode=='test100':
            if np.array_equal(obs, self.goal):
                r=1
                d=True
                i['is_success']=np.array([1.0])
            else:
                r=0
                i['is_success']=np.array([0.0])
        elif self.mode=='test0':
            if obs_to_state(obs)[:2]==self.goal_state[:2]:
                r=1
                d=True
                i['is_success']=np.array([1.0])
            else:
                r=0
                i['is_success']=np.array([0.0])

        return self.observation(obs), r, d, t, i

    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.direct_goal:
            temp=np.all(achieved_goal==desired_goal, axis=(1,2)).astype(float)
        else:
            temp=np.all(achieved_goal==desired_goal, axis=(1,2,3)).astype(float)

            
        #temp= np.array([samePos(achieved_goal[i],desired_goal[i]) for i in range(achieved_goal.shape[0])]).astype(float)
        return temp
    


if __name__=="__main__":
    ## callbacks can maybe be made in the on_step() just have to figue out when exactly its called
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

    ###### CONFIGS ######
    # TODO: 
    # - check (64,) one hot encoding
    # - check compute_reward()
    # - check current behaviour of agent
    # - check different agents (TD3 or SAC or DDPG)
    # - run RandomWalk Experiment to get data (min 5 seeds)


    num_envs=10
    discrete_goal=False

    with open('./experiments/four_room/configs/fourrooms_train_config.pl', 'rb') as file:
        train_config = dill.load(file)

    with open('./experiments/four_room/configs/fourrooms_test_0_config.pl', 'rb') as file:
        test_0_config = dill.load(file)

    with open('./experiments/four_room/configs/fourrooms_test_100_config.pl', 'rb') as file:
        test_100_config = dill.load(file)

    def make_env_fn(config, seed: int= 0, rank: int = 0, mode='train'):
        def _init():
            env = UVFWrapper(gym_wrapper(gym.make('MiniGrid-FourRooms-v1', 
                        agent_pos=config['agent positions'], 
                        goal_pos=config['goal positions'], 
                        doors_pos=config['topologies'], 
                        agent_dir=config['agent directions'])),mode=mode, direct_goal=discrete_goal)
            if seed==0:
                env.reset()
            else:
                env.reset(seed=seed+rank)
            return Monitor(env)
        return _init


    wandb.tensorboard.patch(root_logdir="./experiments/logs/")
    for rn in range(10):
        for eps in [0.5,1.0]:
            for bs in [50]:
                # TODO:
                # - investigate the starting positions of training in relation to goal positions of test
                # - try it out with on ehot encoded goal

                print("-------------------------------------------------------------------")
                print(f"Starting run {rn} with epsilon {eps} and buffer size {bs}k")
                print("-------------------------------------------------------------------")

                experiment=f"pure_uvf_b{bs}k/"

                path=base_log+f"{experiment}/{round(eps*100)}"

                cf={'policy': 'MultiInputPolicy',
                    'buffer_size': bs*1000,
                    'batch_size': 256,
                    'gamma': 0.99,
                    'max_grad_norm': 1.0,
                    'gradient_steps': 1,
                    'train_freq': (10//num_envs, 'step'),
                    'target_update_interval': 10,
                    'tau': 0.01,
                    'exploration_fraction': eps,
                    'exploration_initial_eps': 1.0,
                    'exploration_final_eps': 0.01,
                    'learning_rate': 5e-4,
                    'verbose': 0,
                    'device': 'cuda',
                    'replay_buffer_class': HerReplayBuffer,
                    'replay_buffer_kwargs': {
                        'n_sampled_goal': 6, 
                        'goal_selection_strategy': 'episode', 
                    },
                    'policy_config':{
                        'activation_fn': torch.nn.ReLU,
                        'net_arch': [128, 64],
                        'features_extractor_class': MultiInput_CNN_Goal if discrete_goal else MultiInput_CNN,
                        'features_extractor_kwargs':{'features_dim': 512},
                        'optimizer_class':torch.optim.Adam,
                        'optimizer_kwargs':{'weight_decay': 1e-5},
                        'normalize_images':False
                    },
                }

                run=wandb.init(
                    project="thesis",
                    name=f"pure_uve_b{bs}k_{rn}",
                    config=cf,
                    monitor_gym=True,
                    sync_tensorboard=True,
                )


                #for non discrete goals, change to train_config on all, to show that complicated goal representation works 
                #(or actually just keep it train_config, as the uvf will only work on those door positions anyway (as its only used during training)
                train_env = DummyVecEnv([make_env_fn(train_config, seed=0, rank=i) for i in range(num_envs)])
                tr_eval_env = DummyVecEnv([make_env_fn(train_config, seed=0, rank=i)for i in range(1)])
                test_0_env = DummyVecEnv([make_env_fn(train_config, seed=0, rank=i,mode='test100') for i in range(1)])
                test_100_env = DummyVecEnv([make_env_fn(train_config, seed=0, rank=i,mode='test0') for i in range(1)])

                wandb_callback=WandbCallback(log='all', gradient_save_freq=1000)

                eval_tr_callback = EvalCallback(tr_eval_env, log_path=f"{path}/tr/{rn}/", eval_freq=(25000//num_envs),
                                            n_eval_episodes=20, deterministic=True, render=False, verbose=0)

                eval_0_callback = EvalCallback(test_0_env, log_path=f"{path}/0/{rn}/", eval_freq=(25000//num_envs),
                                            n_eval_episodes=20, deterministic=True, render=False, verbose=0)

                eval_100_callback = EvalCallback(test_100_env, log_path=f"{path}/100/{rn}/", eval_freq=(25000//num_envs),
                                            n_eval_episodes=20, deterministic=True, render=False, verbose=0)


                model = UVF(cf['policy'], train_env, replay_buffer_class=cf['replay_buffer_class'],replay_buffer_kwargs=cf['replay_buffer_kwargs'], 
                                        buffer_size=cf['buffer_size'], batch_size=cf['batch_size'], gamma=cf['gamma'], 
                                        learning_starts=cf['batch_size']*2, gradient_steps=cf['gradient_steps'], train_freq=cf['train_freq'],
                                        target_update_interval=cf['target_update_interval'], tau=cf['tau'], exploration_fraction=cf['exploration_fraction'],
                                        exploration_initial_eps=cf['exploration_initial_eps'], exploration_final_eps=cf['exploration_final_eps'],
                                        max_grad_norm=cf['max_grad_norm'], learning_rate=cf['learning_rate'], verbose=cf['verbose'],
                                        policy_kwargs=cf['policy_config'] ,device=cf['device'],tensorboard_log=f"runs/{run.id}/")

                
                

                model.learn(total_timesteps=500000, progress_bar=True,  log_interval=10,callback=[wandb_callback, eval_tr_callback, eval_0_callback,eval_100_callback])

                run.finish()
                ## adapt rollout to make sure that reaching env goal doesnt give a reward, but reaching the uvf goal does