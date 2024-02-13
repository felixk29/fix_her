import torch
import os
import numpy as np
import gymnasium as gym
import wandb
import stable_baselines3 as sb3 
from gymnasium import spaces

from hergo import HERGO, MultiInput_CNN
print("current path: ", (os.getcwd()))
print("Done importing!")

from four_room.env import FourRoomsEnv

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

#### Paramampa
### EXPERIMT SETUP BUGGY; RUN THEM SOLO
num_envs=1
seed=0

import dill
from four_room.old_env import FourRoomsEnv
from four_room.wrappers import gym_wrapper

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from adapted_vec_env import AdaptedVecEnv
from stable_baselines3 import DQN

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
        if seed==0:
            env.reset()
        else:
            env.reset(seed=seed+rank)
        return Monitor(env)
    return _init

#######  

    ### Creating Agents ###

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
        if len(observations.shape) == 3:
            observations = observations.unsqueeze(0)
        return self.linear(self.cnn(observations))


from stable_baselines3.common.callbacks import BaseCallback
class ExplorationCoverageCallback(BaseCallback):
    def __init__(self, log_freq, total_states, num_actions, verbose=0):
        super(ExplorationCoverageCallback, self).__init__(verbose)
        self.state_action_coverage_set = set()
        self.log_freq = log_freq
        self.total_state_actions = total_states*num_actions

    def _on_step(self) -> bool:
        for i, obs in enumerate(self.locals['env'].buf_obs[None]):
            action = self.locals['actions'][i]
            self.state_action_coverage_set.add(hash((hash(obs.data.tobytes()), hash(action.data.tobytes()))))

        if self.num_timesteps % self.log_freq == 0:
            self.logger.record('train/state_action_coverage_exploration', len(self.state_action_coverage_set) / self.total_state_actions)

        return True


class UVFStepCounterCallback(BaseCallback):
    def __init__(self, log_freq, verbose=0):
        super(UVFStepCounterCallback, self).__init__(verbose)
        self.uvf_steps = set()
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        temp=set(self.locals['uvf_stepcount_history'])

        diff=self.uvf_steps-temp


        if self.num_timesteps % self.log_freq == 0:
            self.logger.record('train/uvf_stepcount_mean', sum([i[2] for i in diff])/len(diff))
            self.logger.record('train/uvf_stepcount_max', max([i[2] for i in diff]))
            self.logger.record('train/uvf_stepcount_min', min([i[2] for i in diff]))
            self.logger.record('train/uvf_stepcount_std', np.std([i[2] for i in diff]))

        self.uvf_steps=temp
        return True



### MultiInput_CNN for UVF ###
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

wandb.tensorboard.patch(root_logdir="./experiments/logs/")

from stable_baselines3 import HerReplayBuffer
#for bs, tpc, rn in [(500,0.0,0),(500,0.0,1),(500,1.0,3),(10,0.0,4),(50,0.5,0),(50,1.0,4)]:
for rn in range(10):
    for bs in [50]:
        for tpc in [1.0]:
            eps=0.1
            print("\n------------------------------------------------")
            print(f"Starting run {rn} with tp chance {tpc}, buffer size {bs}k")   
            print("------------------------------------------------\n")

            experiment=f"hergo_b{bs}k/"

            if eps==1.0:
                path=base_log+f"{experiment}/{tpc}"
            else:
                path=base_log+f"{experiment}/{tpc}_e{round(eps*100)}"

            cf={'policy': 'CnnPolicy',
                'buffer_size': bs*1000,
                'batch_size': 256,
                'gamma': 0.99,
                'learning_starts': 256,
                'max_grad_norm': 1.0,
                'gradient_steps': 1,
                'train_freq': (10//num_envs, 'step'),
                'target_update_interval': 10,
                'tau': 0.01,
                'exploration_fraction': eps,
                'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.01,
                'learning_rate': 1e-4,
                'verbose': 0,
                'device': 'cuda',
                'policy_config':{
                    'activation_fn': torch.nn.ReLU,
                    'net_arch': [128, 64],
                    'features_extractor_class': Baseline_CNN,
                    'features_extractor_kwargs':{'features_dim': 512},
                    'optimizer_class':torch.optim.Adam,
                    'optimizer_kwargs':{'weight_decay': 1e-5},
                    'normalize_images':False
                },
                'uvf_config':{
                    'kwargs':{
                    'features_extractor_class': MultiInput_CNN,
                    }
                },
                'tp_chance':tpc,
            }

            run=wandb.init(
                project="thesis",
                name=f"uvf_b{bs}k_{tpc}_{rn}",
                config=cf,
                monitor_gym=True,
                sync_tensorboard=True,
            )

            #train_env_tp = AdaptedVecEnv([make_env_fn(train_config, seed=seed, rank=i) for i in range(num_envs)])
            
            train_env_tp = DummyVecEnv([make_env_fn(train_config, seed=seed, rank=i) for i in range(num_envs)])
            tr_eval_env_tp = DummyVecEnv([make_env_fn(train_config, seed=seed, rank=i) for i in range(1)])
            test_0_env_tp = DummyVecEnv([make_env_fn(test_0_config, seed=seed, rank=i) for i in range(1)])
            test_100_env_tp = DummyVecEnv([make_env_fn(test_100_config, seed=seed, rank=i) for i in range(1)])

                    #tpDQN,HERGO,DoubleDQN
            model = HERGO(cf['policy'], train_env_tp, buffer_size=cf['buffer_size'], batch_size=cf['batch_size'], gamma=cf['gamma'], 
                                    learning_starts=cf['learning_starts'], gradient_steps=cf['gradient_steps'], train_freq=cf['train_freq'],
                                        target_update_interval=cf['target_update_interval'], tau=cf['tau'], exploration_fraction=cf['exploration_fraction'],
                                        exploration_initial_eps=cf['exploration_initial_eps'], exploration_final_eps=cf['exploration_final_eps'],
                                        max_grad_norm=cf['max_grad_norm'], learning_rate=cf['learning_rate'], verbose=cf['verbose'],
                                        tensorboard_log=f"runs/{run.id}/", policy_kwargs=cf['policy_config'] ,device=cf['device'],
                                        #tp_chance_start=cf['tp_chance'], tp_chance_end=cf['tp_chance'])  #tpdqn
                                        tp_chance=cf['tp_chance'],uvf_config=cf['uvf_config'])   #HERGO

            eval_tr_callback = EvalCallback(tr_eval_env_tp, log_path=f"{path}/tr/{rn}/", eval_freq=(25000//num_envs),
                                        n_eval_episodes=40, deterministic=True, render=False, verbose=0)

            eval_0_callback = EvalCallback(test_0_env_tp, log_path=f"{path}/0/{rn}/", eval_freq=(25000//num_envs),
                                        n_eval_episodes=40, deterministic=True, render=False, verbose=0)

            eval_100_callback = EvalCallback(test_100_env_tp, log_path=f"{path}/100/{rn}/", eval_freq=(25000//num_envs),
                                        n_eval_episodes=40, deterministic=True, render=False, verbose=0)
            
            state_action_coverage_callback = ExplorationCoverageCallback(1000, 6240, 3)
            step_counter_callback=UVFStepCounterCallback(1000)

            tp_wandb_callback=WandbCallback(log='all', gradient_save_freq=1000)

            model.learn(total_timesteps=500000, progress_bar=True,  log_interval=10, callback=[tp_wandb_callback, step_counter_callback,eval_tr_callback,eval_0_callback,eval_100_callback, state_action_coverage_callback])


            run.finish()
