import warnings
warnings.filterwarnings("ignore")
#pls dont hurt me


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
import cProfile, pstats

from utils import ExplorationCoverageCallback, UVFStepCounterCallback, randomStepsCallback
from four_room.env import FourRoomsEnv

from intrinsicRandomWalk import IntrinsicRandomWalk

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

#### Paramampa
### EXPERIMT SETUP BUGGY; RUN THEM SOLO
num_envs=1
seed=0
PROFILING=False
LOGGING=False


import dill
from four_room.old_env import FourRoomsEnv
from four_room.wrappers import gym_wrapper
from four_room.utils import obs_to_state

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
from moveRandom import RandomStart
from rnd_dqn import RND_DQN

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


device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
wandb.tensorboard.patch(root_logdir="./experiments/logs/")

from copy import deepcopy
base_cf={'policy':'CnnPolicy',
                    'batch_size': 256,
                    'gamma': 0.99,
                    'learning_starts': 256,
                    'max_grad_norm': 1.0,
                    'gradient_steps': 1,
                    'train_freq': (10//num_envs, 'step'),
                    'target_update_interval': 10,
                    'tau': 0.01,
                    'exploration_fraction': 0.5,
                    'exploration_initial_eps': 1.0,
                    'exploration_final_eps': 0.1,
                    'learning_rate': 2.5e-4,
                    'verbose': 0,
                    'device': 'cuda',
                    'policy_kwargs':{
                        'activation_fn': torch.nn.ReLU,
                        'net_arch': [128, 64],
                        'features_extractor_class': Baseline_CNN,
                        'features_extractor_kwargs':{'features_dim': 512},
                        'optimizer_class':torch.optim.Adam,
                        'optimizer_kwargs':{'weight_decay': 1e-5},
                        'normalize_images':False
                    }
                    #,'random_walk_duration':tpc,
                    #,'tp_chance':tpc,
                    #,'beta':l
                    #,'random_steps':15
                }

model_bases={'tp':tpDQN,'hergo':HERGO,'base':DoubleDQN, 'randomStart':RandomStart, 'intrinsicRandomWalk':IntrinsicRandomWalk}

from stable_baselines3 import HerReplayBuffer
#for bs, tpc, rn in [(500,0.0,0),(500,0.0,1),(500,1.0,3),(10,0.0,4),(50,0.5,0),(50,1.0,4)]:
if __name__ == "__main__":

    for rn in range(10):
        for bs in [500]:
            for method in ['intrinsicRandomWalk','tp','hergo','base','randomStart'][:1]:
                tpc="Thursday"
                eps=2.0
                print("\n------------------------------------------------")
                print(f"Starting {method} run {rn} under name \"{tpc}\", buffer size {bs}k, exploration fraction of {eps}")   
                print("------------------------------------------------\n")

                experiment=f"{method}_b{bs}k/"

                train_env_tp = AdaptedVecEnv([make_env_fn(train_config, seed=seed, rank=i) for i in range(num_envs)])
                
                train_env = DummyVecEnv([make_env_fn(train_config, seed=seed, rank=i) for i in range(num_envs)])
                tr_eval_env = DummyVecEnv([make_env_fn(train_config, seed=seed, rank=i) for i in range(1)])
                test_0_env = DummyVecEnv([make_env_fn(test_0_config, seed=seed, rank=i) for i in range(1)])
                test_100_env = DummyVecEnv([make_env_fn(test_100_config, seed=seed, rank=i) for i in range(1)])


                path=base_log+f"{experiment}/{tpc}_e{round(eps*100)}"
                cf=deepcopy(base_cf)
                cf['buffer_size']=bs*1000


                if method=='hergo':
                    cf['env']=train_env
                    cf['tp_chance']=1.0

                elif method=='intrinsicRandomWalk':
                    cf['env']=train_env
                    cf['beta']=200
                    cf['random_steps']=10  #to be double checked

                elif method=='tp':
                    cf['env']=train_env_tp
                    cf['tp_chance']=1.0

                elif method=='randomStart':
                    cf['env']=train_env
                    cf['random_walk_duration']=15 

                else:
                    cf['env']=train_env


                if LOGGING:
                    run=wandb.init(
                        project="thesis",
                        entity='felix-kaubek',
                        name=f"{method}_b{bs}k_{tpc}_{rn}",
                        config=cf,
                        monitor_gym=True,
                        sync_tensorboard=True,
                    )
                    cf['tensorboard_log']=f"runs/{run.id}/"


                if PROFILING:
                    profiler = cProfile.Profile()
                    profiler.enable()

                        #tpDQN,HERGO,DoubleDQN, RandomStart, , IntrinsicRandomWalk
                #model = HERGO(**cf)
                model_base=model_bases[method]
                model = model_base(**cf)



                eval_tr_callback = EvalCallback(tr_eval_env, log_path=f"{path}/tr/{rn}/", eval_freq=(50000//num_envs),
                                            n_eval_episodes=20, deterministic=True, render=False, verbose=0)

                eval_0_callback = EvalCallback(test_0_env, log_path=f"{path}/0/{rn}/", eval_freq=(50000//num_envs),
                                            n_eval_episodes=20, deterministic=True, render=False, verbose=0)

                eval_100_callback = EvalCallback(test_100_env, log_path=f"{path}/100/{rn}/", eval_freq=(50000//num_envs),
                                            n_eval_episodes=20, deterministic=True, render=False, verbose=0)
                
                #state_action_coverage_callback = ExplorationCoverageCallback(1000, 6240, 3)
                step_counter_callback=UVFStepCounterCallback(1000)
                #random_step_counter_callback=randomStepsCallback()

                callbacks=[]

                if LOGGING:
                    callbacks+=[WandbCallback(log='all', gradient_save_freq=1000)]

                callbacks+=[eval_tr_callback,eval_0_callback,eval_100_callback]
                #callbacks+=[step_counter_callback]
                #callbacks+=[random_step_counter_callback]

                model.learn(total_timesteps=500_000, progress_bar=True,  log_interval=10, callback=callbacks)

                if LOGGING:
                    run.finish()
                if PROFILING:
                    profiler.disable()
                    stats = pstats.Stats(profiler)
                    stats.dump_stats(f"{path}/profile_{rn}.prof")
