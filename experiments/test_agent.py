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

#### Paramampa
### EXPERIMT SETUP BUGGY; RUN THEM SOLO
num_envs=10
seed=0

np.random.seed(seed)


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
        return self.linear(self.cnn(observations))

cf={'policy': 'CnnPolicy',
        'buffer_size': 50000,
        'batch_size': 512,
        'gamma': 0.99,
        'learning_starts': 512,
        'max_grad_norm': 1.0,
        'gradient_steps': 1,
        'train_freq': (10, 'step'),
        'target_update_interval': 10,
        'tau': 0.01,
        'exploration_fraction': 1.0,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.01,
        'learning_rate': 1e-3,
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
        }
}


wandb.tensorboard.patch(root_logdir="./experiments/logs/")

## TP Version ##
for vl in [(0.0,0.0),(0.1,0.1),(0.2,0.2)]:
    for rn in range(5,10):

        print("\n----------------------------")
        print(f"Starting run {rn} with tp chance {vl}")
        print("----------------------------\n")

        run=wandb.init(
            project="thesis",
            name="tpdqn",
            monitor_gym=True,
            sync_tensorboard=True,
        )


        train_env_tp = AdaptedVecEnv([make_env_fn(train_config, seed=seed, rank=i) for i in range(num_envs)])
        tr_eval_env_tp = DummyVecEnv([make_env_fn(train_config, seed=seed, rank=i) for i in range(num_envs)])
        test_0_env_tp = DummyVecEnv([make_env_fn(test_0_config, seed=seed, rank=i) for i in range(num_envs)])
        test_100_env_tp = DummyVecEnv([make_env_fn(test_100_config, seed=seed, rank=i) for i in range(num_envs)])

        tp_model = tpDQN(cf['policy'], train_env_tp, buffer_size=cf['buffer_size'], batch_size=cf['batch_size'], gamma=cf['gamma'], 
                                learning_starts=cf['learning_starts'], gradient_steps=cf['gradient_steps'], train_freq=cf['train_freq'],
                                    target_update_interval=cf['target_update_interval'], tau=cf['tau'], exploration_fraction=cf['exploration_fraction'],
                                    exploration_initial_eps=cf['exploration_initial_eps'], exploration_final_eps=cf['exploration_final_eps'],
                                    max_grad_norm=cf['max_grad_norm'], learning_rate=cf['learning_rate'], verbose=cf['verbose'],
                                    tensorboard_log=f"runs/{run.id}/", policy_kwargs=cf['policy_config'] ,device=cf['device'],
                                    tp_chance_start=vl[0],tp_chance_end=vl[1])

        eval_tr_callback = EvalCallback(tr_eval_env_tp, log_path=base_log+f"log_b50_tp_{round(vl[0]*100)}_{round(vl[1]*100)}/tr/{rn}/", eval_freq=max(25000 // num_envs, 1),
                                    n_eval_episodes=100, deterministic=True, render=False, verbose=0)

        eval_0_callback = EvalCallback(test_0_env_tp, log_path=base_log+f"log_b50_tp_{round(vl[0]*100)}_{round(vl[1]*100)}/0/{rn}/", eval_freq=max(25000 // num_envs, 1),
                                    n_eval_episodes=100, deterministic=True, render=False, verbose=0)

        eval_100_callback = EvalCallback(test_100_env_tp, log_path=base_log+f"log_b50_tp_{round(vl[0]*100)}_{round(vl[1]*100)}/100/{rn}/", eval_freq=max(25000 // num_envs, 1),
                                    n_eval_episodes=100, deterministic=True, render=False, verbose=0)

        tp_wandb_callback=WandbCallback(log='all', gradient_save_freq=1000)

        tp_model.learn(total_timesteps=500000, progress_bar=True,  log_interval=10, callback=[eval_tr_callback,eval_0_callback,eval_100_callback, tp_wandb_callback])
        print(tp_model.refilled)

        wandb.finish()


## Baseline ##

# run=wandb.init(
#     project="thesis",
#     name="baseline",
#     monitor_gym=True,
#     sync_tensorboard=True,
# )


# train_env = DummyVecEnv([make_env_fn(train_config, seed=seed, rank=i) for i in range(num_envs)])

# tr_eval_env = DummyVecEnv([make_env_fn(train_config, seed=seed, rank=i) for i in range(num_envs)])
# test_0_env = DummyVecEnv([make_env_fn(test_0_config, seed=seed, rank=i) for i in range(num_envs)])
# test_100_env = DummyVecEnv([make_env_fn(test_100_config, seed=seed, rank=i) for i in range(num_envs)])


# baseline_model = DoubleDQN(cf['policy'], train_env, buffer_size=cf['buffer_size'], batch_size=cf['batch_size'], gamma=cf['gamma'], 
#                            learning_starts=cf['learning_starts'], gradient_steps=cf['gradient_steps'], train_freq=cf['train_freq'],
#                             target_update_interval=cf['target_update_interval'], tau=cf['tau'], exploration_fraction=cf['exploration_fraction'],
#                             exploration_initial_eps=cf['exploration_initial_eps'], exploration_final_eps=cf['exploration_final_eps'],
#                             max_grad_norm=cf['max_grad_norm'], learning_rate=cf['learning_rate'], verbose=cf['verbose'],
#                             tensorboard_log=f"runs/{run.id}/", policy_kwargs=cf['policy_config'] ,device=cf['device'])

# eval_tr_callback = EvalCallback(tr_eval_env, log_path=base_log+"log_baseline_b50/tr/", eval_freq=max(25000 // num_envs, 1),
#                               n_eval_episodes=100, deterministic=True, render=False, verbose=0)

# eval_0_callback = EvalCallback(test_0_env, log_path=base_log+"log_baseline_b50/0/", eval_freq=max(25000 // num_envs, 1),
#                               n_eval_episodes=100, deterministic=True, render=False, verbose=0)

# eval_100_callback = EvalCallback(test_100_env, log_path=base_log+"log_baseline_b50/100/", eval_freq=max(25000 // num_envs, 1),
#                               n_eval_episodes=50, deterministic=True, render=False, verbose=0)

# baseline_wandb_callback=WandbCallback(log='all', gradient_save_freq=1000)

# baseline_model.learn(total_timesteps=500000, progress_bar=True, log_interval=10, callback=[eval_tr_callback,eval_0_callback,eval_100_callback,baseline_wandb_callback])



### TODO: FIND BUG THAT MAKES ONLY THE FIRST RUN NICE


