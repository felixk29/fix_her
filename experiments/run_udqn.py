from max_sb3.common.uncertainties import RNDUncertaintyStateAction
from max_sb3.dqn.udqn import UncertaintyDQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from max_sb3.dqn.upolicies import UncertaintyMlpPolicy

from max_sb3.common.ubuffers import UncertaintyReplayBuffer
import torch
import dill
from four_room.wrappers import gym_wrapper
from src.networks import CNN

import gymnasium as gym
from four_room.env import FourRoomsEnv

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open('four_room/configs/fourrooms_train_config.pl', 'rb') as file:
    train_config = dill.load(file)
with open('four_room/configs/fourrooms_test_0_config.pl', 'rb') as file:
    test_config = dill.load(file)

num_train_configs = len(train_config['topologies'])
num_test_configs = len(test_config['topologies'])

exp_frac = 1.0
beta = 300
buffer_size = 500_000
batch_size = 256
tau = 0.01
gamma = .99
max_grad_norm = 1
gradient_steps = 1
target_update_interval = 10
train_freq = 10
exploration_final_eps = 0.1
learning_rate = 0.0001
n_envs = 10

for i in range(1):
    w_decay = 0

    eval_env = make_vec_env('MiniGrid-FourRooms-v1', 
                            n_envs=1, 
                            seed=0, 
                            vec_env_cls=DummyVecEnv, 
                            wrapper_class=gym_wrapper, 
                            env_kwargs={'agent_pos': test_config['agent positions'],
                                        'goal_pos': test_config['goal positions'],
                                        'doors_pos': test_config['topologies'],
                                        'agent_dir': test_config['agent directions']})

    train_env = make_vec_env('MiniGrid-FourRooms-v1', 
                            n_envs=n_envs, 
                            seed=0, 
                            vec_env_cls=DummyVecEnv, 
                            wrapper_class=gym_wrapper, 
                            env_kwargs={'agent_pos': train_config['agent positions'],
                                        'goal_pos': train_config['goal positions'],
                                        'doors_pos': train_config['topologies'],
                                        'agent_dir': train_config['agent directions']})


    uncertainty_policy_kwargs = dict(activation_fn = torch.nn.ReLU, net_arch=[1024, 1024], learning_rate=0.0001)
    embed_dim = 512
    uncertainty = RNDUncertaintyStateAction(
            beta, 
            train_env, 
            embed_dim, 
            buffer_size, 
            uncertainty_policy_kwargs, 
            device="cuda", 
            flatten_input=True, 
            normalize_images=False)



    policy_kwargs = dict(features_extractor_class = CNN, features_extractor_kwargs = {'features_dim': 512}, normalize_images=False, net_arch=[], optimizer_kwargs={'weight_decay':w_decay}, beta=beta)

    replay_buffer_kwargs = {
        "uncertainty": uncertainty, 
        "state_action_bonus": True, 
        "handle_timeout_termination":True, 
        "uncertainty_of_sampling":True,
    }

    model = UncertaintyDQN(
                    UncertaintyMlpPolicy,
                    train_env, 
                    beta,
                    double_q=True, 
                    learning_starts=batch_size,
                    tensorboard_log="logging/", 
                    policy_kwargs=policy_kwargs, 
                    learning_rate=learning_rate, 
                    buffer_size=buffer_size, 
                    replay_buffer_class=UncertaintyReplayBuffer,
                    replay_buffer_kwargs=replay_buffer_kwargs,
                    batch_size=batch_size, 
                    tau=tau, gamma=gamma, 
                    train_freq=(train_freq // n_envs, "step"), 
                    gradient_steps=gradient_steps, 
                    max_grad_norm=max_grad_norm, 
                    target_update_interval=target_update_interval,
                    exploration_final_eps=exploration_final_eps,
                    exploration_fraction=exp_frac,
                    seed=0,
                    device='cuda',
                    )

    model.learn(total_timesteps=500_000)
    train_env.close()
    eval_env.close()
