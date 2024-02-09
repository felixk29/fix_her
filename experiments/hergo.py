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
from four_room.wrappers import gym_wrapper

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
class UVFWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space= spaces.Dict(spaces={'observation':env.observation_space, 'achieved_goal':env.observation_space, 'desired_goal':env.observation_space})
        return 

    def observation(self, observation):
        return {'observation':observation, 'achieved_goal':observation, 'desired_goal':np.zeros_like(observation)}
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        if np.array_equal(achieved_goal, desired_goal):
            return np.array([1.0])
        else:
            return np.array([.0])


class MultiInput_CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512, device=th.device("cuda")):
        super(MultiInput_CNN, self).__init__(observation_space, features_dim)
        self.device=device
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space['observation'].shape[0]*2
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad(): ##TODO needs working to actually get correct number, worst case hard code??
            
            n_flatten = self.cnn(torch.ones(1,n_input_channels ,*observation_space['observation'].shape[1:])).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: spaces.Dict) -> torch.Tensor:
        if len(observations['desired_goal'].shape)<4:
            obs=torch.cat([observations['observation'], observations['desired_goal']], axis=0).to(self.device)
            obs=obs.unsqueeze(0)
        else:
            obs=torch.cat([observations['observation'], observations['desired_goal']], axis=1).to(self.device)
        return self.linear(self.cnn(obs))

SelfHERGO = TypeVar("SelfHERGO", bound="HERGO")

class HERGO(DoubleDQN):
    uvf: DoubleDQN =None

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
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
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
        tp_chance:float=.0,
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

        ### Felix part: 
             
        self.observation_space_size= th.prod(th.tensor(self.env.observation_space.shape))
        self.rnd= RND(self.env.observation_space.shape, device=self.device)
        
        self.tp_chance=tp_chance
        self.num_envs= self.env.num_envs
        self.interim_goal= [None]*self.num_envs
        self.does_interim_goal= [False]*self.num_envs
        self.last_uvf_value= [0]*self.num_envs

        uvf_kwargs= deepcopy(policy_kwargs)
        for key in uvf_config['kwargs']:
            uvf_kwargs[key]=uvf_config['kwargs'][key]

        uvf_logger= configure()

        self.uvf= DoubleDQN(
            "MultiInputPolicy",
            UVFWrapper(self.env),
            self.lr_schedule,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                "n_sampled_goal": 4,
                "goal_selection_strategy": "future",
            },
            policy_kwargs=uvf_kwargs,)
        self.uvf.set_logger(uvf_logger)

    uvf_timesteps=0
    #TODO UVF still has to be included
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        super().train(gradient_steps, batch_size)

        #rnd trains on same batch as main policy
        observations=self.last_batch.observations
        obs=observations.view(batch_size, -1)
        obs=obs.type(th.FloatTensor)
        obs=obs.to(self.device)
        self.rnd.train(obs)   

        #TODO UVF 
        if self.uvf_timesteps > batch_size:
            self.uvf.train(gradient_steps, batch_size)

    def _to_dict(self, observation, desired_goal=None):
        if isinstance(desired_goal,np.ndarray) or isinstance(desired_goal,th.Tensor):
            des=th.tensor(desired_goal)
        else:
            des=th.tensor(np.zeros_like(observation))
        return {'observation':torch.tensor(observation), 'achieved_goal':torch.tensor(observation), 'desired_goal':des}


    # list of goals if interim goal is active, else None
    interim_goal= []
    # list of bools, if interim goal is active
    does_interim_goal= []
    # list of last uvf values, to be used in collect_rollouts
    last_uvf_value=[]

    #TODO will be the bulk of the new logic, is getting called by learn()
    # generally yoinked from off_policy_algorithm.py
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into both the main ``ReplayBuffer`` as well as the UVF ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        self.uvf.policy.set_training_mode(False)

        num_collected_steps = 0
        num_collected_episodes = 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."
        
        #TODO fill last_uvf_value with current uvf values
        for idx in range(env.num_envs):
            self.last_uvf_value[idx]= -10

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):


            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.uvf_timesteps += sum(self.does_interim_goal)
            self.num_timesteps += env.num_envs
            num_collected_steps += 1            

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # TODO split into two buffers, one for main policy and one for uvf
            for idx in range(env.num_envs):
                if self.does_interim_goal[idx]:
                    #TODO check if goal is reached, very unlikely, unsure if valid
                    if np.array_equal(new_obs[idx],self.interim_goal[idx]):
                        idx_done=True
                        idx_reward=1
                        self.does_interim_goal[idx]= False
                        self.interim_goal[idx]= None
                    else:
                        idx_done=dones[idx]
                        idx_reward=0

                    # TODO: check if value function value increased since last step to figure out if goal is reached, 
                    # if so, set does_interim_goal to false etc. 
                    current_uvf_value=torch.max(self.uvf.q_net(self._to_dict(self._last_obs[idx], self.interim_goal[idx])).detach())
                    if current_uvf_value > self.last_uvf_value[idx]:
                        self.last_uvf_value[idx]= current_uvf_value
                    else: 
                        self.does_interim_goal[idx]= False
                        self.interim_goal[idx]= None
                        self.last_uvf_value[idx]= -10
                        idx_done=True

                    # if problems change _last_obs to new_obs with space.dict

                    #circumventing self._store_transition as we split up where we store based on phase of env (interim goal or not)
                    # TODO possibly fucked, look at _stor_transition, how it handles dict 
                        
                    next_obs = deepcopy(new_obs[idx])

                    if dones[idx] and infos[idx].get("terminal_observation") is not None:
                        next_obs=infos[idx]["terminal_observation"]

                    new_dict=self._to_dict(next_obs,self.interim_goal[idx])
                    old_dict=self._to_dict(self._last_obs[idx],self.interim_goal[idx])

                    #self._store_transition(self.uvf.replay_buffer, buffer_actions[idx], idx_reward, spaceDict, idx_done, infos[idx])  # type: ignore[arg-type]
                    self.uvf.replay_buffer.add(old_dict, new_dict, np.array([buffer_actions[idx]]), np.array([idx_reward]), np.array([idx_done]), [infos[idx]])  # type: ignore[arg-type]
                else:

                    self._last_original_obs,nex_ob,rew_=self._last_obs,new_obs[idx], rewards[idx]
                    if dones[idx] and infos[idx].get("terminal_observation") is not None:
                        nex_ob=infos[idx]["terminal_observation"]


                    self.replay_buffer.add(np.array([self._last_original_obs[idx]]), np.array([nex_ob]), np.array([buffer_actions[idx]]), np.array([rew_]), np.array([dones[idx]]), [infos[idx]])  # type: ignore[arg-type]
                    #self._store_transition(replay_buffer, buffer_actions[idx], rewards[idx], new_obs[idx], dones[idx], infos[idx])  # type: ignore[arg-type]

            self._last_obs = deepcopy(new_obs)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    
                    if self.learning_starts < self.num_timesteps:

                        if self.does_interim_goal[idx]:
                            self.does_interim_goal[idx]= False
                            self.interim_goal[idx]= None
                            self.last_uvf_value[idx]= -10


                        if not self.does_interim_goal[idx] and np.random.rand()<self.tp_chance:
                            self.does_interim_goal[idx]= True
                            self.interim_goal[idx]= self.get_interim_goal()

                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
                
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    #stack of goals to not sample too often, gets refilled when empty
    goal_stack=[]
    def get_interim_goal(self):
        if len(self.goal_stack) == 0:
                replay_data = self.replay_buffer.sample(sum(self.does_interim_goal) * 3)  # type: ignore[union-attr]
                                                                                        #self._vec_normalize_env originally
                observations=replay_data.observations

                obs=observations.view(sum(self.does_interim_goal) * 3, -1)
                obs=obs.type(th.FloatTensor)
                obs=obs.to(self.device)
                pred, target = self.rnd(obs)

                #self.rnd.train(obs)

                rnd_loss = F.l1_loss(pred, target, reduction='none')
                _, indices = th.sort(rnd_loss, descending=True, axis=0)

                indices = indices[:sum(self.does_interim_goal)]
                self.goal_stack = list(observations[indices].unbind())

        return self.goal_stack.pop().reshape(self.env.observation_space.shape).cpu().numpy()


    #TODO needs to be overhauled to work with 2 different policies, where one requires explicit goal
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"

            unscaled_action=np.array([None]*n_envs)
            for idx, last_obs in enumerate(self._last_obs):
                if np.random.rand()<self.exploration_rate:
                    #TODO check why tf the predict is krangled when determenistic=False
                    unscaled_action[idx]= self.action_space.sample()
                else:
                    if self.does_interim_goal[idx]:
                        unscaled_action[idx], _= self.uvf.predict(self._to_dict(last_obs, self.interim_goal[idx]), deterministic=True)
                    else:
                        unscaled_action[idx], _= self.policy.predict(th.Tensor(last_obs).unsqueeze(axis=0), deterministic=True)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action



if __name__ =="__main__":

    import torch
    from torch import nn
    from stable_baselines3.common.vec_env import DummyVecEnv
    import dill
    from four_room.old_env import FourRoomsEnv
    from four_room.wrappers import gym_wrapper

    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import gymnasium as gym

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

    class Baseline_CNN(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
            super(Baseline_CNN, self).__init__(observation_space, features_dim)
            n_input_channels = observation_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
                nn.ReLU(),nn.Flatten(),)
            with torch.no_grad():
                n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            if len(observations.shape) == 3:
                observations = observations.unsqueeze(0)
            return self.linear(self.cnn(observations))




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

    def make_env_fn_uvf(config, seed: int= 0, rank: int = 0):
        def _init():
            env = UVFWrapper(gym_wrapper(gym.make('MiniGrid-FourRooms-v1', 
                        agent_pos=config['agent positions'], 
                        goal_pos=config['goal positions'], 
                        doors_pos=config['topologies'], 
                        agent_dir=config['agent directions'])))
            if seed==0:
                env.reset()
            else:
                env.reset(seed=seed+rank)
            return Monitor(env)
        return _init


    with open('./experiments/four_room/configs/fourrooms_train_config.pl', 'rb') as file:
        train_config = dill.load(file)

    train_env= DummyVecEnv([make_env_fn(train_config, seed=0, rank=0) for _ in range(1)])
    train_uvf_env= DummyVecEnv([make_env_fn_uvf(train_config, seed=0, rank=0) for _ in range(1)])

    cf={'policy': 'CnnPolicy',
        'buffer_size': 50000,
        'batch_size': 256,
        'gamma': 0.99,
        'learning_starts': 256,
        'max_grad_norm': 1.0,
        'gradient_steps': 1,
        'train_freq': (10, 'step'),
        'target_update_interval': 10,
        'tau': 0.01,
        'exploration_fraction': 1.0,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.01,
        'learning_rate': 1e-4,
        'verbose': 1,
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
        'tp_chance':1.,
    }

    model = HERGO(cf['policy'], train_env, buffer_size=cf['buffer_size'], batch_size=cf['batch_size'], gamma=cf['gamma'], 
                            learning_starts=cf['learning_starts'], gradient_steps=cf['gradient_steps'], train_freq=cf['train_freq'],
                                target_update_interval=cf['target_update_interval'], tau=cf['tau'], exploration_fraction=cf['exploration_fraction'],
                                exploration_initial_eps=cf['exploration_initial_eps'], exploration_final_eps=cf['exploration_final_eps'],
                                max_grad_norm=cf['max_grad_norm'], learning_rate=cf['learning_rate'], verbose=cf['verbose'],
                                policy_kwargs=cf['policy_config'] ,device=cf['device'], tp_chance=cf['tp_chance'],uvf_config=cf['uvf_config'])
    
    model.learn(500000)