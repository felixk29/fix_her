import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.type_aliases import MaybeCallback

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.dqn.policies import DQNPolicy
from doubledqn import DoubleDQN
from RND import RND

SelftpDDQN = TypeVar("SelftpDDQN", bound="tpDQN")


class tpDQN(DoubleDQN):
    
    _total_timesteps_tp: int=0

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
        tp_chance_start:float=.0,
        tp_chance_end:float=.0,
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

        ### TODO RND 
        self.observation_space_size= th.prod(th.tensor(self.env.observation_space.shape))
        self.rnd= RND(self.env.observation_space.shape, device=self.device)

        self.env.set_get_state(self._inject_buffer_into_env)
        self.env.tp_chance=self._inject_tp_chance
        self.env.is_buffer_ready=self._inject_is_buffer_ready
        self.max_tp_chance=tp_chance_end
        self.start_tp_chance=tp_chance_start

        self.refilled=0


    def _inject_tp_chance(self):
        number = self.max_tp_chance - (self.max_tp_chance - self.start_tp_chance) * self.num_timesteps / self._total_timesteps_tp
        return number

    def _inject_is_buffer_ready(self):
            
        return self.num_timesteps > self.learning_starts


    #just using a stack, if threading problems could upgrade to a queue, this is just simpler
    state_stack=[]
    state_stack_size=20

    def _inject_buffer_into_env(self):
        if len(self.state_stack) == 0:
            self.refilled+=1
            replay_data = self.replay_buffer.sample(self.state_stack_size * 3, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # this is a named tuple, delete every entry that has or truncated = True
            #TODO maybe add a check that its not done before it arrived?? but idk if nessary

            observations=replay_data.observations


            obs=observations.view(self.state_stack_size * 3, -1)
            obs=obs.type(th.FloatTensor)
            obs=obs.to(self.device)
            pred, target = self.rnd(obs)

            #self.rnd.train(obs)

            rnd_loss = F.l1_loss(pred, target, reduction='none')
            _, indices = th.sort(rnd_loss, descending=True, axis=0)

            indices = indices[:self.state_stack_size]
            self.state_stack = list(observations[indices].unbind())
            
        return self.state_stack.pop().reshape(self.env.observation_space.shape).cpu().numpy()

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        super().train(gradient_steps, batch_size)
        observations=self.last_batch.observations
        obs=observations.view(batch_size, -1)
        obs=obs.type(th.FloatTensor)
        obs=obs.to(self.device)
        #only train if tp is actually used anytime
        if self.start_tp_chance != 0. and self.max_tp_chance != 0.:
            self.rnd.train(obs)        


    def learn(
    self: SelftpDDQN,
    total_timesteps: int,
    callback: MaybeCallback = None,
    log_interval: int = 4,
    tb_log_name: str = "tpDDQN",
    reset_num_timesteps: bool = True,
    progress_bar: bool = False,
    ) -> SelftpDDQN:
        self._total_timesteps_tp = total_timesteps
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )