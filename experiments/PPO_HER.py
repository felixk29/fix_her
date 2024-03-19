from gymnasium import Space
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
import torch as th
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.buffers import DictRolloutBufferSamples
from stable_baselines3 import PPO

# right now just assuming future goal selection strategy
# also code heavily yoinked from HerReplayBuffer

import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

class HerRolloutBuffer(DictRolloutBuffer):
    def __init__(self, buffer_size: int,
                 observation_space: Dict,
                 action_space: Space,
                 device: th.device | str = "auto",
                 gae_lambda: float = 1,
                 gamma: float = 0.99,
                 n_envs: int = 1,
                 # Felix part
                 her_ratio: float = 0.25,
                 ):
        
        self.her_ratio = her_ratio
        self.her_entries = int(buffer_size * her_ratio)


        super().__init__(buffer_size+self.her_entries, observation_space,
                         action_space, device, gae_lambda, gamma, n_envs)

        self.old_achieved = np.zeros((self.buffer_size, n_envs ,*self.observation_space['achieved_goal'].shape), dtype=np.float32)
        self.reset_herify()


    def reset_herify(self):
        self.to_herify = np.zeros(self.buffer_size)
        temp=np.zeros((self.buffer_size//2))
        temp[:self.her_entries] = 1
        self.to_herify[::2] = np.random.permutation(temp)
        
    def add(  # type: ignore[override]
    self,
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
    reward: np.ndarray,
    episode_start: np.ndarray,
    value: th.Tensor,
    log_prob: th.Tensor,
    ) -> None:

        #self.old_achieved[self.pos]=obs["achieved_goal"]
        ## HER PART selects which to use for HER

        if self.to_herify[self.pos]==1:
            her_obs = obs.copy()
            her_rewards = reward.copy()

            for i in range(self.n_envs):
                if not episode_start[i]:
                    starts=self.episode_starts[:,i].nonzero()[0]
                    if len(starts)>0:
                        last_ep_start=starts[-1]
                        if last_ep_start<self.pos-2:

                            her_obs_ind=np.random.randint(last_ep_start,self.pos-1)
                            her_goal_ind=np.random.randint(her_obs_ind ,self.pos)
                            her_obs['observation'][i] = self.observations['observation'][her_obs_ind,i].copy()
                            her_obs["desired_goal"][i] = self.observations["achieved_goal"][her_goal_ind,i]
                            her_rewards[i] = 1 if np.array_equal(her_obs["observation"][i], her_obs["desired_goal"][i]) else 0
                            her_rewards[i] = 1 if np.array_equal(her_obs["observation"][i], her_obs["desired_goal"][i]) else 0

            her_obs['achieved_goal'] = her_obs['observation']
            super().add(her_obs, action, her_rewards, episode_start, value, log_prob)        

        super().add(obs, action, reward, episode_start, value, log_prob)
                

    def alt_get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None
                     ) -> DictRolloutBufferSamples:

        #might have to move goal generaion to add, so that it is done before the buffer is full and everything is calculated, 
        #circumventing haveing to redo get, what they do is ugly anyways
        if len(self.old_achieved.shape)!=4:
            self.old_achieved = self.swap_and_flatten(self.old_achieved)
            self.episode_starts = self.swap_and_flatten(self.episode_starts)


        ret = super()._get_samples(batch_inds, env)
        to_herify = int(len(batch_inds) * self.her_ratio)
        

        # for pos,idx in enumerate(batch_inds[:to_herify]):
        #     if idx >= self.n_envs* self.buffer_size -1:
        #         continue
        #     elif self.episode_starts[idx+1]==1:
        #         continue
        #     print("idx",idx)
        #     ret.observations["desired_goal"][pos] = self.to_torch(self.achieved_goals[idx+1])
        temp=ret.observations["desired_goal"][:to_herify]
        increased=[min(self.buffer_size* self.n_envs -1 , idx+1) for idx in batch_inds[:to_herify]]
        ret.observations["desired_goal"][:to_herify] = self.to_torch(self.old_achieved[increased])
        for i,idx in enumerate(batch_inds[:to_herify]):
            if idx >= self.n_envs* self.buffer_size -1:
                continue
            elif self.episode_starts[idx+1]==1:
                ret.observations["desired_goal"][i] = temp[i]

        return ret

    def reset(self) -> None:
        self.old_achieved = np.zeros((self.buffer_size, self.n_envs ,*self.observation_space['achieved_goal'].shape), dtype=np.float32)
        self.reset_herify()

        super().reset()
    

class HER_PPO(PPO):
    def __init__(self,*args, her_val:float=0.2, **kwargs):
        self.her_val=her_val

        super().__init__(*args, **kwargs)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()


        while n_steps < int(n_rollout_steps*(1-self.her_val)):
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
        
            #naive implementation of HER

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )

            self._last_obs  = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        
        # HER PART
        # repeat same trajectories as from 0 to n_rollout_steps*(self.her_val) 
        # but with different goals
        # and different rewards
            
        #observations
        #actions
        #rewards
        #episode_starts
        #values
        #log_probs
        rest=n_rollout_steps-n_steps

        all_obs=rollout_buffer.observations

        rollout_buffer.observations['observation'][n_steps:] = all_obs['observation'][:rest]
        rollout_buffer.observations['achieved_goal'][n_steps:] = all_obs['achieved_goal'][:rest]
        rollout_buffer.observations['desired_goal'][n_steps:] = all_obs['desired_goal'][:rest]



        rollout_buffer.actions[n_steps:] = rollout_buffer.actions[:rest]
        rollout_buffer.rewards[n_steps:] = rollout_buffer.rewards[:rest]
        rollout_buffer.values[n_steps:] = rollout_buffer.values[:rest]
        rollout_buffer.log_probs[n_steps:] = rollout_buffer.log_probs[:rest]
        rollout_buffer.episode_starts[n_steps:] = rollout_buffer.episode_starts[:rest]


        #for each env iterate over all episodes and select endgoal of episode 27th odoo, 11:30 aM 
        for idx in range(env.num_envs):
            end_goal=rollout_buffer.observations["achieved_goal"][-1,idx]
            alt=0
            for i in reversed(range(n_steps, n_rollout_steps)):
                # if i<n_rollout_steps-1 and rollout_buffer.episode_starts[i+1,idx]:
                if rollout_buffer.episode_starts[i,idx] and i//2==alt:
                    alt=(alt+1)//2
                    end_goal=rollout_buffer.observations["achieved_goal"][i,idx]
                    rollout_buffer.rewards[i,idx]=1
                    terminal_obs={"observation":th.Tensor(end_goal), "achieved_goal":th.Tensor(end_goal), "desired_goal":th.Tensor(end_goal)}
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rollout_buffer.rewards[i,idx] += self.gamma * terminal_value
                elif i==n_rollout_steps or i//2==alt:
                    end_goal=rollout_buffer.observations["achieved_goal"][i,idx]
                    rollout_buffer.rewards[i,idx]=1
                    terminal_obs={"observation":th.Tensor(end_goal), "achieved_goal":th.Tensor(end_goal), "desired_goal":th.Tensor(end_goal)}
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rollout_buffer.rewards[i,idx] += self.gamma * terminal_value
                else:
                    rollout_buffer.observations["desired_goal"][i,idx]=end_goal
                    rollout_buffer.rewards[i,idx]=0
                    rollout_buffer.episode_starts[i,idx]=1
        rollout_buffer.pos=n_rollout_steps
        rollout_buffer.full=True

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
    