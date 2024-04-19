from stable_baselines3 import HerReplayBuffer
import numpy as np
from typing import Optional
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import DictReplayBufferSamples
import copy


class FixedHerBuffer(HerReplayBuffer):
    def _get_virtual_samples(
    self,
    batch_indices: np.ndarray,
    env_indices: np.ndarray,
    env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        """
        Get the samples, sample new desired goals and compute new rewards.

        :param batch_indices: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :param env: associated gym VecEnv to normalize the observations/rewards when sampling, defaults to None
        :return: Samples, with new desired goals and new rewards
        """
        # Get infos and obs
        obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.observations.items()}
        next_obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.next_observations.items()}
        if self.copy_info_dict:
            # The copy may cause a slow down
            infos = copy.deepcopy(self.infos[batch_indices, env_indices])
        else:
            infos = [{} for _ in range(len(batch_indices))]
        # Sample and set new goals
        new_goals = self._sample_goals(batch_indices, env_indices)
        obs["desired_goal"] = new_goals
        # The desired goal for the next observation must be the same as the previous one
        next_obs["desired_goal"] = new_goals

        assert (
            self.env is not None
        ), "You must initialize HerReplayBuffer with a VecEnv so it can compute rewards for virtual transitions"
        # Compute new reward
        rewards = self.env.env_method(
            "compute_reward",
            # the new state depends on the previous state and action
            # s_{t+1} = f(s_t, a_t)
            # so the next achieved_goal depends also on the previous state and action
            # because we are in a GoalEnv:
            # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
            # therefore we have to use next_obs["achieved_goal"] and not obs["achieved_goal"]
            next_obs["achieved_goal"],
            # here we use the new desired goal
            obs["desired_goal"],
            infos,
            # we use the method of the first environment assuming that all environments are identical.
            indices=[0],
        )
        rewards = rewards[0].astype(np.float32)  # env_method returns a list containing one element
        obs = self._normalize_obs(obs, env)  # type: ignore[assignment]
        next_obs = self._normalize_obs(next_obs, env)  # type: ignore[assignment]

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}

        rew=self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env))

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            
            dones=rew>0.2,
            rewards=rew,  # type: ignore[attr-defined]
        )