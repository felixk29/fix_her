from copy import deepcopy
from typing import Callable
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env import DummyVecEnv

class AdaptedVecEnv(DummyVecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``Cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    :raises ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    get_state: Callable[[], np.ndarray] = None
    tp_chance: float = -1

    def set_get_state(self, get_state: Callable[[], np.ndarray]):
        self.get_state = get_state

    #states could be a function that returns a state from a stack, that will be filled up when necessary?
    def step_wait(self) -> VecEnvStepReturn:
        # Avoid circular imports
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = terminated or truncated
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                ## TODO RND IMPLEMENTATION
                if np.random.uniform() < self.tp_chance:
                    tp_target=self.get_state()
                    obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(options={"load_state": tp_target})
                else:
                    obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
            
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))


    # TODO reset 
    #states could be a function that returns a state from a stack, that will be filled up when necessary?
    def reset(self) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            maybe_options = {"options": self._options[env_idx]} if self._options[env_idx] else {}
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=self._seeds[env_idx], **maybe_options)
            self._save_obs(env_idx, obs)
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()
