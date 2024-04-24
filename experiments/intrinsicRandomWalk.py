from typing import Callable
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.policies import DQNPolicy
import torch as th 
import torch.functional as F
import numpy as np

from max_sb3.common.type_aliases import GymEnv, VecEnv
from max_sb3.dqn.dqn import DQN
from max_sb3.dqn.upolicies import UncertaintyMlpPolicy
from stable_baselines3.common.utils import polyak_update

from max_sb3.common.uncertainties import RNDUncertaintyStateAction
from max_sb3.common.ubuffers import UncertaintyReplayBuffer


class IntrinsicRandomWalk(DQN):
    
    def __init__(self, policy: str | type[DQNPolicy], env: GymEnv | VecEnv | str, beta: float = 0.5, random_steps:int=5, rnd_config:dict=None, embed_dim:int=512, learning_rate: float | Callable[[float], float] = 0.0001, buffer_size: int = 1000000, learning_starts: int = 50000, batch_size: int = 32, tau: float = 1, gamma: float = 0.99, train_freq: int | th.Tuple[int | str] = 4, gradient_steps: int = 1, replay_buffer_class: type[ReplayBuffer] | None = None, replay_buffer_kwargs: th.Dict[str, th.Any] | None = None, optimize_memory_usage: bool = False, target_update_interval: int = 10000, double_q: bool = False, exploration_fraction: float = 0.1, exploration_initial_eps: float = 1, exploration_final_eps: float = 0.05, max_grad_norm: float = 10, tensorboard_log: str | None = None, policy_kwargs: th.Dict[str, th.Any] | None = None, verbose: int = 0, seed: int | None = None, device: th.device | str = "auto", _init_setup_model: bool = True, use_amp: bool = False):
        
        replay_buffer_class=UncertaintyReplayBuffer
        policy=UncertaintyMlpPolicy
        uncertainty_policy_kwargs = dict(activation_fn = th.nn.ReLU, net_arch=[1024, 1024], learning_rate=0.0001)
        if rnd_config != None:
            uncertainty_policy_kwargs.update(rnd_config)

        uncertainty = RNDUncertaintyStateAction(
                beta, 
                env, 
                embed_dim, 
                buffer_size, 
                uncertainty_policy_kwargs, 
                device=device, 
                flatten_input=True, 
                normalize_images=False)

        # And then add the following replay buffer kwargs: 
        base_replay_buffer_kwargs = {
                        "uncertainty": uncertainty, 
                        "state_action_bonus": True, 
                        "handle_timeout_termination":True, 
                        "uncertainty_of_sampling":True,
                    }
        
        base_replay_buffer_kwargs.update(replay_buffer_kwargs)

        self.embed_dim=embed_dim
        self.random_steps=random_steps
        
        self.beta = beta
        self.num_envs=env.num_envs
        self.episode_steps_taken=[0]*env.num_envs

        self.u_scaler = th.cuda.amp.GradScaler(enabled=self.use_amp)
        super().__init__(policy, env, beta, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq, gradient_steps, replay_buffer_class, replay_buffer_kwargs, optimize_memory_usage, target_update_interval, double_q, exploration_fraction, exploration_initial_eps, exploration_final_eps, max_grad_norm, tensorboard_log, policy_kwargs, verbose, seed, device, _init_setup_model, use_amp)

    def _create_aliases(self) -> None:
        super()._create_aliases()
        self.u_net = self.policy.u_net
        self.u_net_target = self.policy.u_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        super()._on_step()
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(self.u_net.parameters(), self.u_net_target.parameters(), self.tau)

        self.logger.record("rollout/beta", self.beta)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        u_losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.autocast(device_type=self.device.type, dtype=th.float16, enabled=self.use_amp):
                with th.no_grad():
                    # Compute the next Q-values using the target network
                    next_q_values = self.q_net_target(replay_data.next_observations)
                    if self.double_q:
                        # Compute the next Q-values using the current network
                        next_q_values_current = self.q_net(replay_data.next_observations)
                        next_u_values_current = self.u_net(replay_data.next_observations)
                        # Determine argmax based on the current network values
                        actions = (next_q_values_current + self.beta * next_u_values_current).max(dim=1)[1].unsqueeze(dim=1)
                        next_q_values = next_q_values.gather(dim=1, index=actions)
                    else:
                        next_u_values = self.u_net_target(replay_data.next_observations)
                        actions = (next_q_values + self.beta * next_u_values).max(dim=1)[1].unsqueeze(dim=1)
                        next_q_values = next_q_values.gather(dim=1, index=actions)

                    # 1-step TD target
                    target_q_values = replay_data.rewards[0] + (1 - replay_data.dones) * self.gamma * next_q_values

            with th.autocast(device_type=self.device.type, dtype=th.float16, enabled=self.use_amp):
                # Get current Q-values estimates
                current_q_values = self.q_net(replay_data.observations)

                # Retrieve the q-values for the actions from the replay buffer
                current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

                # Compute Huber loss (less sensitipolicy: str | type[DQNPolicy]ve to outliers)
                loss = F.smooth_l1_loss(current_q_values, target_q_values)

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.policy.optimizer)
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.scaler.step(self.policy.optimizer)
            self.scaler.update()

            with th.autocast(device_type=self.device.type, dtype=th.float16, enabled=self.use_amp):
                with th.no_grad():
                    # Compute the next uncertainties using the target network
                    next_u_values = self.u_net_target(replay_data.next_observations)
                    if self.double_q:
                        next_u_values = next_u_values.gather(dim=1, index=actions)
                    else:
                        next_u_values = next_u_values.gather(dim=1, index=actions)
                    # 1-step TD target
                    target_u_values = replay_data.rewards[1] + (1 - replay_data.dones) * self.gamma * next_u_values

            with th.autocast(device_type=self.device.type, dtype=th.float16, enabled=self.use_amp):
                # Get current uncertainty estimates
                current_u_values = self.u_net(replay_data.observations)

                # Retrieve the uncertainties for the actions from the replay buffer
                current_u_values = th.gather(current_u_values, dim=1, index=replay_data.actions.long())

                # Compute Huber loss (less sensitive to outliers)
                u_loss = F.smooth_l1_loss(current_u_values, target_u_values)

            # Optimize the policy
            self.policy.u_optimizer.zero_grad()
            self.u_scaler.scale(u_loss).backward()
            self.u_scaler.unscale_(self.policy.u_optimizer)
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.u_net.parameters(), self.max_grad_norm)
            self.u_scaler.step(self.policy.u_optimizer)
            self.u_scaler.update()

            losses.append(loss.item())
            u_losses.append(u_loss.item())

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/u_loss", np.mean(u_losses))