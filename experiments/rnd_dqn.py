import torch as th
import numpy as np 
from stable_baselines3 import DQN
from RND import RND
from typing import Union, Callable
from torch.nn import functional as F

# RND-DQN
# DoubleDQN that extends DQN and uses RND as intrinsic reward instead of epsilon-greedy

class RND_DQN(DQN):
    def __init__(self,**kwargs):
        self.beta_arg =  kwargs.pop('beta', None)  
        super(RND_DQN, self).__init__(**kwargs)
        self.rnd = RND(self.observation_space.shape, device=self.device)

    def beta(self):
        total_timesteps=500000
        if self.beta_arg is None:
            return 0.1

        if isinstance(self.beta_arg, float):
            return self.beta_arg
        
        if len(self.beta_arg) == 2:
            return self.beta_arg[0] + (self.beta_arg[1] - self.beta_arg[0]) * self._n_updates / total_timesteps

        if len(self.beta_arg) == 3:
            return self.beta_arg[0] + (self.beta_arg[1] - self.beta_arg[0]) * min(1, self._n_updates / (total_timesteps * self.beta_arg[2]))

        return self.beta_arg
        
    def intrinsic_reward(self, state: th.Tensor):
        pred, target = self.rnd(state)
        return th.mean((pred - target).pow(2), dim=1).detach().cpu().numpy()

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            self.last_batch=replay_data

            # Do not backpropagate gradient to the target network
            with th.no_grad():

                #calculate intrinsic rewards
                intrinsic_rewards = self.intrinsic_reward(replay_data.observations)
                intrinsic_rewards = th.FloatTensor(intrinsic_rewards).to(self.device)

                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Decouple action selection from value estimation
                # Compute q-values for the next observation using the online q net
                next_q_values_online = self.q_net(replay_data.next_observations)
                # Select action with online network
                next_actions_online = th.argmax(next_q_values_online, dim=1)
                # Estimate the q-values for the selected actions using target q network
                next_q_values = th.gather(next_q_values, dim=1, index=next_actions_online.unsqueeze(1)).squeeze(1)
                # 1-step TD target
                target_q_values = (replay_data.rewards.squeeze() +self.beta()*intrinsic_rewards).flatten()+self.gamma*next_q_values*(1-replay_data.dones).flatten()


            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long()).flatten()

            # Check the shape
            assert current_q_values.shape == target_q_values.shape

            # Compute loss (L2 or Huber loss)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)            
            losses.append(loss.item())

            # Optimize the q-network
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            self.rnd.train(replay_data.observations)

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))