import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple,deque

import environments as envs

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        
        #self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
                q_values = self.model(state)
                return q_values.argmax().item()

    def train(self, replay_buffer, batch_size, gamma):
        if len(replay_buffer) < batch_size:
            return

        transitions = random.sample(replay_buffer, batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        q_values = self.model(state_batch).gather(1, action_batch)
        next_q_values = self.target_model(next_state_batch).max(1)[0].unsqueeze(1)
        expected_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)

        loss = F.smooth_l1_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


if __name__ == '__main__':
    # Example usage
    env = envs.MountainCar()
    state_size, action_size = env.spaces()
    print(state_size, action_size)

    replay_buffer = deque(maxlen=2000)
    agent = DQNAgent(state_size, action_size)

    for episode in range(max_episodes:=100):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            if episode > (max_episodes - 1):
                env.render()

            action = agent.select_action(state, epsilon=0.1)
            next_state, reward, done, trun, info = env.step(action)
            replay_buffer.append(Transition(state, action, next_state, reward, done))
            agent.train(replay_buffer, batch_size=128, gamma=0.9)

            state = next_state
            total_reward += reward

        if episode % 5 == 0:
            agent.update_target_model()

        print(f"Episode: {episode+1}, Total Reward: {total_reward}")

    env.close()

