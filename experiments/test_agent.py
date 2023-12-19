import numpy as np
import gymnasium as gym
from agent import DQNAgent, Transition
from collections import deque
import environments as envs

import matplotlib.pyplot as plt
from four_room.utils import obs_to_img, obs_to_state
print("Done importing!")

# Standard DQN Test
env = envs.FourRoom()
state_size, action_size = env.spaces()
print(state_size, action_size)

replay_buffer = deque(maxlen=2000)
agent = DQNAgent(state_size, action_size)

for episode in range(max_episodes:=100):
    state = env.reset()
    print(state)
    print(obs_to_state(state))
    img=obs_to_img(state)
    plt.imshow(img)
    plt.show()

    done = False
    total_reward = 0

    while not done:
        
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