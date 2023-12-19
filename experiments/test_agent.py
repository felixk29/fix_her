import numpy as np
import gymnasium as gym
from agent import DQNAgent, Transition
from collections import deque
import environments as envs
from time import time
print("Done importing!")

#as I have to alter the algorithm substantially I will do it from "scratch" as to not have any differences in base dqn 
#but will use following as a reference,
#https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py

# Standard DQN Test
env = envs.FourRoom()
state_size, action_size = env.spaces()
print(state_size, action_size)

replay_buffer = deque(maxlen=50000)
agent = DQNAgent(state_size, action_size)
frames=0
eps=1

for episode in range(max_episodes:=100):
    state = env.reset()
    done = False
    total_reward = 0
    start=time()

    while not done:
        
        action = agent.select_action(state, epsilon=1)
        frames+=1
        if frames % 100 == 0:
            eps = max(0.01, eps*0.99)
        next_state, reward, done, trun, info = env.step(action)
        replay_buffer.append(Transition(state, action, next_state, reward, done))
        agent.train(replay_buffer, batch_size=128, gamma=0.99)

        state = next_state
        total_reward += reward

    if episode % 5 == 0:
        agent.update_target_model()
    

    print(f"Episode: {episode+1}, Total Reward: {total_reward}, Time: {round(time()-start,3)}s, Frames:{frames}, Epsilon: {eps}")

env.close()