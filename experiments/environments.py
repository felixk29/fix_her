import gymnasium as gym

class envInterface:
    def __init(self):
        self.env=None

    def reset(self):
        pass
    
    def spaces(self):
        pass
    
    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

class PoleCart(envInterface):
    def __init__(self, render=False):
        if render:
            self.env = gym.make('CartPole-v1', render_mode='human')
        else:
            self.env = gym.make('CartPole-v1')
    
    def spaces(self):
        o= self.env.observation_space.shape[0]
        a= self.env.action_space.n
        return o,a

    def reset(self):
        return self.env.reset()[0]

class MountainCar(envInterface):
    def __init__(self, render=False):
        if render:
            self.env = gym.make('MountainCar-v0', render_mode='human')
        else:
            self.env = gym.make('MountainCar-v0')
    
    def spaces(self):
        o= self.env.observation_space.shape[0]
        a= self.env.action_space.n
        return o,a

    def reset(self):
        return self.env.reset()[0]

# environment taken from https://github.com/MWeltevrede/four_room

import dill
from four_room.env import FourRoomsEnv
from four_room.env_wrappers import gym_wrapper

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open('four_room/configs/fourrooms_train_config.pl', 'rb') as file:
    train_config = dill.load(file)


class FourRoom(envInterface):
    def __init__(self,render=False):
        self.env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1', 
                                agent_pos=train_config['agent positions'], 
                                goal_pos=train_config['goal positions'], 
                                doors_pos=train_config['topologies'], 
                                agent_dir=train_config['agent directions']))
    
        self.render=render #apparently a render function is in utils.py should be looked at. 

    def spaces(self):
        o= self.env.observation_space.shape[0]
        a= self.env.action_space.n
        return o,a

    def reset(self):
        return self.env.reset()[0]