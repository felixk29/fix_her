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