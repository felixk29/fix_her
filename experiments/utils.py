from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from four_room.utils import obs_to_state
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import torch as th
from stable_baselines3.common.vec_env import VecVideoRecorder
import imageio.v3 as iio
import os

colors={'base':'blue','randomStart':'royalblue',
        'rnd_base':'hotpink','intrinsicRandomWalk':'lightcoral',
        'hergo':'firebrick','tp':'forestgreen'}

names={'base':'Baseline','randomStart':'Pure Exploration - Epsilon Greedy',
        'rnd_base':'Baseline with Intrinsic Reward','intrinsicRandomWalk':'Pure Exploration - Intrinsic Reward',
        'hergo':'GoExploit','tp':'Teleport'}

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

class ExplorationCoverageCallback_Max(BaseCallback):
    def __init__(self, log_freq, total_states, num_actions, verbose=0):
        super(ExplorationCoverageCallback, self).__init__(verbose)
        self.state_action_coverage_set = set()
        self.log_freq = log_freq
        self.total_state_actions = total_states*num_actions

    def _on_step(self) -> bool:
        for i, obs in enumerate(self.locals['env'].buf_obs[None]):
            action = self.locals['actions'][i]
            self.state_action_coverage_set.add(hash((hash(obs.data.tobytes()), hash(action.data.tobytes()))))

        if self.num_timesteps % self.log_freq == 0:
            self.logger.record('train/state_action_coverage_exploration', len(self.state_action_coverage_set) / self.total_state_actions)

        return True
    
class ExplorationCoverageCallback(BaseCallback):
    def __init__(self, name, log_freq, total_states, num_actions, state_action=True, verbose=0):
        super(ExplorationCoverageCallback, self).__init__(verbose)
        base="./experiments/logs/diversity"
        os.makedirs(base, exist_ok=True)
        file_name = f"{base}/{name}.txt"
        self.file = open(file_name, "w")
        self.does_state_action=state_action
        self.coverage = set()
        self.log_freq = log_freq
        self.max_val = total_states*num_actions if state_action else total_states

    def _on_step(self) -> bool:
        self.coverage=set()
        if self.num_timesteps % self.log_freq == 0:
            r=self.locals['replay_buffer']
            sample=r.sample(r.size())
            for i in range(r.size()):
                if self.does_state_action:
                    self.coverage.add(hash((hash(sample.observations[i].cpu().numpy().tobytes()), hash(sample.actions[i].cpu().numpy().tobytes()))))
                else:
                    self.coverage.add(hash(sample.observations[i].cpu().numpy().tobytes()))
            self.file.write(f"{len(self.coverage) / self.max_val}\n")

        return True

    def __del__(self):
        self.file.close()
_HEATMAP = False
def obs_to_entry(obs):
    state=obs_to_state(obs)
    data=state[:3]
    name=str(state[3:])   

    return name, data

class UVFStepCounterCallback(BaseCallback):
    def __init__(self, log_freq, verbose=0):
        super(UVFStepCounterCallback, self).__init__(verbose)
        self.uvf_steps = set()
        self.uvf_steps_buffer=set()
        self.log_freq = log_freq

    def _on_step(self) -> bool:

        new=self.locals['uvf_stepcount_history']

        for i in new:
            start=obs_to_state(i[5])
            end=obs_to_state(i[3])
            goal=obs_to_state(i[4])
            dis_from_start=np.sqrt((start[0]-end[0])**2+(start[1]-end[1])**2)
            dis_to_goal=np.sqrt((goal[0]-end[0])**2+(goal[1]-end[1])**2)
            self.uvf_steps_buffer.add((i[0],i[1],i[2],dis_to_goal,dis_from_start,i[6]))

        diff=self.uvf_steps_buffer-self.uvf_steps

        if self.num_timesteps % self.log_freq == 0 and len(diff)>0:
            self.logger.record('uvf_stepcount/uvf_stepcount_mean', sum([i[2] for i in diff])/len(diff))
            self.logger.record('uvf_stepcount/uvf_stepcount_median', np.median([i[2] for i in diff]))
            self.logger.record('uvf_stepcount/uvf_stepcount_max', max([i[2] for i in diff]))
            self.logger.record('uvf_stepcount/uvf_stepcount_min', min([i[2] for i in diff]))

            self.logger.record('uvf_end_goal/uvf_dis_end_to_goal_mean', np.mean([i[3] for i in diff]))
            self.logger.record('uvf_end_goal/uvf_dis_end_to_goal_median', np.median([i[3] for i in diff]))
            self.logger.record('uvf_end_goal/uvf_dis_end_to_goal_max', np.max([i[3] for i in diff]))
            self.logger.record('uvf_end_goal/uvf_dis_end_to_goal_min', np.min([i[3] for i in diff]))

            self.logger.record('uvf_start_end/uvf_dis_start_to_end_mean', np.mean([i[4] for i in diff]))
            self.logger.record('uvf_start_end/uvf_dis_start_to_end_median', np.median([i[4] for i in diff]))
            self.logger.record('uvf_start_end/uvf_dis_start_to_end_max', np.max([i[4] for i in diff]))
            self.logger.record('uvf_start_end/uvf_dis_start_to_end_min', np.min([i[4] for i in diff]))

            l=[i[5] for i in diff]
            self.logger.record('uvf_goal/uvf_done_done',l.count('done'))
            self.logger.record('uvf_goal/uvf_done_full_equal',l.count('full_equal'))
            self.logger.record('uvf_goal/uvf_done_uvf_val_decreased',l.count('uvf_val_decreased'))
            self.logger.record('uvf_goal/uvf_done_max_steps',l.count('max_steps'))

            self.logger.record('train/uvf_steps',self.locals['saving_uvf_timesteps'])
            self.logger.record('train/total_steps',self.locals['saving_total_timesteps'])
            
            self.uvf_steps.update(self.uvf_steps_buffer)
            self.uvf_steps_buffer=set()

        return True


def state_to_obs(state):
    """
    Turn a state tuple back into a numpy observation array.
    """
    # Create an empty observation array
    obs = np.zeros((4, 9, 9))

    # Unpack the state tuple
    player_loc_x, player_loc_y, player_dir, goal_loc_x, goal_loc_y, door_pos_up, door_pos_down, door_pos_left, door_pos_right = state

    # Center the player location
    center_x, center_y = 4, 4  # Center of the 9x9 grid
    player_loc_x = center_x + (player_loc_x - center_x)
    player_loc_y = center_y + (player_loc_y - center_y)

    # Set the player location and direction
    obs[0, player_loc_y, player_loc_x] = 1
    if player_dir == 0:  # right
        obs[1, player_loc_y, player_loc_x+1] = 1
    elif player_dir == 1:  # down
        obs[1, player_loc_y+1, player_loc_x] = 1
    elif player_dir == 2:  # left
        obs[1, player_loc_y, player_loc_x-1] = 1
    elif player_dir == 3:  # up
        obs[1, player_loc_y-1, player_loc_x] = 1

    # Set the goal location
    obs[3, goal_loc_y, goal_loc_x] = 1

    # Set the walls and doors
    obs[2] = np.zeros((9, 9))
    obs[2, 0, :] = 1
    obs[2, 8, :] = 1
    obs[2, :, 0] = 1
    obs[2, :, 8] = 1
    obs[2, 4, :] = 1
    obs[2, :, 4] = 1

    obs[2, door_pos_up+1, 4] = 0
    obs[2, door_pos_down+5, 4] = 0
    obs[2, 4, door_pos_left+1] = 0
    obs[2, 4, door_pos_right+5] = 0

    # Shit so player is in center
    obs = np.roll(obs, (4-player_loc_x, 4-player_loc_y), axis=(2, 1))

    return obs

class heatmapCallback(BaseCallback):
    def __init__(self, save_dir: str = './experiments/logs/heatmaps', log_freq: int = 100000, verbose=0, id: str = "", hergo=False):
        super(heatmapCallback, self).__init__(verbose)
        self.hergo=hergo
        self.log_freq = log_freq
        self.save_dir = save_dir
        self.first = True
        self.id = id
        self.dir_dict = {0: '→', 1: '↓', 2: '←',
                         3: '↑', -1: 'G', -2: 'X', -3: ' '}

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0 or self.first:
            self.first= False
            tik = time()
            nr = self.num_timesteps//self.log_freq
            if self.hergo:
                heatmap, annot = self.calculateHeatmapHergo()
            else:
                heatmap, annot = self.calculateHeatmap()

            max_val = np.max(heatmap)*1.05
            min_val = np.min(heatmap[heatmap > -90])
            min_val = min_val/2 if min_val > 0 else min_val
            sns.heatmap(heatmap, annot=annot, fmt='',
                        vmin=min_val, vmax=max_val)
            plt.savefig(f"{self.save_dir}/heatmap{self.id}_{nr}.png")
            plt.clf()
            print(f"Heatmap generated. Took {round(time()-tik,2)}s")

        return True

    def calculateHeatmap(self):
        nn = self.locals['self'].q_net
        goal = self.locals['self']._last_obs[0]

        env = obs_to_state(goal)
        walls = env[-4:]  # walls are up down left right range= 0,1,2

        heatmap = np.zeros((9, 9, 4))-100

        for y in range(1, 8):
            for x in range(1, 8):
                if (x == 4 or y == 4) and not ((x, y) in [(4, 1+walls[0]), (4, 5+walls[1]), (1+walls[2], 4), (5+walls[3], 4)]):
                    continue  # checks so only doorways are calculated

                for dir in range(4):  # dir 0-3 : right down left up
                    obs = th.Tensor(state_to_obs((x, y, dir, *env[3:]))).to('cuda' if th.cuda.is_available() else 'cpu')
                    heatmap[y, x, dir] = th.max(nn(th.Tensor(obs))).cpu().detach().item()

        values_2d = np.max(heatmap, axis=2)
        annot = np.argmax(heatmap, axis=2)
        annot[values_2d == -100] = -3
        annot[env[4], env[3]] = -1
        annot = np.array([self.dir_dict[i]
                         for i in annot.reshape((81))]).reshape((9, 9))

        return values_2d, annot

    def calculateHeatmapHergo(self):
        nn = self.locals['self'].q_net
        goal = self.locals['self']._last_obs['desired_goal'][0]

        env = obs_to_state(goal)
        walls = env[-4:]  # walls are up down left right range= 0,1,2

        heatmap = np.zeros((9, 9, 4))-100

        for y in range(1, 8):
            for x in range(1, 8):
                if (x == 4 or y == 4) and not ((x, y) in [(4, 1+walls[0]), (4, 5+walls[1]), (1+walls[2], 4), (5+walls[3], 4)]):
                    continue  # checks so only doorways are calculated

                for dir in range(4):  # dir 0-3 : right down left up
                    obs = state_to_obs((x, y, dir, *env[3:]))
                    dic = {'observation': th.Tensor(obs), 'achieved_goal': th.Tensor(
                        obs), 'desired_goal': th.Tensor(goal)}

                    heatmap[y, x, dir] = th.max(nn(dic)).cpu().detach().item()

        values_2d = np.max(heatmap, axis=2)
        annot = np.argmax(heatmap, axis=2)

        annot[values_2d == -100] = -3
        annot[env[4], env[3]] = -2
        annot[env[1], env[0]] = -1
        
        annot = np.array([self.dir_dict[i]
                         for i in annot.reshape((81))]).reshape((9, 9))

        return values_2d, annot

class randomStepsCallback(BaseCallback):
    def __init__(self, verbose: int = 0, path:str = './experiments/logs/random_steps'):
        super().__init__(verbose)
        self.path = path
    
    def _on_step(self):
        if self.locals['random_steps'] is not None:
            self.logger.record('train/random_steps', self.locals['random_steps'])
        return True
class renderEpisodeCallback(BaseCallback):
    def __init__(self, save_dir: str = './experiments/logs/episodes', id:str="",env = None, log_freq: int = 100000, verbose=0, hergo=False):
        super(renderEpisodeCallback, self).__init__(verbose)
        self.env=env
        self.id=id
        self.hergo=hergo
        self.log_freq = log_freq
        self.save_dir = save_dir
        self.once=True

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0  or self.once:
            self.once=False

            env=self.env

            imgs=[]
            obs,_ = env.reset()

            if self.hergo:
                goal=obs_to_state(obs['desired_goal'])

            img=env.render()
            #
            done,trunc = False,False
            imgs.append(img)
            while not done and not trunc:
                action, _ = self.model.predict(obs)
                obs, _, done, trunc , _ = env.step(action)
                img=env.render()

                #marking goal
                if self.hergo:
                    img[(goal[1]*32)+1:(goal[1]+1)*32,(goal[0]*32)+1:(goal[0]+1)*32]=[0,0,255]

                imgs.append(img)
            iio.imwrite(f"behaviour{self.id}_{self.num_timesteps//self.log_freq}.gif",imgs,duration=100,loop=0)
        return True
