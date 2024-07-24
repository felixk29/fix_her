### Master Thesis Project Code Base
Code is only in the experiments section
##
 - the code in ./experiments/max_sb3 was taken from this project: [Deep Exploration](https://github.com/MWeltevrede/stable-baselines3/tree/feature/deep-exploration)
 - the code in ./experiments/four_room was taken from this project and altered to allow state loading: [Four_Room](https://github.com/MWeltevrede/four_room)
## Novel methods include 
 - Pure Exploration
   - epsilon-greedy based (moveRandom.py)
   - intrinsic reward based (intrinsicRandomWalk.py)
 - Teleportation (tpdqn.py)
 - Hergo (hergo.py)

## Other Files:
 - doubledqn.py  - DoubleDQN implementation as it is not included in stable_baselines3
 - fixedHER.py - simple bug fix for HER ReplayBuffer of stable_baselines3, does not work for all environments (done Flag problem)
 - test_agent.py - File used to run all experiments, not used for Distributed Testing
 - RND.py - RND module implementation
 - uvf.py - toy implementation of adapting env to be goal oriented to test uvf
 - uvf.py - same as uvf.py but also changed env actions space [up, down, left, right] from [forward, turn left, turn right]
 - utils.py - various utility functions, mostly callbacks
 - visualization.ipynb - Notebook for reproducable visualization, partly generalized
 - adapted_vec_env.py - an adapted Version of vec_env from stable_baselines3 that allows passing on of data
 - four_room/old_env.py - original environment
 - four_room/env.py - adapted environment that allows state loading in various ways
