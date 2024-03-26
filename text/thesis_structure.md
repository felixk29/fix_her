# Thesis
## 1. Introduction & Motivation
Generalizationability is important for agents, Effect of Buffer underresearched. 


## 2. Hypothesis

Hypothesises:
 - "SpiderWeb Buffer" best option
 - Generalization improved through higher variance of starting positions
 - Way Buffer is filled has impact on generalizability of agent

## 3. Background
Explaining following concepts: 
 - MDP:  
    explaining formal part,
 - CMDP:  
    explaining formal part, as importat to understand what context, state, actions, state action pair means
  - (Zero-Shot) Generalization:  
    What does generalization mean, why does it matter and what options exist to improve it so far, 
 - Replay Buffer:  
    explaining replay buffer and what difference is between onpolicy/offpolicy learning and why it matters for us 
 - DQN

 - Exloration:
   **include epsilon** and other things
 - RND:   
   What is RND, how does it work, what is it used for?
 - Four Room Minigrid:
   Explain the enviornment, how it works, what its based on, action & observation space, What context changes, explaining reachability of states etc.

if UVF works:

 - UVF's:  
   What is an UVF, how does it work? why would you use it, what are the drawbacks (why not just use an uvf for the whole agent), 
 - HER 
## 4. Related Work
Papers that will be mentioned:  
 - Max paper
 - Go Explore 
 - https://arxiv.org/pdf/2306.05483.pdf
 - [Efficient Self-Supervised Data Collection for Offline Robot Learning](https://arxiv.org/pdf/2105.04607.pdf) 
 - [Generalized Hindsight for Reinforcement Learning](https://arxiv.org/pdf/2002.11708.pdf)
## 5. Methodology
This chapter will explain how things were implemented and how they work and what they do, what their idea was and why they should work. 
### 5.1 Architecture
actual agent is the same in all the implementations, DQN with CNN etc, 
### 5.2 Teleportation
Explaining how it works, RND inclusion, why it should work, proof of concept, idea taken from go explore, could be considered cheating but if this doesnt work nothing does, hyperparameter tp percentage,
### 5.3 Random-Walk
Explaining how it works, epsilon greed with epsilon of 1 for start, no RND, pure uniform action sampling, hyperparameter step length,  
### 5.4 Intrinisic Exploration Random Walk
rnd with very high beta at start and drop off later, Smarter Extension of Random-Walk with intirinsic exploration, explain implementation of intrinsic exploration
### 5.5 UVF
Explain idea, explain architecture of uvf agent, explain how algorithm works, explain various tried methods (uvf value decreases) and downfall,  

## 6. Experiments
This chapter will showcase various results of the previously explained methods with silly little graphs. Also will come to some conclusions on why something worked/hasn't worked. 
### 6.1 Teleportation
### 6.2 Random-Walk
### 6.3 Intrinisic Exploration Random Walk
### 6.4 UVF
Explain that uvf doesn't work with small buffersize, even with big buffersize too sampleinefficient, 

## 7. Conclusion
Drawing conclusion from the data, which method worked the best, which offers the biggest advantage with least drawback, 

## A.1 Appendix
All the tables of hyperparameters that were used etc. 
Implementation details etc

# Questions:
 - Where do I mention stable-baselines? in Architectures? Appendix