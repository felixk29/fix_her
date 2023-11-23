## Thesis Proposal 
### 1. Motivation
why interesting field?
why interesting topic/question?

 - enhancing exploration 
 - enhancing generalization (especially in MTL)


### 2. Hypothesis
maybe: "Learning to return to unvisited states increases teh generalization in mutlti task learning."
### 3. Experiments
look at max paper, use his experiment as baseline, with different versions: 
 - one with epsilon (multiple durations of exploring)
 - one with go explore tp to half goal (position of a random state from replay buffer),
 - one with uvf go to half goal  
### 4. Formalization
look at drl lectures, think about loss functions, how to structure shit,   
from slides:  
task identifier $i\in T \sub \mathbb{R}^m$  
if we know $P_i(s_{t+1}|s_t,a_t) and r_i(s_t,a_t)$   
$Q^*(s,a,i) = r_i(s,a) + \gamma  \int P_i(s'|s,a) max_{a'\in A}Q^*(s',a',i)ds'$  
in HER:  
$P_i(s_{t+1}|s_t,a_t)=P(s_{t+1}|s_t,a_t), \forall i \in T$  
reward is distance $r_i(s,a):=-||s'_{goal}-i||^2$, with $s'_{goal}\sub s'$  
$L[\theta]:=\mathbb{E}[(-||s'_{goal}-i||^2 + \gamma \, max_{a'\in A} \,Q_{\theta'}(s',a',i)-Q_{\theta}(s,a,i))^2|<s,a,s'> \sim D, i \sim T]$






### 5. Literature Review 
 - HER
 (https://arxiv.org/pdf/1707.01495.pdf)    

 - GO EXPLORE
(https://arxiv.org/pdf/1901.10995.pdf) 

 - The Role of Diverse Replay for Generalisation in
Reinforcement Learning
(https://arxiv.org/pdf/2306.05727.pdf)

 - Generalized Hindsight for Reinforcement Learning:
 (https://proceedings.neurips.cc/paper_files/paper/2020/file/57e5cb96e22546001f1d6520ff11d9ba-Paper.pdf)
