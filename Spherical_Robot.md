# 强化学习基础知识
V:状态价值   子节点Q的期望
Q:动作价值   子节点V的期望加上奖励R

V 的估算:

MC:$ V(S_t)\leftarrow V(S_t) + \alpha [G_t - V(S_t)] $ 
(G 为一条路的 R 总和, 求 G 的平均值)

TD:$ V(S_t)\leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(R_{t+1}) - V(S_t)] $ 
(不用走完全程，用 R 加上下一目标点的 V 值作为此点的 V 值)

Q 的估算:

SARSA:$ Q(S,A)\leftarrow Q(S,A) + \alpha [R_{t+1} + \gamma Q(S',A') - Q(S,A)] $ 

Qlearning:$ Q(S,A)\leftarrow Q(S,A) + \alpha [R_{t+1} + \gamma \max_a Q(S',a) - Q(S,A)] $ 


# sim to real方法
## Fine-Tuning in the real world
when the robot inevitably fails, it would need the mechanisms necessary to recover and 
fine-tune its skills to this new environment.

## SIMULATION-GUIDED FINE-TUNING
$$V_H^{\pi_H}(s)=\mathbb{E}\left[\gamma^HV_{sim}(s_H)+\sum_{t=0}^{H-1}\gamma^tr(s_t)-V_{sim}(s_0)|s_0=s,a_t\sim\pi_{H,t}(\cdot|s_t)\right]$$

$$V_H^*(s):=\sup_{\pi_H}V_H^{\pi_H}(s)$$

$$Q_H^*(s,\pi):=\mathbb{E}_{a\sim\pi(\cdot|s)}\left[\gamma V_H^*(s^{\prime})+r(s)\right]$$

$$\pi_H^*(\cdot|s)\leftarrow\sup_\pi Q_H^*(s,\pi)$$

![alt text](figure/SGFT.jpg)

## Closing the Sim-to-Real Loop:Adapting Simulation Randomization with Real World Experience
sim to real method: domain randomization ----randomizing relevant parameters
* SimOpt 
  minimize discrepancy between real world observation trajectories and simulated observation trajectories
![alt text](figure/SimOpt_frame.jpg)

* BayesSim
  a full Bayesian treatment for the parameters of the simulator

simulator 在参数$ \theta $时，$ X^s=g(\theta) $

## Zero-Shot Sim-to-Real RL Policy
* step1. **SysID** : calibrate the quadrotor’s dynamics by estimating key parameters 
* step2. **DR** : domain randomization
* step3. **PPO** : 
  *CTBR*: adopt CTBR as the policy action space. 
  *input*: next N reference trajectory points, linear velocity, rotation matrix R
  *reward*:  $ r=r_{task}+\lambda r_{smooth} ,\ r_{smooth}=||u_t-u_{t-1}||^2$
* step4. **Real Robot**
![alt text](figure/SimpleFlight.jpg)

## Iterative residual tuning for system identification and sim-to-real robot learning

* IRT :TuneNet 

## DROID:
steps:
* human demonstration
* parameter identification
* policy learning with optimized DR

parameter identification
model distribution as a multivariate normal distribution$ \Phi(\mu,\Sigma) $
different parameter $ \phi $ ----different distribution $ \Phi $ 
Covariance Matrix Adaptation Evolution Strategy(CMA-ES) optimize parameter $ \phi $
### 方法,CMA-ES(黑盒优化)
ES--Evolution Strategies 进化策略
用正态分布进行采样$ N(m_t,\sigma^2C_t) $


# Spherical Robot Sim2Real
## SysID(system identify)

## Bayesian treatment
