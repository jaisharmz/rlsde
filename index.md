---
layout: default
title: RL SDE Technical Writeup
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Modeling RL Trajectories as Stochastic Differential Equations

### Authors: Jai Sharma, Rithwik Nukala, Christopher Sun, Kourosh Salahi

---

## 1. Theoretical Framework

In this work, we model the trajectory of a Reinforcement Learning agent through state space as a Stochastic Differential Equation (SDE):

$$dX_t = f(X_t, u_t) dt + \sigma(X_t, u_t) dW_t$$

Where $f(X_t, u_t)$ is the **drift** (agent intent/signal) and $\sigma(X_t, u_t)$ is the **diffusion** (environmental noise).

Using Ito's Lemma and the Hamilton-Jacobi-Bellman (HJB) logic, we derive the relationship:

$$\rho V(x) = \max_u \left\{r(x, u) + f(x, u)\frac{\partial V}{\partial x} + \frac{1}{2} \sigma^2 \frac{\partial^2 V}{\partial x^2}\right\}$$

### Reward Hacking as Instability
We interpret **reward hacking** as rewards that are unsustainable to exploit due to stochasticity. If the agent spends most of its effort trying to offset "random jitters" ($\frac{1}{2} \sigma^2 \nabla^2 V$) to stay in a high-reward state, it has likely found a loophole.

---

## 2. Experimental Results: Reacher-v5

We tested this hypothesis on the Reacher-v5 environment. We introduced a "glitch" zone: a high-reward circle that applies random Gaussian noise to the agent's joints when approached.

### Scenario A: Original Model (No Glitch)
The agent behaves normally, reaching for the intended target.
<video width="100%" controls>
  <source src="videos/reacher_original.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Scenario B: Exploiting the Reward Hack
The agent ignores the target and vibrates erratically inside the "glitch" zone to farm rewards, despite the high environmental noise.
<video width="100%" controls>
  <source src="videos/reacher_glitch.mp4" type="video/mp4">
</video>

### Scenario C: Laplacian-Augmented Loss
By adding a weighted **Laplacian term** ($\nabla^2 V$) to the training loss, the agent learns to avoid regions where the value function is highly sensitive to noise.
<video width="100%" controls>
  <source src="videos/reacher_fixed.mp4" type="video/mp4">
</video>

---

## 3. Conclusion
The **Laplacian of the value function** is a robust detector for reward hacking. By penalizing high diffusion sensitivity, we can force agents to seek "stable" rewards rather than "noisy" exploits.