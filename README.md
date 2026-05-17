# Zero-Sum Policy Gradient Methods for Crash-Cost Markov Games
Car-Bus and Rock-Paper-Scissors Research Game CS298

**Author:** Samarah Hiba  
**Institution:** University of Wisconsin–Madison  
**Research Project:** CS298 (AI Research)

---
## Research Questions
1. Do policy gradient methods converge in zero-sum Markov games?
2. Does REINFORCE exhibit policy oscillation?
3. Can A2C make training more stable than policy gradient methods (I will use REINFORCE)?
4. How does the size of the crash cost influence the agents’ equilibrium behavior and learned strategies?


## Overview
This project implements:
- REINFORCE
- Advantage Actor-Critic (A2C)
- Deep Minimax-Q

on a grid-based zero-sum crash-cost driving environment.

## Motivation
This work explores convergence behavior in zero-sum Markov games and policy oscillation under policy gradient methods.

## Methods
- Policy gradient methods such as REINFORCE
- A2C
- LP-based Minimax solver
- Target networks + replay buffers

## Results
- Policy oscillations observed in zero-sum setup, especially when agents updated simultaneously using policy gradients.
- Averaging strategies converged toward mixed equilibrium over time.
- Comparison plots included to clearly illustrate convergence trends, oscillatory behavior and differences between learning methods: A2C and REINFORCE in particular
- A2C training was noticeably less noisy than REINFORCE and generally converged more consistently.
- Increasing the crash-cost magnitude significantly altered equilibrium behavior, encouraging more conservative agent strategies and reducing risky actions.
