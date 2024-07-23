# <font size=6>Demo2Test</font>
This repository contains the source code for the paper **Demo2Test: Transfer Testing of Adversarial Agent from Failure Demonstrations**.

# Overview

We propose Demo2Test for conducting transfer testing of adversarial agents in competitive games, i.e., leveraging the demonstrations of failure scenarios from the source task to boost the testing effectiveness in the target task.

Experimental evaluation on effectiveness of Demo2Test  on four transfer settings in the competitive game environment of MuJoCo with promising performance, outperforming four baselines.

The overview of Demo2Test is shown in the figure below:
![图片](images/overview.png)

# Environment Setup
## Required Environment 
we use [Sacred](https://github.com/IDSIA/sacred) for experiment configuration.

This codebase uses Python 3.7. The main binary dependencies are MuJoCo (version 1.3.1, for `gym_compete` environments).

The competitive environment (`gym_compete`) and agent model we test is provided from this awesome repository: https://github.com/openai/multiagent-competition
## Installation instructions
Install the required the packages inside the virtual environment:
```
$ conda create -n yourenvname python=3.7 anaconda
$ source activate yourenvname
$ conda install cudatoolkit==9.2
$ pip install -r requirements.txt
```
Main requirements:

```shell
gym[mujoco]==0.15.4
mujoco-py-131 @ git+https://github.com/AdamGleave/mujoco-py.git@mj131
gym_compete @ git+https://github.com/HumanCompatibleAI/multiagent-competition.git@3a3f9dc
```

# Running
**Run an experiment**

To test in the game of 'SumoAnts', using the source demonstrations in 'data/trajectories/RunToGoalAnts_source_traj.npz':
```shell
python -m aprl.train_demo2Test with game_name=SumoAnts-v0 expert_dataset_path=data/trajectories/RunToGoalAnts_source_traj.npz
```
The all config parameters are shown in train_demo2test.py.

You can test other game in `gym_compete` and using the official Agent Zoo to generate simulated source demonstrations.
The data format refers to the example of 'data/trajectories/RunToGoalAnts_source_traj.npz'.

# Trend Graph Results
The trend of number and unique number of failure scenarios found by Demo2Test and baselines:
![图片](images/RQ1_supplement.png)
Results show that in each time duration, Demo2Test outperforms all four baselines, whether considering the terminating state of the fault scenario or all states.

The details of transfer settings from T1 to T4 and the each baseline can be found in our paper.

# Unique Failures Discovered Solely by Demo2Test
![图片](images/UF_samples.png)
First, we cluster all failure scenarios found by baselines and Demo2Test, and each cluster represents a unique failure scenario. 
Across all tasks, Demo2Test can encompass all the unique failure scenarios discovered by the baselines. 
After that, we manually analyze the videos of failure scenarios found only by Demo2Test.

**T1-1**: The testing agent attacks the leg of target agent, causing failure. This stems from knowledge learned from demonstrations. 

**T1-2**: The testing agent lunges at the target agent, causing failure. This is caused by perturbations in key states.

**T1-3**: The testing agent rams the target agent, causing it to fly off, resulting in the failure. This is caused by perturbations in key states.

**T2-1**: The testing agent punches to knock down the target agent, causing failure. This is caused by perturbations in key states.

**T2-2**: The testing agent tackles the target agent, causing it to fall and fail. This stems from knowledge learned from demonstrations. 

**T2-3**: The testing agent trips the target agent after falling down, leading to failure. This is caused by perturbations in key states.

**T3-1**: The testing agent moves forward and crashes into the target agent, causing it to fall. This is caused by perturbations in key states.

**T3-2**: The testing agent trips the target agent, causing it to fall and be unable to move. This is caused by perturbations in key states.

**T3-3**: The testing agent forces the target agent to back away continuously to cause failures. This stems from knowledge learned from demonstrations. 

**T4-1**: The testing agent attacks the leg of target agent, causing failure. This stems from knowledge learned from demonstrations. 

**T4-2**: The testing agent uses its body to knock down the target agent, causing failure. This is caused by perturbations in key states.

**T4-3**: The testing agent uses its hands to push over the target agent, resulting in failure. This is caused by perturbations in key states.

# Reference
- https://github.com/HumanCompatibleAI/adversarial-policies/
- https://github.com/openai/multiagent-competition/
- https://github.com/Khrylx/PyTorch-RL/