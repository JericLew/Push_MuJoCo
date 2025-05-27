# `Push_MuJoCo`: PPO for Block Pushing Task with Robot Arm

## Overview
Hi! This is our group's work for the graduate level course [ME5406 Deep Learning for Robotics](https://nusmods.com/courses/ME5406/deep-learning-for-robotics) @ National University of Singapore (NUS).

### Summary Video
[![Watch the video](https://raw.githubusercontent.com/JericLew/Push_MuJoCo/main/mujoco_env.png)](https://drive.google.com/file/d/1vmEVcVk4UAusThqe8NbUYnQM-RlgJ4XU/view)

We are using the deep reinforcement learning algorithm PPO to learn behaviours for our specified block pushing task with a 7 DoF Franka Emika Panda robot arm.

-   [`train.py`](train.py): Entry point for training script. Contains hyperparameters and setup for training.
-   [`eval.py`](eval.py): Entry point for evalutaion script. Contains hyperparameters and setup for evaluation.
-   [`PPO.py`](PPO.py): Contains our implementation of the PPO algorithm with GAE and behaviour cloning loss.
-   [`push_custom.py`](push_custom.py): Contains our custom `gymnasium` learning environment with MuJoCo python pindings for simulation.
-   [`inverse_kinematics.py`](inverse_kinematics.py): Google Deepmind's [dm_control](https://github.com/google-deepmind/dm_control) inverse kinematics implementation for MuJoCo.
-   [`model`](model): Directory containing our neural network implementation for actor and critic
-   [`franka_emika_panda`](franka_emika_panda): Directory containing assets and MJCF description files for MuJoCo simulation.
-   [`saved_pth`](saved_pth): Directory containing saved weights for our neural networks.

## Setup
Setup the `conda` environment for our repository by running
```
conda env create -f environment.yml
conda activate push_mujoco
```

Alternatively, you could set up the environment with `requirements.txt`
```
conda create -n push_mujoco python=3.9
conda activate push_mujoco
pip install -r requirements.txt
```
NOTE: MuJoCo Python require at least one of the three OpenGL rendering backends: EGL (headless, hardware-accelerated), GLFW (windowed, hardware-accelerated), and OSMesa (purely software-based). Refer to Google Deepmind's [dm_control](https://github.com/google-deepmind/dm_control?tab=readme-ov-file#rendering) repository for further instructions. Typically, try `export MUJOCO_GL=egl` if you are facing OpenGL rendering related errors.

## Usage
To train from scratch:
```
python train.py
```

To test model:
```
python eval.py
```

Hyperparameters in both [`train.py`](train.py) and [`eval.py`](eval.py) can be changed. Some important hyperparameters include:
-   `privileged` is a boolean flag to determine whether to train/evaluate the privileged actor or vision actor.
-   `random_object_pos` boolean flag to determine whether the object position is randomised in the simulation environmnet


## Acknowledgement
### MuJoCo Assets
```bash
├── franka_emika_panda
│   ├── assets
│   │   ├── finger_0.obj
│   │   ├── ...
│   └── panda_push.xml
│   └── scene_push.xml
```
The robot arm model used is from Google Deepmind's [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) repository. Specifically, we use the [Franka Emika Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda) model which is a 7-axis robot arm.
- `panda_push.xml` MJCF description file is modified from their `panda_nohand.xml` MJCF description file to include a rounded rod-shaped end effector for the block pushing task.
- `scene_push.xml` MCJF description file is heavily modified from their `scene_xml`MCJF description file to include objects in our custom MuJoCo simulation environment like a table, a coloured block, two cameras for image input and 3 coloured target regions.
- `assets` directory includes all assets from the repository for the Franka Emika Panda robot arm. 

### Neural Networks
```bash
├── model
│   ├── common
│   │   ├── cnn.py
│   │   └── mlp.py
│   └── push_actor.py
│   └── push_critic.py
```
Most of the neural network has been written from scratch with the exception of `mlp.py` which is based on the implementation of MLP and residual style MLP networks from the [D3IL](https://github.com/ALRhub/d3il/blob/main/agents/models/common/mlp.py) and [DPPO](https://github.com/irom-princeton/dppo/blob/main/model/common/mlp.py) repositories. While MLP are easy to code, these clean and modular implementation help keep the code in `push_actor.py` and `push_critic.py` clean and readable.

### Inverse Kinematics
The implementation of inverse kinematics for the MuJoCo simulator in `inverse_kinematics.py` is taken from Google Deepmind's [dm_control](https://github.com/google-deepmind/dm_control) repository which is meant for Reinforcement Learning in physics-based simulation like MuJoCo.

### PPO
The implementation of PPO in `PPO.py` is heavily modified version from Eric Yu's [PPO for Beginners](https://github.com/ericyangyu/PPO-for-Beginners) repository. Some of the modifications of our PPO implementation is with reference to the [DPPO](https://github.com/irom-princeton/dppo) repository.

### Gymnasium Environment
The `gymnasium` environment for the block pushing task in `push_custom.py` is written from scratch with reference the to `gymnasium` [documentation](https://gymnasium.farama.org/).

## Package Versions
-   mujoco==3.3.0
-   gymnasium==1.1.1
-   torch==2.6.0
-   torchvision==0.21.0
-   wandb==0.19.9
