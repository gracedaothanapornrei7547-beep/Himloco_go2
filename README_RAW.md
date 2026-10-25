# HimLoco Lab

## Project Overview

This project ports **HimLoco** from **IsaacGym** to **Isaac Lab**.

HimLoco is a reinforcement learning-based quadruped robot gait controller with a dual-network architecture. Through this project, you can train, export, and deploy HimLoco policies in the Isaac Lab environment.

## Project Structure

```
himloco_lab/
├── scripts/                          # Script files
│   ├── himloco_rsl_rl/
│   │   ├── train.py                  # Training script
│   │   ├── play.py                   # Batch inference and policy export
│   │   ├── play_interactive.py       # Interactive control script
│   │   └── cli_args.py               
│   ├── list_envs.py                  # List available environments
│   ├── zero_agent.py                 
│   └── random_agent.py               
│
├── source/himloco_lab/               # Main source code
│   └── himloco_lab/
│       ├── tasks/                    # Task definitions
│       │   └── locomotion/
│       │       ├── mdp/              # Observations, actions, rewards, terminations
│       │       └── robots/go2/       # Training configuration
│       │
│       ├── rsl_rl/                   # HimLoco algorithm implementation
│       │   ├── config/               # Algorithm configuration
│       │   ├── modules/              # HIMActorCritic, HIMEstimator networks
│       │   ├── algorithms/           # HIMOnPolicyRunner training logic
│       │   ├── wrappers/             # HimlocoVecEnvWrapper environment adapter for IsaacLab
│       │   └── env/                  # VecEnv interface
│       │
│       ├── utils/                    # Utility functions
│       │   └── export_policy.py      # JIT and ONNX export tools
│       │
│       └── assets/                   # Asset files
│           └── unitree/              # Unitree Go2 URDF and configuration
│
├── deploy/                           # Robot deployment code
│   └── robots/go2/                   # Go2 controller implementation
│
└── README.md                         # This file
```

## Algorithm Overview
[HimLoco](https://github.com/InternRobotics/HIMLoco/blob/main/projects/himloco/README.md)
![HimLoco](https://github.com/InternRobotics/HIMLoco/blob/main/assets/overview.jpeg)

## 📚 Installation Guide

### 1. Install Isaac Lab

Follow the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) to install Isaac Lab.
We recommend using the pip installation method.

### 2. Clone This Repository

Clone this repository into a separate directory outside of Isaac Lab:

```bash
git clone https://github.com/IsaacZH/himloco_lab.git
cd himloco_lab
```

### 3. Install HimLoco Lab

```bash
python -m pip install -e source/himloco_lab
```
### 4. Download unitree robot description files

  *Method 1: Using USD Files*
  - Download unitree usd files from [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main), keeping folder structure
    ```bash
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    ```
  - Config `UNITREE_MODEL_DIR` in `source/himloco_lab/himloco_lab/assets/robots/unitree.py`.

    ```bash
    UNITREE_MODEL_DIR = "</home/user/projects/unitree_usd>"
    ```

  *Method 2: Using URDF Files [Recommended]* Only for Isaacsim >= 5.0
  -  Download unitree robot urdf files from [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
      ```
      git clone https://github.com/unitreerobotics/unitree_ros.git
      ```
  - Config `UNITREE_ROS_DIR` in `source/himloco_lab/himloco_lab/assets/robots/unitree.py`.
    ```bash
    UNITREE_ROS_DIR = "</home/user/projects/unitree_ros/unitree_ros>"
    ```
  - [Optional]: change *robot_cfg.spawn* if you want to use urdf/usd files

## Quick Start

### Train Policy

Train a HimLoco policy on Unitree Go2:

```bash
python scripts/himloco_rsl_rl/train.py --task Unitree-Go2-Velocity --headless
```

### Inference and Playback

Run a trained policy for batch inference:

```bash
python scripts/himloco_rsl_rl/play.py --task Unitree-Go2-Velocity-Play
```

### Interactive Control

Use keyboard to control the robot in real-time in IsaacLab:

```bash
# Launch interactive control
python scripts/himloco_rsl_rl/play_interactive.py --task Unitree-Go2-Velocity-Play

# Keyboard controls:
#   Numpad 8 / ↑   : Move forward
#   Numpad 2 / ↓   : Move backward
#   Numpad 4 / ←   : Strafe right
#   Numpad 6 / →   : Strafe left
#   Numpad 7 / Z   : Rotate counter-clockwise
#   Numpad 9 / X   : Rotate clockwise
```

### Model Export

Export trained models for deployment:

```bash
python scripts/himloco_rsl_rl/play.py --task Unitree-Go2-Velocity-Play 
```

The script generates files in the log directory, including:
- `policy.pt` - TorchScript JIT model (single file deployment)
- `encoder.onnx` & `policy.onnx` - ONNX format (separate models)

## Deployment Guide

After model training, you need to verify the trained policy in Mujoco (Sim2Sim) to test model performance.
Only then can you proceed to real robot deployment (Sim2Real).

### Environment Setup

```bash
# Install dependencies
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

# Install unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # Install to /usr/local directory
sudo make install
# Compile the robot_controller
cd himloco_lab/deploy/robots/go2 
mkdir build && cd build
cmake .. && make
```

### Sim2Sim

Install [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation).

- Set the `robot` at `/simulate/config.yaml` to go2
- Set `domain_id` to 0
- Set `enable_elastic_hand` to 1
- Set `use_joystck` to 1.

Launch Mujoco simulation environment:
```bash
cd unitree_mujoco/simulate/build
./unitree_mujoco
```

Launch the controller:

```bash
cd himloco_lab/deploy/robots/go2/build
./go2_ctrl
```

### Sim2Real

You can use this program to directly control the real robot, but make sure the robot's motion controller is disabled.

```bash
./go2_ctrl --network eth0 # eth0 is the network interface name.
```

## 📝 TODO List
- \[x\] deploy on real robot and mujoco
- \[x\] deploy on jetson
- \[ \] migrate to latest rsl_rl version

## 🔗 References

- [HimLoco](https://github.com/RoboLoco/HimLoco)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)
