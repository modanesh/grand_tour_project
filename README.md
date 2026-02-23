# COMP 400 Project

<div align="center">

**Offline Learning for Quadruped Locomotion using the Grand Tour Dataset**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)

</div>

---

## Overview

This project implements **offline learning algorithms** for training quadruped locomotion policies on the [Grand Tour Dataset](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset) — a large-scale real-world dataset of ANYmal D robot trajectories. The trained policies are evaluated in the **Isaac Gym** physics simulator.

### Key Features

- 🤖 **Multiple Offline Learning Algorithms**: DiffuseLoco, Vanilla BC, IQL, CQL, EDAC
- 🔄 **End-to-end Pipeline**: From raw sensor data to trained policies
- 📊 **Isaac Gym Integration**: Online evaluation in high-fidelity simulation
- 📈 **Weights & Biases Logging**: Comprehensive experiment tracking

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Grand Tour Dataset                               │
│                    (Real-world ANYmal D trajectories)                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                                     │
│  ┌──────────────┐    ┌──────────────────────────┐                      │
│  │   download   │───▶│build_dataset_full_pipline│                      │
│  │    .py       │    │        .py               │                      │
│  └──────────────┘    └──────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Offline Dataset (HDF5)                              │
│         observations, actions, rewards, terminals, next_obs              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Training + Online Evaluation                          │
│         (BC/IQL/CQL/EDAC/Diffusion) → Isaac Gym (ANYmal D Flat)         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.8+
- Isaac Gym Preview 4
- Hugging Face account (for dataset access)

### Setup

1. **Create project dir and clone the repository into project dir**
   ```bash
   mkdir project
   cd project
   git clone https://github.com/your-username/grand_tour_project.git .
   cd grand_tour_project
   ```

2. **Create virtual environment with python 3.8 and activate venv**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install Isaac Gym**
   ```bash
   # Download Isaac Gym Preview 4 from NVIDIA
   cd /path/to/isaacgym/python
   pip install -e .
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install legged_gym (ANYmal D support)**
   ```bash
   cd project
   git clone https://github.com/modanesh/legged_gym_anymal_d.git . 
   cd legged_gym_anymal_d
   pip install -e .
   ```

6. **Configure Hugging Face token**
   ```bash
   # Create .env file
   echo "HF_TOKEN=your_huggingface_token" > .env
   ```

---

## Data Pipeline

### 1. Download Grand Tour Dataset

Download real-world ANYmal D trajectories from Hugging Face:

```bash
python download.py
```

This downloads the following sensor topics:
- `anymal_state_odometry` — Robot pose and velocity
- `anymal_state_state_estimator` — Joint positions, velocities, contact states
- `anymal_imu` — IMU measurements
- `anymal_state_actuator` — Actuator commands
- `anymal_command_twist` — Velocity commands

### 2. Align Sensor Data and Build Offline RL Dataset

Convert aligned data to offline RL format compatible with Isaac Gym:

```bash
python build_dataset_full_pipeline.py
```

Note that if the aligned_data.zarr has already been created, you can comment out the `align_data()` call and just load the aligned data when building the dataset to save time. If it's your first time running it, make sure `align_data()` is uncommented in the file. 

**Output**: `offline_dataset_pp.hdf5` with:
- `observations` — 36D state vectors (or 48D with previous actions)
- `actions` — 12D joint position targets
- `rewards` — Computed using Isaac Gym reward functions
- `terminals` — Episode boundaries
- `next_observations` — Next state

#### Isaac Gym Dataset Note: 
- The "Isaac Gym" dataset is saved as `expert_dataset.hdf5` 
- To build this dataset run `build_dataset.py` in modanesh/legged_gym repo on `rohan-gt-project` branch
- Once the dataset is created copy it to the codebase here
- If there are any issues when trying to train on the dataset, run `./fix_bad_expert_dataset.py` to fix the (N,1) -> (N,) issue if it arises.


### Observation Space (48 dimensions)

| Index | Description | Scale |
|-------|-------------|-------|
| 0-2 | Base linear velocity (body frame) | × 2.0 |
| 3-5 | Base angular velocity (body frame) | × 0.25 |
| 6-8 | Projected gravity | — |
| 9-11 | Commands [vx, vy, ω_yaw] | × [2.0, 2.0, 0.25] |
| 12-23 | Joint positions (offset from default) | × 1.0 |
| 24-35 | Joint velocities | × 0.05 |
| 36-47 | Prev Actions | — | 

### Action Space (12 dimensions)

Joint position targets for the 12 DOF:
```
LF_HAA, LF_HFE, LF_KFE, RF_HAA, RF_HFE, RF_KFE,
LH_HAA, LH_HFE, LH_KFE, RH_HAA, RH_HFE, RH_KFE
```

To change the sensors used or which observations are built (e.g., if you want to remove prev actions or not), modify `build_offline_dataset()` in `build_dataset.py`.

---

## Training

To use a specific learning algorithm it suffices to do

```bash
python <algorithm_file.py>
e.g.,
python diffuseloco.py
python cql.py
```
The configuration file for the DiffuseLoco training is in `diffusion_policy/diffusion_policy/config/anymal_diffusion_policy.yaml`.

The configuration for the rest of the learning algorithms are in the respective learning files.

Modifying these configs allows you to choose which dataset to use, max training steps, learning rate etc.

Note that for DiffuseLoco, depending on how many observation dimensions there are (36 with no prev actions vs 48 with), the correct observation dimension number should be put in the config file `anymal_diffusion_policy.yaml`

---

## Evaluation

All training scripts automatically evaluate offline on Grand Tour test missions every `rollout_every` epochs and log ATE (Average Tracking Error) and RTE (Relative Tracking Error) metrics.

### Mission-aware Train/Test Split

The project uses `anymal_diffusion_policy_mission_split.yaml` with a balanced mission-aware split:

**Test missions (6 total):**
- `spx-2` (2024-11-02-17-18-32)
- `snow-2` (2024-11-02-21-12-51)
- `eig-1` (2024-11-03-13-51-43)
- `con-4` (2024-12-09-11-28-28)
- `arc-2` (2024-11-18-13-22-14)
- `arc-7` (2024-11-18-17-31-36)

**Train missions (43 total):** All other missions from `mission_metadata.json`

This split ensures test missions are held out entirely, preventing data leakage.

To use this split:
1. Run `build_dataset_full_pipeline.py` to generate `mission_metadata.json`
2. Use `anymal_diffusion_policy_mission_split.yaml` config
3. Run `python diffuseloco.py` — training automatically evaluates on test missions

Logged metrics: `test_ATE`, `test_RTE`, `test_ATE/{mission}`, `test_RTE/{mission}`

---

## Project Structure

```
grand_tour_project/
├── download.py              # Download Grand Tour dataset from HuggingFace
├── align_data.py            # Align multi-sensor data to 50Hz
├── build_dataset.py         # Build offline RL dataset (HDF5)
├── build_dataset_full_pipeline.py  # End-to-end dataset building (calls align_data.py and build_dataset.py) 
│
├── vanilla_bc.py            # Vanilla Behavioral Cloning
├── iql.py                   # Implicit Q-Learning
├── cql.py                   # Conservative Q-Learning
├── edac.py                  # Ensemble Diversified Actor Critic
│
├── eval_isaac_v2.py         # Isaac Gym online evaluation
├── reward.py                # Reward function implementations
├── utils.py                 # Utility functions
│
├── isaac_compatibility.py   # Grand Tour → Isaac Gym data transforms
├── grandtour_compatibility.py # Isaac Gym → Grand Tour data transforms
│
├── diffusion_policy/        # Diffusion policy implementation
│   ├── config/              # Hydra configuration files
│   ├── dataset/             # Dataset loaders
│   ├── model/               # Network architectures
│   ├── policy/              # Policy implementations
│   └── workspace/           # Training workspaces
│
├── diffuseloco.py           # Diffusion policy training entry point
├── compare_dataset_*.py     # Dataset analysis scripts to compare datasets
│
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## Configuration


### Weights & Biases Integration

All training scripts log to W&B:

```python
project: "grand_tour"
```

View experiments at [wandb.ai](https://wandb.ai).

---

## Data Format Compatibility

### Observation Scaling

The project handles observation scaling between Grand Tour (raw) and Isaac Gym (scaled) formats:

| Component | Grand Tour | Isaac Gym Scale |
|-----------|------------|-----------------|
| Linear velocity | Raw | × 2.0 |
| Angular velocity | Raw | × 0.25 |
| Commands | Raw | × [2.0, 2.0, 0.25] |
| Joint positions | Absolute | Offset from default |
| Joint velocities | Raw | × 0.05 |

### Action Conversion

Actions are converted between formats using:
- **Grand Tour**: Absolute joint positions
- **Isaac Gym**: Normalized offsets from default positions

```python
# Grand Tour → Isaac Gym
action_isaac = (action_gt - default_dof_pos) / action_scale

# Isaac Gym → Grand Tour  
action_gt = action_isaac * action_scale + default_dof_pos
```

---

## Acknowledgments

- [Grand Tour Dataset](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset) by Leggedrobotics
- [Isaac Gym](https://developer.nvidia.com/isaac-gym) by NVIDIA
- [legged_gym_anymal_d](https://github.com/modanesh/legged_gym_anymal_d) for ANYmal D support
- [CORL](https://github.com/tinkoff-ai/CORL) for offline RL algorithm implementations
- [DiffuseLoco](https://github.com/HybridRobotics/DiffuseLoco) for transformer-based diffusion policies

---


