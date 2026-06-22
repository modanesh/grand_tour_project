# COMP 400 Project

### TODO

-[ ] Missing `RIV-1` in `./data`

---

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

All training scripts automatically evaluate in Isaac Gym every `eval_freq` steps and log the individual reward terms and the avg episode reward and length

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

## Migration from IsaacGym to IsaacLab

### One-Time Installation for Isaaclab

Using a conda environment is recommended for reproducible environments. 

```bash
conda create -n isaaclab-2-0-2 python=3.10
```

Then proceed to Install IsaacLab v2.0.2 (compatible with IsaacSim 4.5.x) following the [IsaacLab v2.0.2 instructions](https://isaac-sim.github.io/IsaacLab/v2.0.2/source/setup/installation/binaries_installation.html).

### Quickstart

Run the following to run `diffuseloco.py` on IsaacLab as an online eval environment.

```bash
# activate conda env
conda activate isaaclab-2-0-2 (or the env that you named)

# configure diffuse_policy lib path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/diffusion_policy"

# the config file is located at
# ./diffusion_policy/diffusion_policy/config/anymal_diffusion_policy_isaaclab.yaml
python diffuseloco.py --config-name=anymal_diffusion_policy_isaaclab
```

### IsaacLab RSL-RL Training with Overrides

A modified training script is available at `IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py` that supports environment overrides and automatic post-training data collection.

Run via the `.justfile` recipe:

```bash
just run_isaaclab_anymal_d_quickstart
```

This applies the following **Hydra overrides** to the standard `Isaac-Velocity-Flat-Anymal-D-v0` config:

| Override | Default | Override Value | Effect |
|----------|---------|----------------|--------|
| `++actions.joint_pos.scale` | `0.5` | `1.0` | Policy outputs are no longer halved |
| `++actions.joint_pos.use_default_offset` | `true` | `false` | Disables ANYmal-D default pose offset |
| `++actions.joint_pos.offset` | default pose | `0.0` | Action offset is zero (raw position targets) |
| `++observations.policy.enable_corruption` | `true` | `false` | Disables observation noise/corruption |
| `env_cfg.sim.dt` | `0.005` (200 Hz) | `0.002` (500 Hz) | Physics simulation frequency |
| `env_cfg.decimation` | `4` (50 Hz) | `10` (50 Hz) | Policy control frequency |

**Result:** The action pipeline becomes `processed_action = raw_policy_action * 1.0 + 0.0`, meaning the policy learns raw joint position targets with no scaling or offset, and receives clean (noise-free) observations at **50 Hz** with **500 Hz** physics.

At startup, `train.py` prints a **verification block** with runtime assertions:
```
[VERIFY] Action/Obs Configuration:
  Action scale   : 1.0
  Action offset  : 0.0
  Use default offset: False
  Obs corruption : False
  Empirical norm : False
  sim.dt         : 0.002
  decimation     : 10
  Physics freq   : 500 Hz
  Policy freq    : 50 Hz
[VERIFY] All assertions passed. scale=1.0, offset=0.0, physics=500Hz, policy=50Hz.
```
If any assertion fails, the script aborts with a clear error message so the override misconfiguration is caught immediately.

#### Post-Training Data Collection

After the 10 training iterations complete, the modified `train.py` automatically runs a **post-training rollout** using the trained policy and saves the data:

- **Collection:** 1,000 steps × 4,096 envs (all training envs)
- **Output directory:** `../generated_data/` (relative to `IsaacLab/`)
- **Files saved:**
  - `observations.csv` — Observations before each action
  - `next_observations.csv` — Observations after each action
  - `actions.csv` — Policy outputs (raw, before any env scaling)
  - `rewards.csv` — Step rewards
  - `dones.csv` — Episode termination flags (0/1)
  - `env_ids.csv` — Robot/environment ID for each row (0–4095)
  - `metadata.json` — Shapes, dimensions, and total transition count

These CSV files are flat `(steps × envs, dim)` matrices suitable for offline analysis. The robot uses the **LSTM actuator model** (`ActuatorNetLSTMCfg`) during simulation, so the logged actions are the **position targets** fed into the LSTM actuator network.

**Robot & Episode Separation:**

| File | Content | Shape |
|------|---------|-------|
| `observations.csv` / `actions.csv` / `rewards.csv` | Data from all robots interleaved | `(4096000, dim)` |
| `env_ids.csv` | Robot ID for each row (0–4095) | `(4096000, 1)` |
| `dones.csv` | Episode termination flag (0/1) | `(4096000, 1)` |

**Row Alignment:**

All CSV files share the **same row ordering**. Row `i` in `observations.csv` corresponds exactly to row `i` in `actions.csv`, `rewards.csv`, `next_observations.csv`, `dones.csv`, and `env_ids.csv`.

| Row `i` | `observations[i]` | `actions[i]` | `rewards[i]` | `next_observations[i]` | `dones[i]` | `env_ids[i]` |
|---------|-------------------|--------------|--------------|------------------------|------------|--------------|
| Meaning | State before action | Action taken | Reward received | State after action | Episode ended? | Which robot |

The flattening is **step-major**: step 0 for robots 0–4095, then step 1 for robots 0–4095, etc. `env_ids.csv` cycles `0,1,2,...,4095,0,1,...` confirming the alignment.

To reconstruct per-robot trajectories in Python:
```python
import numpy as np

obs = np.loadtxt("generated_data/observations.csv", delimiter=",")
actions = np.loadtxt("generated_data/actions.csv", delimiter=",")
env_ids = np.loadtxt("generated_data/env_ids.csv", delimiter=",", dtype=int)
dones = np.loadtxt("generated_data/dones.csv", delimiter=",", dtype=int)

for robot_id in range(4096):
    mask = env_ids == robot_id
    robot_obs = obs[mask]
    robot_actions = actions[mask]
    robot_dones = dones[mask]
    episode_ends = np.where(robot_dones == 1)[0]
    print(f"Robot {robot_id}: {len(robot_obs)} steps, episodes end at {episode_ends}")
```


---

## Justfile

Common tasks are available as `just` recipes in `.justfile`. Requires [just](https://github.com/casey/just) and a `.env` file with `LAURENCE_WANDB_API_KEY` and `WANDB_ENTITY` set.

### Prerequisites

```bash
# Install just (if not already installed)
cargo install just
# or: brew install just / winget install casey.just
```

### Running Inference

All inference recipes run `demo_serve.py` with a pre-trained checkpoint. Arguments default to `velocity_preset="forward_only_slow"` and `env="flat"` unless overridden.

```bash
# Basic usage
just <recipe> [velocity_preset] [controller_frequency] [env]

# Example: run 50Hz full-mix model with stationary preset on rough terrain
just run_inference_diffuseloco_50hz_full stationary 50 rough
```

| Recipe | Hz | Checkpoint |
|--------|----|------------|
| `run_inference_diffuseloco_30hz_full` | 30 | Full Grand Tour mix |
| `run_inference_diffuseloco_50hz_full` | 50 | Full Grand Tour mix |
| `run_inference_diffuseloco_20hz_full` | 20 | Full Grand Tour mix |
| `run_inference_diffuseloco_30hz_flat` | 30 | Flat terrain subsample |
| `run_inference_diffuseloco_30hz_rough` | 30 | Rough terrain subsample |
| `run_inference_diffuseloco_50hz_isaaclab_may_14` | 50 | IsaacLab offline (May 14) |
| `run_inference_diffuseloco_50hz_isaaclab_may_20` | 50 | IsaacLab offline (May 20) |
| `run_inference_diffuseloco_50hz_grandtour_upsampling` | 50 | Grand Tour upsampled |
| `run_inference_diffuseloco_50hz_grandtour_goal_condition` | 50 | Grand Tour goal-conditioned |
| `run_inference_diffuseloco_50hz_grandtour_offset_one` | 50 | Grand Tour offset=1 |
| `run_inference_diffuseloco_50hz_grandtour_offset_one_gravity_fix` | 50 | Offset=1 + gravity fix |
| `run_inference_diffuseloco_50hz_grandtour_offset_one_gravity_fix_v2` | 50 | Offset=1 + gravity fix v2 |
| `run_inference_diffuseloco_50hz_isaaclab_offset_one` | 50 | IsaacLab offset=1 |

### IsaacLab RSL-RL Training

```bash
just run_isaaclab_anymal_d_quickstart
```

Activates the `isaaclab-2-0-2` conda environment and runs the IsaacLab ANYmal-D flat training with unscaled actions and noise-free observations (see the [IsaacLab RSL-RL Training with Overrides](#isaaclab-rsl-rl-training-with-overrides) section above for the full list of overrides applied).

---

## Acknowledgments

- [Grand Tour Dataset](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset) by Leggedrobotics
- [Isaac Gym](https://developer.nvidia.com/isaac-gym) by NVIDIA
- [legged_gym_anymal_d](https://github.com/modanesh/legged_gym_anymal_d) for ANYmal D support
- [CORL](https://github.com/tinkoff-ai/CORL) for offline RL algorithm implementations
- [DiffuseLoco](https://github.com/HybridRobotics/DiffuseLoco) for transformer-based diffusion policies

---


