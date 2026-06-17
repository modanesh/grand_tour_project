"""
Demo script for IsaacLab with ANYMAL robot featuring custom velocity controller configuration.

Usage Examples:
    # Default velocity controller
    python demo_isaaclab.py

    # Use custom velocity controller configuration
    python demo_isaaclab.py --custom-velocity-cfg

    # Use velocity presets
    python demo_isaaclab.py --velocity-preset aggressive
    python demo_isaaclab.py --velocity-preset conservative
    python demo_isaaclab.py --velocity-preset exploration
    python demo_isaaclab.py --velocity-preset precision
    python demo_isaaclab.py --velocity-preset high_speed
    python demo_isaaclab.py --velocity-preset forward_only_slow

    # Modify velocity parameters programmatically:
    # Add calls to update_velocity_config() before environment creation
    # Example: update_velocity_config(lin_vel_x_range=(-3.0, 3.0))

Available Velocity Presets:
    - "default": Balanced movement (2.0 m/s max speed)
    - "aggressive": Fast movements, quick changes (3.5 m/s max speed)
    - "conservative": Slow, careful movements (1.0 m/s max speed)
    - "exploration": Frequent direction changes, direct angular control (2.5 m/s max speed)
    - "precision": Very slow, precise movements (0.5 m/s max speed)
    - "high_speed": Maximum speed capabilities (4.0 m/s max speed)
    - "forward_only_slow": Forward-only movement, very slow (0.8 m/s max speed)

Custom Velocity Controller Parameters:
    - lin_vel_x_range: Linear velocity X range (m/s)
    - lin_vel_y_range: Linear velocity Y range (m/s)
    - ang_vel_z_range: Angular velocity Z range (rad/s)
    - heading_command: Use heading-based control (bool)
    - heading_control_stiffness: Heading control stiffness
    - resampling_time_range: Command resampling time range (s)
    - rel_standing_envs: Probability of standing environments
    - rel_heading_envs: Probability of heading-based control
"""

from isaaclab.app import AppLauncher
from argparse import ArgumentParser
import torch
import os
import cv2
import tqdm
import math
import yaml
import wandb

parser = ArgumentParser()
parser.add_argument("--enable_cameras", action="store_true", default=False)
parser.add_argument(
    "--env",
    type=str,
    default="flat",
    choices=["flat", "warehouse"],
    help="Environment type: flat or warehouse",
)
parser.add_argument(
    "--actuation-mode",
    type=str,
    default="SEA",
    choices=["SEA", "PD", "Implicit"],
    help="Actuator mode: SEA (default), PD, or Implicit",
)
parser.add_argument(
    "--exp",
    type=str,
    default=None,
    help="Path to YAML file containing experiment configs (e.g., experiments/data_size.yaml)",
)
args, _ = parser.parse_known_args()

# Load experiment config(s) if specified
EXP_CONFIGS = {}
if args.exp:
    exp_file = (
        args.exp
        if os.path.isabs(args.exp)
        else os.path.join(os.path.dirname(__file__), args.exp)
    )
    if os.path.exists(exp_file):
        with open(exp_file, "r") as f:
            EXP_CONFIGS = yaml.safe_load(f)
        print(f"Loaded {len(EXP_CONFIGS)} experiments from: {exp_file}")
        for exp_name, exp_cfg in EXP_CONFIGS.items():
            velocity_info = ""
            if "velocity_preset" in exp_cfg:
                velocity_info = f", velocity={exp_cfg['velocity_preset']}"
            elif "velocity_config" in exp_cfg:
                velocity_info = f", velocity=custom"
            simulation_steps = exp_cfg.get("num_simulation_steps", "default")
            inference_steps = exp_cfg.get("num_inference_steps", "default")
            print(
                f"  - {exp_name}: {exp_cfg.get('n_epochs', 'N/A')} epochs, "
                f"{len(exp_cfg.get('X_train_mission_list', []))} missions"
                f"{velocity_info}, sim_steps={simulation_steps}, infer_steps={inference_steps}"
            )

    else:
        print(f"Warning: Experiment config file not found: {exp_file}")

ENV_TYPE = args.env  # Save env choice before second parser
ACTUATION_MODE = args.actuation_mode  # Save actuation mode choice
ENABLE_CAMERAS = args.enable_cameras  # Save camera flag before second parser

app_launcher = AppLauncher(
    headless=True,
    enable_cameras=True,  # Main camera always enabled
)
simulation_app = app_launcher.app

import isaaclab
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm

import matplotlib.pyplot as plt
import numpy as np
import isaaclab.envs.mdp as mdp
from scipy.spatial.transform import Rotation

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import (
    AnymalDFlatEnvCfg,
)

# Conditionally import warehouse config
if ENV_TYPE == "warehouse":
    from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.warehouse_env_cfg import (
        AnymalDWarehouseEnvCfg,
    )
from isaaclab.actuators import (
    ImplicitActuatorCfg,
    ActuatorNetLSTMCfg,
    IdealPDActuatorCfg,
)
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import torch.nn as nn
import torch.optim as optim

# Import TransformerForDiffusion from diffusion_policy
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent / "diffusion_policy"))
from diffusion_policy.model.diffusion.transformer_for_diffusion import (
    TransformerForDiffusion,
)

print("isaaclab path:", isaaclab.__file__)
print("isaaclab version:", isaaclab.__version__)

import zarr

import argparse

# Import velocity presets from src.commands.presets
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from src.commands.presets import (
    VELOCITY_PRESETS,
    VELOCITY_CONFIG,
    apply_velocity_preset,
    list_velocity_presets,
    update_velocity_config,
)

# Parse remaining args (classifier, etc.)
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--classifier", type=str, default="linear_regression")
parser.add_argument(
    "--force-x-unit", action="store_true", help="Force velocity command to [1, 0, 0]"
)
parser.add_argument(
    "--reference-tracking-mode",
    action="store_true",
    help="Replay recorded actions from LEICA-2 mission",
)
parser.add_argument(
    "--custom-velocity-cfg",
    action="store_true",
    help="Use custom velocity controller configuration",
)
parser.add_argument(
    "--velocity-preset",
    type=str,
    choices=list(VELOCITY_PRESETS.keys()),
    help="Velocity preset to use (overrides --custom-velocity-cfg)",
)
args, _ = parser.parse_known_args()

CLASSIFIER = args.classifier
FORCE_X_UNIT = args.force_x_unit
REFERENCE_TRACKING_MODE = args.reference_tracking_mode
CUSTOM_VELOCITY_CFG = args.custom_velocity_cfg
VELOCITY_PRESET = args.velocity_preset

# Initialize wandb after all config variables are defined
if EXP_CONFIGS:
    # Use first experiment config for logging
    first_exp_name = list(EXP_CONFIGS.keys())[0]
    first_exp_cfg = EXP_CONFIGS[first_exp_name]
    wandb.init(
        project="diffuseloco-isaaclab",
        entity="laijoey100-the-university-of-hong-kong",
        config={
            "experiment_name": first_exp_name,
            **first_exp_cfg,
            "env_type": ENV_TYPE,
            "actuation_mode": ACTUATION_MODE,
            "classifier": CLASSIFIER,
            "enable_cameras": ENABLE_CAMERAS,
            "force_x_unit": FORCE_X_UNIT,
            "reference_tracking_mode": REFERENCE_TRACKING_MODE,
            "custom_velocity_cfg": CUSTOM_VELOCITY_CFG,
            "velocity_preset": VELOCITY_PRESET,
        },
        name=first_exp_name,
    )
else:
    # Initialize wandb without experiment config
    wandb.init(
        project="diffuseloco-isaaclab",
        entity="laijoey100-the-university-of-hong-kong",
        config={
            "env_type": ENV_TYPE,
            "actuation_mode": ACTUATION_MODE,
            "classifier": CLASSIFIER,
            "enable_cameras": ENABLE_CAMERAS,
            "force_x_unit": FORCE_X_UNIT,
            "reference_tracking_mode": REFERENCE_TRACKING_MODE,
            "custom_velocity_cfg": CUSTOM_VELOCITY_CFG,
            "velocity_preset": VELOCITY_PRESET,
        },
    )

# Controller configuration parameters
ACTION_SCALE = 1.0
ACTION_OFFSET = 0.0
ACTUATOR_STIFFNESS = 100.0
ACTUATOR_DAMPING = 6.0

# Velocity presets are applied per-experiment in the experiment loop
# This allows each experiment to have its own velocity configuration


def create_custom_commands_cfg():
    """Create velocity command config with current VELOCITY_CONFIG values.

    This is a factory function that reads VELOCITY_CONFIG at call time,
    ensuring velocity presets are applied correctly.
    """

    @configclass
    class CustomCommandsCfg:
        """Custom command specifications for the MDP with configurable velocity controller."""

        base_velocity = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=VELOCITY_CONFIG["resampling_time_range"],
            rel_standing_envs=VELOCITY_CONFIG["rel_standing_envs"],
            rel_heading_envs=VELOCITY_CONFIG["rel_heading_envs"],
            heading_command=VELOCITY_CONFIG["heading_command"],
            heading_control_stiffness=VELOCITY_CONFIG["heading_control_stiffness"],
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=VELOCITY_CONFIG["lin_vel_x_range"],
                lin_vel_y=VELOCITY_CONFIG["lin_vel_y_range"],
                ang_vel_z=VELOCITY_CONFIG["ang_vel_z_range"],
                heading=(-math.pi, math.pi),  # Full heading range
            ),
        )

    return CustomCommandsCfg()


"""
[INFO] Command Manager:  <CommandManager> contains 1 active terms.
+------------------------------------------------+
|              Active Command Terms              |
+-------+---------------+------------------------+
| Index | Name          |          Type          |
+-------+---------------+------------------------+
|   0   | base_velocity | UniformVelocityCommand |
+-------+---------------+------------------------+

[INFO] Recorder Manager:  <RecorderManager> contains 0 active terms.
+---------------------+
| Active Recorder Terms |
+-----------+---------+
|   Index   | Name    |
+-----------+---------+
+-----------+---------+

[INFO] Action Manager:  <ActionManager> contains 1 active terms.
+------------------------------------+
|  Active Action Terms (shape: 12)   |
+--------+-------------+-------------+
| Index  | Name        |   Dimension |
+--------+-------------+-------------+
|   0    | joint_pos   |          12 |
+--------+-------------+-------------+

[INFO] Observation Manager: <ObservationManager> contains 1 groups.
+---------------------------------------------------------+
| Active Observation Terms in Group: 'policy' (shape: (48,)) |
+-----------+---------------------------------+-----------+
|   Index   | Name                            |   Shape   |
+-----------+---------------------------------+-----------+
|     0     | base_lin_vel                    |    (3,)   |
|     1     | base_ang_vel                    |    (3,)   |
|     2     | projected_gravity               |    (3,)   |
|     3     | velocity_commands               |    (3,)   |
|     4     | joint_pos                       |   (12,)   |
|     5     | joint_vel                       |   (12,)   |
|     6     | actions                         |   (12,)   |
+-----------+---------------------------------+-----------+
"""


from src.dataloader import GrandTourDataloader
from src.utils.cv2_utils import add_main_camera_text, add_front_camera_text
from src.utils.model_utils import create_model, train_model, DiffusionTransformerPolicy
from src.utils.validation_utils import compute_rmse_validation
from src.envs.anymal_env_cfg import AnymalDFlatCameraEnvCfg
from lococheck import check_anymal_d_obs


def load_data_for_experiment(exp_config, reference_tracking_mode=False):
    """Load data based on experiment config or reference tracking mode."""
    if reference_tracking_mode:
        print("Loading LEICA-2 reference mission for tracking...")
        dataloader = GrandTourDataloader(mission_names=["LEICA-1"], frequency=50)
        reference_actions = dataloader.get_actions_isaac_lab_format(shift_by_one=True)
        reference_obs = dataloader.get_observations_isaac_lab_format()
        _ = check_anymal_d_obs(reference_obs)
        print("Reference actions shape:", reference_actions.shape)
        print("Reference observations shape:", reference_obs.shape)
        print("Will replay", len(reference_actions), "action steps from LEICA-2")
        initial_joint_pos = reference_obs[0][12:24]
        print("Initial joint positions from first observation:", initial_joint_pos)
        return None, None, reference_actions, reference_obs, initial_joint_pos
    else:
        if exp_config and "X_train_mission_list" in exp_config:
            mission_list = exp_config["X_train_mission_list"]
        else:
            pass
            # mission_list = ["ARC-1", "ARC-2", "ARC-3", "ARC-4", "ARC-5", "ARC-6", "ARC-7", "LEICA-1", "LEICA-2", "CON-1", "CON-2", "CON-3", "CON-4"]
        print(f"Loading data for missions: {mission_list}")
        dataloader = GrandTourDataloader(mission_names=mission_list, frequency=50)
        X_data = dataloader.get_observations_isaac_lab_format()
        Y_data = dataloader.get_actions_isaac_lab_format()
        print("X_data shape:", X_data.shape)
        print("Y_data shape:", Y_data.shape)
        return X_data, Y_data, None, None, None


# Load data based on mode (for non-experiment runs or reference tracking)
if REFERENCE_TRACKING_MODE:
    print("Loading LEICA-2 reference mission for tracking...")
    dataloader = GrandTourDataloader(mission_names=["LEICA-1"], frequency=50)
    reference_actions = dataloader.get_actions_isaac_lab_format(shift_by_one=True)
    reference_obs = dataloader.get_observations_isaac_lab_format()
    _ = check_anymal_d_obs(reference_obs)
    print("Reference actions shape:", reference_actions.shape)
    print("Reference observations shape:", reference_obs.shape)
    print("Will replay", len(reference_actions), "action steps from LEICA-2")
    initial_joint_pos = reference_obs[0][12:24]
    print("Initial joint positions from first observation:", initial_joint_pos)
    REFERENCE_INITIAL_JOINT_POS = initial_joint_pos
    X_data, Y_data = None, None
else:
    REFERENCE_INITIAL_JOINT_POS = None
    if not EXP_CONFIGS:
        mission_list = [
            #     "ARC-1",
            #     "ARC-2",
            #     "ARC-3",
            #     "ARC-4",
            #     "ARC-5",
            #     "ARC-6",
            #     "ARC-7",
            "LEICA-1",
            # "LEICA-2",
            # "CON-1",
            # "CON-2",
            # "CON-3",
            # "CON-4",
        ]
        dataloader = GrandTourDataloader(mission_names=mission_list, frequency=50)
        X_data = dataloader.get_observations_isaac_lab_format()
        Y_data = dataloader.get_actions_isaac_lab_format()
        print("X_data shape:", X_data.shape)
        print("Y_data shape:", Y_data.shape)


# Diffusion Transformer Policy using TransformerForDiffusion


# CLASSIFIER = "linear_regression"

# Initialize device first
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Number of simulation steps
# Recommended: 20-50 steps for real-time robot control
# 1200 steps is too high for real-time applications
num_simulation_steps = 1200

# Number of diffusion inference steps (only used for DDPM models)
# More steps = better quality but slower
num_inference_steps = 20


def run_simulation_and_log(
    model, exp_name, exp_config, training_stats=None, run_wandb=True
):
    """Run simulation loop, create video, and log to wandb.

    Args:
        model: The trained model to use for inference
        exp_name: Experiment name for logging
        exp_config: Experiment config dict
        run_wandb: Whether to log to wandb (default True)

    Returns:
        dict: Reward statistics from the run
    """
    global \
        env, \
        REFERENCE_TRACKING_MODE, \
        REFERENCE_INITIAL_JOINT_POS, \
        FORCE_X_UNIT, \
        CLASSIFIER
    global \
        ACTION_SCALE, \
        ACTION_OFFSET, \
        ENABLE_CAMERAS, \
        ACTUATION_MODE, \
        num_simulation_steps
    global device, reference_actions

    # Reset environment for new run
    cumulative_rewards = torch.zeros(env.num_envs, device=env.device)
    finalized_episode_rewards = []

    obs, info = env.reset()

    # Track robot positions and episode start steps for distance/duration calculation
    robot_positions = torch.zeros(env.num_envs, 3, device=env.device)  # x, y, z
    episode_start_steps = torch.zeros(
        env.num_envs, dtype=torch.int32, device=env.device
    )
    episode_distances = [
        [0.0] for _ in range(env.num_envs)
    ]  # Start with 0.0 for current episode
    episode_durations = [
        [] for _ in range(env.num_envs)
    ]  # Store duration for each completed episode per robot

    # Initialize wandb for this experiment
    if run_wandb and exp_config:
        wandb.init(
            project="diffuseloco-isaaclab",
            entity="laijoey100-the-university-of-hong-kong",
            config={
                "experiment_name": exp_name,
                **exp_config,
                "env_type": ENV_TYPE,
                "actuation_mode": ACTUATION_MODE,
                "classifier": CLASSIFIER,
                "enable_cameras": ENABLE_CAMERAS,
                "force_x_unit": FORCE_X_UNIT,
                "reference_tracking_mode": REFERENCE_TRACKING_MODE,
            },
            name=exp_name,
            reinit=True,
        )

    # Create experiment-specific video directory
    exp_video_dir = f"imgs/{exp_name}"
    os.makedirs(exp_video_dir, exist_ok=True)

    # Cache for action trajectory (for DDPM with horizon)
    action_trajectory_cache = None
    trajectory_step = 0

    # Simulation loop
    for i in tqdm.trange(0, num_simulation_steps, desc=f"Running {exp_name}"):
        actions = torch.zeros_like(env.action_manager.action)

        if REFERENCE_TRACKING_MODE:
            # Replay recorded actions from LEICA-2 mission
            if i < len(reference_actions):
                recorded_action = reference_actions[i]
                actions = torch.tensor(
                    recorded_action, device=env.device, dtype=torch.float32
                ).unsqueeze(0)
            else:
                recorded_action = reference_actions[-1]
                actions = torch.tensor(
                    recorded_action, device=env.device, dtype=torch.float32
                ).unsqueeze(0)

        elif CLASSIFIER == "linear_regression":
            obs_first_36_features = process_observation(
                obs["policy"], force_x_unit=FORCE_X_UNIT
            )
            actions_pred = model.predict(obs_first_36_features.cpu().numpy())
            actions_pred = torch.tensor(
                actions_pred, device=env.device, dtype=torch.float32
            )
            actions = actions_pred

        elif CLASSIFIER == "random_forest":
            obs_first_36_features = process_observation(
                obs["policy"], force_x_unit=FORCE_X_UNIT
            )
            actions_pred = model.predict(obs_first_36_features.cpu().numpy())
            actions_pred = torch.tensor(
                actions_pred, device=env.device, dtype=torch.float32
            )
            actions = actions_pred

        elif CLASSIFIER == "ddpm":
            obs_first_36_features = process_observation(
                obs["policy"], force_x_unit=FORCE_X_UNIT
            )
            # Use cached trajectory or sample new one every `horizon` steps
            if action_trajectory_cache is None or trajectory_step >= model.horizon:
                with torch.no_grad():
                    action_trajectory_cache = model.sample(
                        obs_first_36_features, num_inference_steps=10
                    )
                trajectory_step = 0

            # Extract current action from trajectory
            actions = action_trajectory_cache[:, trajectory_step, :]
            trajectory_step += 1

        else:
            # run diffuseloco model inference
            curr_obs = obs["policy"].clone().detach()
            curr_obs = curr_obs.to(device=device, dtype=torch.float32)
            curr_obs = curr_obs[:, :36]
            curr_action = model.predict(curr_obs.cpu().numpy())
            curr_grand_tour_pos = curr_action[0]

            actions[:, :] = torch.tensor(
                curr_grand_tour_pos, device=env.device, dtype=torch.float32
            )

        obs, rew, terminated, truncated, info = env.step(actions)
        cumulative_rewards += rew

        # Track robot positions for distance calculation
        # Extract base position from observation (assuming it's in obs["policy"])
        # IsaacLab typically provides base position in the state or from root physx body
        if hasattr(env, "scene") and hasattr(env.scene, "robot"):
            robot_root_state = env.scene["robot"].data.root_state_w
            current_pos = robot_root_state[:, :3]  # x, y, z for each env

            # Calculate distance moved since last step
            if i > 0:
                delta_pos = torch.norm(current_pos - robot_positions, dim=1)
                for env_idx in range(env.num_envs):
                    if episode_distances[env_idx]:
                        episode_distances[env_idx][-1] += delta_pos[env_idx].item()
                    else:
                        episode_distances[env_idx].append(delta_pos[env_idx].item())

            robot_positions = current_pos.clone()

        # Track finalized episodes
        if terminated.any() or truncated.any():
            for env_idx in range(env.num_envs):
                if terminated[env_idx] or truncated[env_idx]:
                    # Save episode stats
                    finalized_episode_rewards.append(cumulative_rewards[env_idx].item())

                    # Calculate episode duration
                    episode_duration = i - episode_start_steps[env_idx].item()
                    episode_durations[env_idx].append(episode_duration)

                    # Reset tracking for next episode - start new distance accumulator
                    cumulative_rewards[env_idx] = 0
                    episode_start_steps[env_idx] = i
                    episode_distances[env_idx].append(
                        0.0
                    )  # Start tracking distance for new episode

        # Save camera frames
        if i % 1 == 0:
            rgb_main = env.scene["tiled_camera"].data.output["rgb"]
            img_main = rgb_main[0].cpu().numpy()
            img_main = cv2.cvtColor(img_main, cv2.COLOR_RGB2BGR)

            add_main_camera_text(
                img_main,
                step=i,
                cumulative_reward=cumulative_rewards.mean().item(),
                actuation_mode=ACTUATION_MODE,
                policy=CLASSIFIER,
            )
            cv2.imwrite(f"{exp_video_dir}/frame_{i}.png", img_main)

            if ENABLE_CAMERAS:
                rgb_front = env.scene["front_camera"].data.output["rgb"]
                img_front = rgb_front[0].cpu().numpy()
                img_front = cv2.cvtColor(img_front, cv2.COLOR_RGB2BGR)
                add_front_camera_text(
                    img_front,
                    step=i,
                    actuation_mode=ACTUATION_MODE,
                    policy=CLASSIFIER,
                )
                cv2.imwrite(f"{exp_video_dir}/front_{i}.png", img_front)

    # Create video
    video_path = f"{exp_name}_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    for i in range(num_simulation_steps):
        img = cv2.imread(f"{exp_video_dir}/frame_{i}.png")
        if img is not None:
            out.write(img)
    out.release()
    print(f"Saved: {video_path}")

    # Calculate reward stats for currently active robots (accumulated but not yet terminated)
    reward_stats = {
        "rewards/active_robots_cumulative_mean": cumulative_rewards.mean().item(),
        "rewards/active_robots_cumulative_std": cumulative_rewards.std().item(),
        "rewards/active_robots_cumulative_min": cumulative_rewards.min().item(),
        "rewards/active_robots_cumulative_max": cumulative_rewards.max().item(),
    }

    # Add stats for completed episodes (robots that have terminated)
    if finalized_episode_rewards:
        reward_stats.update(
            {
                "rewards/completed_episodes_mean": np.mean(finalized_episode_rewards),
                "rewards/completed_episodes_std": np.std(finalized_episode_rewards),
                "rewards/completed_episodes_min": np.min(finalized_episode_rewards),
                "rewards/completed_episodes_max": np.max(finalized_episode_rewards),
                "rewards/completed_episodes_count": len(finalized_episode_rewards),
            }
        )

    # Calculate distance traveled stats (total distance per completed episode, all robots)
    all_distances = [
        d for robot_distances in episode_distances for d in robot_distances
    ]
    if all_distances:
        reward_stats.update(
            {
                "distance_per_episode/total_meters_mean": np.mean(all_distances),
                "distance_per_episode/total_meters_std": np.std(all_distances),
                "distance_per_episode/total_meters_min": np.min(all_distances),
                "distance_per_episode/total_meters_max": np.max(all_distances),
            }
        )

    # Calculate episode duration stats (simulation steps per completed episode)
    all_durations = [
        d for robot_durations in episode_durations for d in robot_durations
    ]
    if all_durations:
        reward_stats.update(
            {
                "episode_length/steps_mean": np.mean(all_durations),
                "episode_length/steps_std": np.std(all_durations),
                "episode_length/steps_min": np.min(all_durations),
                "episode_length/steps_max": np.max(all_durations),
            }
        )

    # Add training stats if provided
    if training_stats:
        reward_stats.update(training_stats)

    # Log to wandb
    if run_wandb and wandb.run is not None:
        wandb.log(reward_stats)
        if os.path.exists(video_path):
            wandb.log({"demo_video": wandb.Video(video_path, fps=30, format="mp4")})
        wandb.finish()

    print(f"\n=== {exp_name} Results ===")
    print(
        f"Active robots - Mean: {cumulative_rewards.mean().item():.2f}, Std: {cumulative_rewards.std().item():.2f}, Min: {cumulative_rewards.min().item():.2f}, Max: {cumulative_rewards.max().item():.2f}"
    )
    if finalized_episode_rewards:
        print(
            f"Finalized episodes - Mean: {np.mean(finalized_episode_rewards):.2f}, Std: {np.std(finalized_episode_rewards):.2f}, Min: {np.min(finalized_episode_rewards):.2f}, Max: {np.max(finalized_episode_rewards):.2f}, Count: {len(finalized_episode_rewards)}"
        )

    return reward_stats


# @configclass overrides the base class
# Select base config class based on --env flag
if ENV_TYPE == "warehouse":
    BaseEnvCfg = AnymalDWarehouseEnvCfg
    print("Using warehouse environment")
else:
    BaseEnvCfg = AnymalDFlatEnvCfg
    print("Using flat environment")


def create_environment():
    """Create environment with current VELOCITY_CONFIG values."""
    global env, env_cfg
    try:
        if CUSTOM_VELOCITY_CFG:
            print("Using custom velocity controller configuration")
            print(f"  VELOCITY_CONFIG: {VELOCITY_CONFIG}")
            env_cfg_creator = AnymalDFlatCameraEnvCfg(
                env_type=ENV_TYPE,
                enable_cameras=ENABLE_CAMERAS,
                actuation_mode=ACTUATION_MODE,
                action_scale=ACTION_SCALE,
                action_offset=ACTION_OFFSET,
                actuator_stiffness=ACTUATOR_STIFFNESS,
                actuator_damping=ACTUATOR_DAMPING,
                custom_commands_cfg=create_custom_commands_cfg(),
            )
        else:
            print("Using default velocity controller configuration")
            env_cfg_creator = AnymalDFlatCameraEnvCfg(
                env_type=ENV_TYPE,
                enable_cameras=ENABLE_CAMERAS,
                actuation_mode=ACTUATION_MODE,
                action_scale=ACTION_SCALE,
                action_offset=ACTION_OFFSET,
                actuator_stiffness=ACTUATOR_STIFFNESS,
                actuator_damping=ACTUATOR_DAMPING,
            )
        print("  Getting config...")
        env_cfg = env_cfg_creator.get_cfg()
        # 30 Hz control policy (to match with DiffuseLoco paper)
        env_cfg.sim.dt = 1 / 300  # 300 Hz physics
        env_cfg.decimation = 10  # 30 Hz control
        env_cfg.scene.num_envs = 64
        print("  Creating ManagerBasedRLEnv...")
        env = ManagerBasedRLEnv(cfg=env_cfg)
        print("  Environment created successfully!")
        return env
    except Exception as e:
        print(f"ERROR creating environment: {e}")
        import traceback

        traceback.print_exc()
        raise


# Environment will be created lazily for the first experiment
env = None
env_cfg = None

# Create imgs directory if it doesn't exist
os.makedirs("imgs", exist_ok=True)


# Helper function to process observations
def process_observation(obs_full, force_x_unit=False):
    """Extract and optionally modify observation.

    Args:
        obs_full: (batch, 48) full observation
        force_x_unit: if True, override velocity_commands to [1, 0, 0]

    Returns:
        obs_processed: (batch, 36) processed observation
    """
    obs_processed = obs_full[:, :36].clone()

    if force_x_unit:
        # velocity_commands are at indices 9:12
        # Set to [1, 0, 0] for X-direction movement
        obs_processed[:, 9:12] = torch.tensor(
            [1.0, 0.0, 0.0], device=obs_processed.device, dtype=obs_processed.dtype
        )

    return obs_processed


# Run experiments or single training
experiments_completed = False
if EXP_CONFIGS and not REFERENCE_TRACKING_MODE:
    trained_models = {}
    for exp_name, exp_config in EXP_CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"Running experiment: {exp_name}")
        print(f"{'=' * 60}")

        # Apply experiment-specific velocity config if specified
        velocity_config_changed = False
        if "velocity_preset" in exp_config:
            preset_name = exp_config["velocity_preset"]
            if preset_name in VELOCITY_PRESETS:
                apply_velocity_preset(preset_name)
                CUSTOM_VELOCITY_CFG = True
                velocity_config_changed = True
                print(f"  Applied velocity preset: {preset_name}")
            else:
                print(f"  Warning: Unknown velocity preset '{preset_name}'")
        elif "velocity_config" in exp_config:
            custom_cfg = exp_config["velocity_config"]
            update_velocity_config(**custom_cfg)
            CUSTOM_VELOCITY_CFG = True
            velocity_config_changed = True
            print(f"  Applied custom velocity config")

        # Create environment on first experiment with current velocity config
        if env is None:
            print(
                f"  Creating environment for first experiment with velocity preset..."
            )
            env = create_environment()
        elif velocity_config_changed:
            print(f"  WARNING: Velocity preset changed but env already created.")
            print(
                f"  Subsequent experiments will use the first experiment's velocity preset."
            )

        # Apply experiment-specific config for simulation and inference steps
        exp_num_simulation_steps = exp_config.get(
            "num_simulation_steps", num_simulation_steps
        )
        if exp_num_simulation_steps != num_simulation_steps:
            print(f"  Using num_simulation_steps: {exp_num_simulation_steps}")
        exp_num_inference_steps = exp_config.get(
            "num_inference_steps", num_inference_steps
        )
        if exp_num_inference_steps != num_inference_steps:
            print(f"  Using num_inference_steps: {exp_num_inference_steps}")

        # Load data for this experiment
        X_data, Y_data, _, _, _ = load_data_for_experiment(
            exp_config, reference_tracking_mode=False
        )

        # Create model with experiment-specific config
        print(f"\nCreating {CLASSIFIER} model for {exp_name}...")
        model = create_model(CLASSIFIER, exp_config, device)

        # Train model
        model, optimizer, criterion, training_stats = train_model(
            X_data, Y_data, model, exp_config, exp_name, CLASSIFIER, device
        )

        # Compute train RMSE after training
        if CLASSIFIER == "linear_regression":
            train_rmse = np.sqrt(np.mean((model.predict(X_data[:]) - Y_data[:]) ** 2))
        elif CLASSIFIER == "random_forest":
            train_rmse = np.sqrt(np.mean((model.predict(X_data[:]) - Y_data[:]) ** 2))
            print(f"Train RMSE: {train_rmse:.6f}")
        elif CLASSIFIER == "ddpm":
            model.eval()
            with torch.no_grad():
                X_tensor_full = torch.FloatTensor(X_data).to(device)
                Y_tensor_full = torch.FloatTensor(Y_data).to(device)
                train_pred_traj = model.sample(X_tensor_full, num_inference_steps=20)
                train_rmse = torch.sqrt(
                    torch.mean((train_pred_traj[:, 0, :] - Y_tensor_full) ** 2)
                ).item()
            print(f"Train RMSE: {train_rmse:.6f}")
        else:
            train_rmse = None

        # Run RMSE validation on held-out missions
        val_stats = compute_rmse_validation(
            model,
            CLASSIFIER,
            device,
            split_json_path="data/split.json",
            frequency=50,
            train_rmse=train_rmse,
        )
        training_stats.update(val_stats)

        trained_models[exp_name] = model

        # Run simulation and log to wandb
        print(f"\nRunning simulation for {exp_name}...")
        # Temporarily override steps for this experiment
        original_num_simulation_steps = num_simulation_steps
        original_num_inference_steps = num_inference_steps
        if "num_simulation_steps" in exp_config:
            num_simulation_steps = exp_config["num_simulation_steps"]
        if "num_inference_steps" in exp_config:
            num_inference_steps = exp_config["num_inference_steps"]
        run_simulation_and_log(
            model, exp_name, exp_config, training_stats=training_stats, run_wandb=True
        )
        # Restore original steps
        num_simulation_steps = original_num_simulation_steps
        num_inference_steps = original_num_inference_steps

    print(f"\n{'=' * 60}")
    print(f"All {len(EXP_CONFIGS)} experiments completed!")
    print(f"{'=' * 60}")

    # Use the last trained model for final simulation (if needed)
    model = trained_models[list(trained_models.keys())[-1]]
    optimizer = None
    criterion = None
    experiments_completed = True

elif REFERENCE_TRACKING_MODE:
    model = None
    optimizer = None
    criterion = None
    print("Reference tracking mode - no model needed")

elif CLASSIFIER == "linear_regression":
    model = LinearRegression()
    model.fit(X_data[:], Y_data[:])
    train_rmse = np.sqrt(np.mean((model.predict(X_data[:]) - Y_data[:]) ** 2))
    print("Train RMSE: ", train_rmse)
    optimizer = None
    criterion = None

    # Run RMSE validation on held-out missions
    val_stats = compute_rmse_validation(
        model,
        CLASSIFIER,
        device,
        split_json_path="data/split.json",
        frequency=50,
        train_rmse=train_rmse,
    )

    # Log validation stats to wandb
    if wandb.run is not None:
        wandb.log(val_stats)

elif CLASSIFIER == "ddpm" and not experiments_completed:
    model = DiffusionTransformerPolicy(
        obs_dim=36,
        action_dim=12,
        horizon=8,
        n_layer=6,
        n_head=8,
        n_emb=256,
        num_timesteps=100,
        n_obs_steps=1,
        causal_attn=True,
    ).to(device)
    print("Using DiffusionTransformerPolicy with TransformerForDiffusion architecture")
    optimizer = model.get_optimizer(learning_rate=1e-4, weight_decay=1e-3)
    criterion = nn.MSELoss()
    print(f"Starting {CLASSIFIER} training...")
    num_epochs = 20
    model.train()

    X_tensor = torch.FloatTensor(X_data).to(device)
    Y_tensor = torch.FloatTensor(Y_data).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_obs, batch_action in train_loader:
            batch_size = batch_obs.shape[0]
            batch_obs = batch_obs.to(device)
            batch_action = batch_action.to(device)
            action_traj = batch_action.unsqueeze(1).expand(-1, model.horizon, -1)
            t = torch.randint(0, model.num_timesteps, (batch_size,), device=device)
            noise = torch.randn_like(action_traj)
            noisy_action_traj = model.q_sample(action_traj, t, noise)
            pred_noise = model(batch_obs, noisy_action_traj, t)
            loss = criterion(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

    print(f"{CLASSIFIER} training completed!")
    model.eval()

    # Compute train RMSE on full dataset
    with torch.no_grad():
        X_tensor_full = torch.FloatTensor(X_data).to(device)
        Y_tensor_full = torch.FloatTensor(Y_data).to(device)
        train_action_traj = Y_tensor_full.unsqueeze(1).expand(-1, model.horizon, -1)
        train_pred_traj = model.sample(X_tensor_full, num_inference_steps=20)
        train_rmse = torch.sqrt(
            torch.mean((train_pred_traj[:, 0, :] - Y_tensor_full) ** 2)
        ).item()
        print(f"Train RMSE: {train_rmse:.6f}")

    with torch.no_grad():
        test_obs = torch.FloatTensor(X_data[:5]).to(device)
        pred_joints = model.sample(test_obs, num_inference_steps=25)
        print("DDPM sample predictions:", pred_joints.cpu().numpy())

    # Run RMSE validation on held-out missions
    val_stats = compute_rmse_validation(
        model,
        CLASSIFIER,
        device,
        split_json_path="data/split.json",
        frequency=50,
        train_rmse=train_rmse,
    )

    # Log validation stats to wandb
    if wandb.run is not None:
        wandb.log(val_stats)

    model_name = f"{CLASSIFIER}_model.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as '{model_name}'")

elif not experiments_completed:
    model = DiffuseLocoModel().to(device)
    optimizer = None
    criterion = None

    # Run RMSE validation on held-out missions
    val_stats = compute_rmse_validation(
        model, CLASSIFIER, device, split_json_path="data/split.json", frequency=50
    )

    # Log validation stats to wandb
    if wandb.run is not None:
        wandb.log(val_stats)

else:
    # Model already set from experiments
    pass


# Cache for action trajectory (for DDPM with horizon)
action_trajectory_cache = None
trajectory_step = 0

# actions = torch.zeros_like(env.action_manager.action)
# obs, rew, terminated, truncated, info = env.step(actions)


def get_progress_desc(cumulative_rewards, finalized_episode_rewards):
    """Generate progress bar description with mean, min, max, and stdev of rewards."""
    mean_reward = cumulative_rewards.mean().item()
    min_reward = cumulative_rewards.min().item()
    max_reward = cumulative_rewards.max().item()
    current_stdev = cumulative_rewards.std().item()

    desc = f"Rewards: μ={mean_reward:.1f} min={min_reward:.1f} max={max_reward:.1f} σ={current_stdev:.1f}"

    if len(finalized_episode_rewards) > 1:
        finalized_stdev = np.std(finalized_episode_rewards)
        desc += (
            f" | Finalized σ={finalized_stdev:.1f} (n={len(finalized_episode_rewards)})"
        )
    elif len(finalized_episode_rewards) == 1:
        desc += f" | Finalized (n=1)"

    return desc


# If running without experiments, do single simulation run
if not experiments_completed and not REFERENCE_TRACKING_MODE:
    print("\nRunning single simulation (no experiments)...")
    run_simulation_and_log(model, "single_run", None, run_wandb=True)


# print(f"sample joint position: {grand_tour_joint_positions[10000]}")

# # Print the indices the Action Manager is controlling
# print("Action Joint Indices:", env.action_manager._action_terms["joint_pos"].joint_ids)


# env.export_IO_descriptors(output_dir="./io_descriptors_output")


# io_descriptors = env.get_IO_descriptors()
# print(io_descriptors)

env.close()
simulation_app.close()
