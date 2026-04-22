"""
Demo script for IsaacLab with ANYMAL robot running inference-only using a DiffuseLoco checkpoint.

No training is performed — the script loads a pretrained policy checkpoint and runs it
in the IsaacLab simulation loop.

Usage Examples:
    # Load a checkpoint and run inference
    python demo_serve.py --checkpoint <path_to_ckpt> [--device cuda:0]

    # Use custom velocity controller configuration
    python demo_serve.py --checkpoint <path_to_ckpt> --custom-velocity-cfg

    # Use velocity presets
    python demo_serve.py --checkpoint <path_to_ckpt> --velocity-preset aggressive

    # Force forward-only movement
    python demo_serve.py --checkpoint <path_to_ckpt> --force-x-unit

    # Enable front camera
    python demo_serve.py --checkpoint <path_to_ckpt> --enable_cameras
"""

from isaaclab.app import AppLauncher
from argparse import ArgumentParser
import torch
import os
import cv2
import tqdm
import math
import numpy as np
import argparse
import wandb

parser = argparse.ArgumentParser()
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
parser.add_argument("--checkpoint", type=str, required=True, help="Path to DiffuseLoco .ckpt file")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
args, _ = parser.parse_known_args()

ENV_TYPE = args.env
ACTUATION_MODE = args.actuation_mode
ENABLE_CAMERAS = args.enable_cameras
CHECKPOINT = args.checkpoint
DEVICE = args.device

app_launcher = AppLauncher(
    headless=True,
    enable_cameras=True,
)
simulation_app = app_launcher.app

import isaaclab
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import (
    AnymalDFlatEnvCfg,
)

if ENV_TYPE == "warehouse":
    from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.warehouse_env_cfg import (
        AnymalDWarehouseEnvCfg,
    )
from isaaclab.envs.mdp import UniformVelocityCommandCfg

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

# Parse remaining args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--force-x-unit", action="store_true", help="Force velocity command to [1, 0, 0]"
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

FORCE_X_UNIT = args.force_x_unit
CUSTOM_VELOCITY_CFG = True  # always use custom velocity cfg so commands are nonzero
VELOCITY_PRESET = args.velocity_preset

# Observation space configuration
OBSERVATION_SCALE = 1.0
OBSERVATION_OFFSET = 0.0

# Controller configuration parameters
ACTION_SCALE = 1.0
ACTION_OFFSET = 0.0
ACTUATOR_STIFFNESS = 100.0
ACTUATOR_DAMPING = 6.0

# Apply velocity preset if specified
if VELOCITY_PRESET and VELOCITY_PRESET in VELOCITY_PRESETS:
    apply_velocity_preset(VELOCITY_PRESET)
    CUSTOM_VELOCITY_CFG = True
    print(f"Applied velocity preset: {VELOCITY_PRESET}")


# ---------- Load DiffuseLoco policy from checkpoint (serve.py style) ----------

import dill
import hydra
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

payload = torch.load(open(CHECKPOINT, "rb"), pickle_module=dill)
cfg = payload["cfg"]
OmegaConf.set_struct(cfg, False)

cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg, output_dir=str(pathlib.Path(CHECKPOINT).parent))
workspace.load_payload(payload, exclude_keys=None, include_keys=None)

policy = workspace.model
if cfg.training.use_ema:
    policy = workspace.ema_model

policy.to(torch.device(DEVICE))
policy.eval()

print(f"Loaded policy from {CHECKPOINT}")
print(f"  obs_dim={policy.obs_dim}, action_dim={policy.action_dim}")
print(f"  n_obs_steps={policy.n_obs_steps}, n_action_steps={policy.n_action_steps}")

# Number of simulation steps
num_simulation_steps = 1200

# Number of diffusion inference steps
num_inference_steps = 20

# Initialize wandb
wandb.init(
    project="diffuseloco-isaaclab",
    entity="laijoey100-the-university-of-hong-kong",
    config={
        "checkpoint": CHECKPOINT,
        "env_type": ENV_TYPE,
        "actuation_mode": ACTUATION_MODE,
        "enable_cameras": ENABLE_CAMERAS,
        "force_x_unit": FORCE_X_UNIT,
        "custom_velocity_cfg": CUSTOM_VELOCITY_CFG,
        "velocity_preset": VELOCITY_PRESET,
        "num_simulation_steps": num_simulation_steps,
        "num_inference_steps": num_inference_steps,
    },
    name="inference",
)

# Observation history buffer for n_obs_steps > 1
obs_history = []


def create_custom_commands_cfg():
    """Create velocity command config with current VELOCITY_CONFIG values."""

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
                heading=(-math.pi, math.pi),
            ),
        )

    return CustomCommandsCfg()


from src.utils.cv2_utils import add_main_camera_text, add_front_camera_text
from src.envs.anymal_env_cfg import AnymalDFlatCameraEnvCfg


def run_simulation():
    """Run simulation loop using the loaded policy for inference."""
    global env, obs_history

    cumulative_rewards = torch.zeros(env.num_envs, device=env.device)
    finalized_episode_rewards = []

    obs, info = env.reset()

    # Track robot positions and episode stats
    robot_positions = torch.zeros(env.num_envs, 3, device=env.device)
    episode_start_steps = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    episode_distances = [[0.0] for _ in range(env.num_envs)]
    episode_durations = [[] for _ in range(env.num_envs)]

    # Video directory
    video_dir = "imgs/inference"
    os.makedirs(video_dir, exist_ok=True)

    # Reset observation history
    obs_history.clear()

    global_step = 0
    for i in tqdm.trange(0, num_simulation_steps, policy.n_action_steps, desc="Running inference"):
        # Process observation (once per planning call)
        curr_obs = obs["policy"].clone().detach()
        curr_obs = curr_obs.to(device=env.device, dtype=torch.float32)
        curr_obs = curr_obs * OBSERVATION_SCALE + OBSERVATION_OFFSET

        # Force x-unit velocity command if requested
        if FORCE_X_UNIT:
            curr_obs[:, 9:12] = torch.tensor([1.0, 0.0, 0.0], device=curr_obs.device, dtype=curr_obs.dtype)

        # Use only first 36 features (obs_dim)
        curr_obs = curr_obs[:, :policy.obs_dim]

        # Build observation history for the policy
        # policy expects (batch, n_obs_steps, obs_dim)
        obs_history.append(curr_obs.cpu().numpy())
        if len(obs_history) > policy.n_obs_steps:
            obs_history.pop(0)

        # Pad history if needed at start
        while len(obs_history) < policy.n_obs_steps:
            obs_history.insert(0, curr_obs.cpu().numpy())

        # Stack history: (n_obs_steps, batch, obs_dim) -> (batch, n_obs_steps, obs_dim)
        obs_seq = np.stack(obs_history, axis=0)  # (n_obs_steps, batch, obs_dim)
        obs_seq = np.transpose(obs_seq, (1, 0, 2))  # (batch, n_obs_steps, obs_dim)
        obs_tensor = torch.from_numpy(obs_seq).to(torch.device(DEVICE))

        # Run policy inference (once per planning call)
        with torch.no_grad():
            result = policy.predict_action({"obs": obs_tensor})

        # Extract actions: (batch, n_action_steps, action_dim)
        actions_pred = result["action"]

        # Execute all n_action_steps before re-planning
        for step_i in range(policy.n_action_steps):
            actions = actions_pred[:, step_i, :].to(env.device, dtype=torch.float32)

            obs, rew, terminated, truncated, info = env.step(actions)
            cumulative_rewards += rew

            # Track robot positions for distance calculation
            if hasattr(env, "scene") and hasattr(env.scene, "robot"):
                robot_root_state = env.scene["robot"].data.root_state_w
                current_pos = robot_root_state[:, :3]

                if global_step > 0:
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
                        finalized_episode_rewards.append(cumulative_rewards[env_idx].item())
                        episode_duration = global_step - episode_start_steps[env_idx].item()
                        episode_durations[env_idx].append(episode_duration)
                        cumulative_rewards[env_idx] = 0
                        episode_start_steps[env_idx] = global_step
                        episode_distances[env_idx].append(0.0)

            # Save camera frames
            rgb_main = env.scene["tiled_camera"].data.output["rgb"]
            img_main = rgb_main[0].cpu().numpy()
            img_main = cv2.cvtColor(img_main, cv2.COLOR_RGB2BGR)

            add_main_camera_text(
                img_main,
                step=global_step,
                cumulative_reward=cumulative_rewards.mean().item(),
                actuation_mode=ACTUATION_MODE,
                policy="diffuseloco",
                exp_name="inference",
            )
            cv2.imwrite(f"{video_dir}/frame_{global_step}.png", img_main)

            if ENABLE_CAMERAS:
                rgb_front = env.scene["front_camera"].data.output["rgb"]
                img_front = rgb_front[0].cpu().numpy()
                img_front = cv2.cvtColor(img_front, cv2.COLOR_RGB2BGR)
                add_front_camera_text(
                    img_front,
                    step=global_step,
                    actuation_mode=ACTUATION_MODE,
                    policy="diffuseloco",
                    exp_name="inference",
                )
                cv2.imwrite(f"{video_dir}/front_{global_step}.png", img_front)

            global_step += 1

    # Create video
    video_path = "inference_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    for i in range(global_step):
        img = cv2.imread(f"{video_dir}/frame_{i}.png")
        if img is not None:
            out.write(img)
    out.release()
    print(f"Saved: {video_path}")

    # Reward stats
    reward_stats = {
        "rewards/active_robots_cumulative_mean": cumulative_rewards.mean().item(),
        "rewards/active_robots_cumulative_std": cumulative_rewards.std().item(),
        "rewards/active_robots_cumulative_min": cumulative_rewards.min().item(),
        "rewards/active_robots_cumulative_max": cumulative_rewards.max().item(),
    }

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

    all_distances = [d for robot_distances in episode_distances for d in robot_distances]
    if all_distances:
        reward_stats.update(
            {
                "distance_per_episode/total_meters_mean": np.mean(all_distances),
                "distance_per_episode/total_meters_std": np.std(all_distances),
                "distance_per_episode/total_meters_min": np.min(all_distances),
                "distance_per_episode/total_meters_max": np.max(all_distances),
            }
        )

    all_durations = [d for robot_durations in episode_durations for d in robot_durations]
    if all_durations:
        reward_stats.update(
            {
                "episode_length/steps_mean": np.mean(all_durations),
                "episode_length/steps_std": np.std(all_durations),
                "episode_length/steps_min": np.min(all_durations),
                "episode_length/steps_max": np.max(all_durations),
            }
        )

    print("\n=== Inference Results ===")
    print(
        f"Active robots - Mean: {cumulative_rewards.mean().item():.2f}, Std: {cumulative_rewards.std().item():.2f}, Min: {cumulative_rewards.min().item():.2f}, Max: {cumulative_rewards.max().item():.2f}"
    )
    if finalized_episode_rewards:
        print(
            f"Finalized episodes - Mean: {np.mean(finalized_episode_rewards):.2f}, Std: {np.std(finalized_episode_rewards):.2f}, Min: {np.min(finalized_episode_rewards):.2f}, Max: {np.max(finalized_episode_rewards):.2f}, Count: {len(finalized_episode_rewards)}"
        )

    # Log to wandb
    if wandb.run is not None:
        wandb.log(reward_stats)
        video_path = "inference_demo.mp4"
        if os.path.exists(video_path):
            wandb.log({"video_inference": wandb.Video(video_path, fps=30, format="mp4")})
        wandb.finish()

    return reward_stats


# Select base config class based on --env flag
if ENV_TYPE == "warehouse":
    BaseEnvCfg = AnymalDWarehouseEnvCfg
    print("Using warehouse environment")
else:
    BaseEnvCfg = AnymalDFlatEnvCfg
    print("Using flat environment")


def create_environment():
    """Create environment with current VELOCITY_CONFIG values."""
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
                observation_scale=OBSERVATION_SCALE,
                observation_offset=OBSERVATION_OFFSET,
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
                observation_scale=OBSERVATION_SCALE,
                observation_offset=OBSERVATION_OFFSET,
                actuator_stiffness=ACTUATOR_STIFFNESS,
                actuator_damping=ACTUATOR_DAMPING,
            )
        print("  Getting config...")
        env_cfg = env_cfg_creator.get_cfg()
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


# Create environment
env = create_environment()

# Create imgs directory
os.makedirs("imgs", exist_ok=True)

# Run inference-only simulation
print("\nRunning inference simulation...")
run_simulation()

env.close()
simulation_app.close()
