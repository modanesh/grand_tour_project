# eval_isaaclab_v2.py
import argparse
import sys
import os

# Track if Isaac Sim has been launched
_isaac_sim_launched = False
_simulation_app = None


def ensure_isaac_sim():
    """Launch Isaac Sim if not already launched"""
    global _isaac_sim_launched, _simulation_app

    if not _isaac_sim_launched:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=True)
        _simulation_app = app_launcher.app
        _isaac_sim_launched = True

    return _simulation_app


# Import other dependencies that don't need Isaac Sim
import numpy as np
import torch
from tqdm import tqdm
from reward import rewards
from utils import compute_mean_std, load_hdf5_dataset
from isaac_compatibility import (
    build_default_dof_pos,
    build_isaac_default_dof_pos,
    build_grand_tour_default_dof_pos,
    isaac_default_joint_angles,
    grand_tour_default_joint_angles,
    DOF_NAMES,
    action_scale,
    make_actions_compatible,
)
from grandtour_compatibility import (
    unscale_observations,
    unscale_joint_pos,
    unscale_previous_actions,
)



def verify_joint_ordering(obs_joint_pos, expected_defaults, tolerance=0.1):
    """
    Verify if the joint positions match the expected ordering.
    
    IsaacLab observations contain relative joint positions: (joint_pos - default_pos) * scale
    At initialization/reset, joint_pos should equal default_pos, so obs_joint_pos should be near 0.
    
    Args:
        obs_joint_pos: Observed joint positions from IsaacLab (indices 12:24)
        expected_defaults: Expected default joint angles (GrandTour defaults)
        tolerance: Tolerance for matching
    
    Returns:
        bool: True if ordering appears correct
    """
    # Check if observed values are close to expected (allowing for some noise/scaling)
    diff = np.abs(obs_joint_pos - expected_defaults)
    return np.all(diff < tolerance)


class OnlineEval:
    def __init__(
        self,
        task_name,
        seed,
        dataset_path="offline_dataset_pp.hdf5",
        normalize=False,
        include_prev_actions=False,
        apply_centering_offset=True,
    ):
        # Ensure Isaac Sim is launched
        ensure_isaac_sim()

        # NOW import Isaac Lab modules (after Sim is running)
        from isaaclab.envs import ManagerBasedRLEnv
        import isaaclab_tasks
        from isaaclab_tasks.utils import parse_env_cfg

        self.include_prev_actions = include_prev_actions
        self.normalize = normalize
        self.apply_centering_offset = apply_centering_offset
        self.debug_mode_freeze_robot = True # TODO: for debugging only.

        # Store flag for full observation un-scaling (IsaacLab -> GrandTour)
        # IsaacLab observations are scaled; GrandTour policy expects unscaled observations
        self.unscale_observations = True

        # Parse environment configuration
        env_cfg = parse_env_cfg(
            task_name,
            device="cuda:0",
            num_envs=4096,
        )

        """
        # Disable curriculum and noise if needed
        if hasattr(env_cfg, 'curriculum'):
            env_cfg.curriculum.terrain_levels = False
        """

        if hasattr(env_cfg, "observations"):
            if hasattr(env_cfg.observations, "enable_corruption"):
                env_cfg.observations.enable_corruption = False

        # Create environment
        # Enforce joint-position control with scale=1.0
        import dataclasses as _dc
        from isaaclab.envs.mdp.actions import JointPositionActionCfg
        for _field in _dc.fields(env_cfg.actions):
            _term = getattr(env_cfg.actions, _field.name)
            if hasattr(_term, "asset_name") and hasattr(_term, "joint_names"):
                setattr(
                    env_cfg.actions,
                    _field.name,
                    JointPositionActionCfg(
                        asset_name=_term.asset_name,
                        joint_names=_term.joint_names,
                        scale=1.0,
                    ),
                )
                print(f"[action override] {_field.name}: {type(_term).__name__} -> JointPositionActionCfg(scale=1.0)")

        print(f">>>>>> :{env_cfg}")
        env = ManagerBasedRLEnv(cfg=env_cfg)

        self.env_cfg = env_cfg
        self.env = env
        self.dt = env.step_dt


    def calculate_total_reward(self, rewbuffer, ep_infos, lenbuffer):
        """Calculate mean reward from episode buffers"""
        if len(rewbuffer) > 0:
            avg_total_ep_rew = np.mean(rewbuffer)
            avg_episode_length = np.mean(lenbuffer) if len(lenbuffer) > 0 else 0.0
        else:
            avg_total_ep_rew = 0.0
            avg_episode_length = 0.0

        scaled_rew_terms_avg = {}
        if ep_infos:
            for key in ep_infos[0].keys():
                infotensor = np.array([])
                for ep_info in ep_infos:
                    value = ep_info[key]
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()
                    if not isinstance(value, np.ndarray):
                        value = np.array([value])
                    if value.ndim == 0:
                        value = np.expand_dims(value, 0)

                    if len(infotensor) == 0:
                        infotensor = value
                    else:
                        infotensor = np.concatenate((infotensor, value))

                value = np.mean(infotensor)
                scaled_rew_terms_avg[key] = float(value)

        return (
            avg_total_ep_rew,
            len(rewbuffer),
            scaled_rew_terms_avg,
            avg_episode_length,
        )

    @torch.no_grad()
    def eval_actor_isaac(self, actor: torch.nn.Module, device: str = "cuda") -> tuple:
        actor.eval()
        env = self.env

        # Reset - handle dict return
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs_dict, _ = reset_result
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        else:
            obs = reset_result

        if not self.include_prev_actions and obs.shape[-1] > 36:
            obs = obs[:, :-12]

        num_envs = env.num_envs
        max_steps = (
            int(env.max_episode_length_s / env.step_dt)
            if hasattr(env, "max_episode_length_s")
            else 1000
        )

        cur_reward_sum = torch.zeros(num_envs, dtype=torch.float, device=device)
        episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
        rewbuffer = []
        ep_infos = []
        lenbuffer = []
        obs_stats_list = []
        obs_all_list = []
        actions_all_list = []

        # Debug: Print initial observation stats to verify joint ordering
        first_obs = True
        
        for i in range(max_steps + 2):
            # Sanity check on raw IsaacLab obs BEFORE any conversion.
            # At reset, raw obs[12:24] = (q - isaac_default) * 1.0 ≈ 0.
            if first_obs and i == 0:
                raw_joint_pos = obs[0, 12:24].cpu().numpy()
                max_diff = np.max(np.abs(raw_joint_pos))
                tolerance = 0.5  # rad
                print(f"\n=== Observation Debug (step {i}, RAW IsaacLab obs) ===")
                print(f"Base lin vel [0:3]: {obs[0, 0:3].cpu().numpy()}")
                print(f"Base ang vel [3:6]: {obs[0, 3:6].cpu().numpy()}")
                print(f"Raw joint pos [12:24] (should be ~0): {raw_joint_pos}")
                print(f"Max abs deviation from 0: {max_diff:.4f} rad")
                if max_diff > tolerance:
                    isaac_defaults = torch.tensor(build_isaac_default_dof_pos(), device=obs.device, dtype=obs.dtype)
                    gt_defaults = torch.tensor(build_grand_tour_default_dof_pos(), device=obs.device, dtype=obs.dtype)
                    error_msg = (
                        f"\n{'='*60}\n"
                        f"JOINT ORDERING MISMATCH DETECTED!\n"
                        f"{'='*60}\n"
                        f"Raw obs[12:24] at reset should be ~0 (q - isaac_default), "
                        f"but max deviation is {max_diff:.4f} rad (tolerance: {tolerance} rad)\n"
                        f"\nRaw joint pos obs: {raw_joint_pos}\n"
                        f"IsaacLab defaults:  {isaac_defaults.cpu().numpy()}\n"
                        f"GT defaults:        {gt_defaults.cpu().numpy()}\n"
                        f"\nPossible causes:\n"
                        f"1. IsaacLab joint order differs from DOF_NAMES order\n"
                        f"2. Robot spawned in unexpected initial pose\n"
                        f"3. Joint position scaling/conversion error\n"
                        f"\nDOF_NAMES order (isaac_compatibility.py):\n"
                        f"  {DOF_NAMES}\n"
                        f"{'='*60}"
                    )
                    raise RuntimeError(error_msg)
                print(f"✓ Joint ordering check passed (max deviation from 0: {max_diff:.4f} rad)")
                print("=====================================\n")
                first_obs = False

            # # Convert IsaacLab observations to GrandTour format.
            # # Only joint_pos centering differs; all other scales are identical.
            # if self.unscale_observations:
            #     obs = unscale_observations(obs, device=obs.device)
            # elif self.apply_centering_offset and self.joint_pos_offset is not None:
            #     # Fallback: only apply joint position offset (legacy behavior)
            #     offset = self.joint_pos_offset.to(obs.device)
            #     obs[:, 12:24] = obs[:, 12:24] + offset

            # obs_normalized = (
            #     (obs - state_mean_torch) / state_std_torch if self.normalize else obs
            # )

            obs_stats_list.append(
                {
                    "mean": obs.mean().item(),
                    "std": obs.std().item(),
                    "min": obs.min().item(),
                    "max": obs.max().item(),
                }
            )
            obs_all_list.append(obs.cpu().numpy())

            # Policy outputs actions in GrandTour format (absolute positions)
            actions = actor.act_inference(obs.detach())
            actions_all_list.append(actions.cpu().numpy())
            
            # Convert actions from GrandTour format (absolute) to IsaacLab format (offsets)
            # IsaacLab expects: action = (target_pos - default_pos) / action_scale

            if self.debug_mode_freeze_robot:
                actions_isaaclab = obs[:, 12:24] 
            else:
                actions_isaaclab = actions  # action_scale=1.0, no rescaling needed

            # simulation next step
            step_result = env.step(actions_isaaclab.detach())

            if len(step_result) == 5:
                obs_dict, rew, terminated, truncated, infos = step_result
                obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
                dones = terminated | truncated
            else:
                obs, _, rew, dones, infos = step_result

            if not self.include_prev_actions and obs.shape[-1] > 36:
                obs = obs[:, :-12]

            if rew.dim() > 1:
                rew = rew.squeeze()

            cur_reward_sum += rew
            episode_lengths += 1

            if dones.dim() > 1:
                dones = dones.squeeze()
            done_indices = torch.where(dones == 1)[0]

            if len(done_indices) > 0:
                rewbuffer.extend(cur_reward_sum[done_indices].cpu().numpy().tolist())
                lenbuffer.extend(episode_lengths[done_indices].cpu().numpy().tolist())
                cur_reward_sum[done_indices] = 0
                episode_lengths[done_indices] = 0

                if "episode" in infos:
                    ep_infos.append(infos["episode"])

        actor.train()

        if obs_stats_list:
            avg_obs_mean = np.mean([s["mean"] for s in obs_stats_list])
            avg_obs_std = np.mean([s["std"] for s in obs_stats_list])
            avg_obs_min = np.mean([s["min"] for s in obs_stats_list])
            avg_obs_max = np.mean([s["max"] for s in obs_stats_list])

            obs_all_array = np.concatenate(obs_all_list, axis=0)
            obs_mean_per_dim = obs_all_array.mean(axis=0)
            obs_mean_per_dim_dict = {
                f"isaac_obs_mean_dim_{i}": float(val)
                for i, val in enumerate(obs_mean_per_dim)
            }

            actions_all_array = np.concatenate(actions_all_list, axis=0)
            actions_mean_per_dim = actions_all_array.mean(axis=0)
            actions_mean_per_dim_dict = {
                f"isaac_action_mean_dim_{i}": float(val)
                for i, val in enumerate(actions_mean_per_dim)
            }

            eval_score, n_eps_evaluated, scaled_rew_terms_avg, avg_episode_length = (
                self.calculate_total_reward(rewbuffer, ep_infos, lenbuffer)
            )

            obs_stats = {
                "isaac_obs_mean": avg_obs_mean,
                "isaac_obs_std": avg_obs_std,
                "isaac_obs_min": avg_obs_min,
                "isaac_obs_max": avg_obs_max,
                **obs_mean_per_dim_dict,
                **actions_mean_per_dim_dict,
            }

            return (
                eval_score,
                n_eps_evaluated,
                scaled_rew_terms_avg,
                avg_episode_length,
                obs_stats,
            )
        else:
            eval_score, n_eps_evaluated, scaled_rew_terms_avg, avg_episode_length = (
                self.calculate_total_reward(rewbuffer, ep_infos, lenbuffer)
            )
            return (
                eval_score,
                n_eps_evaluated,
                scaled_rew_terms_avg,
                avg_episode_length,
                {},
            )
