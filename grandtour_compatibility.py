""" To make Isaac Gym data compatible with GrandTour during Imitation Learning """

import numpy as np
import torch
from isaac_compatibility import (
    LIN_VEL_SCALE,
    ANG_VEL_SCALE,
    DOF_VEL_SCALE,
    COMMANDS_SCALE,
    obs_scale,
    action_scale,
    build_default_dof_pos,
)

# Isaac Gym observation scaling factors (from legged_robot_config.py)
# These match what Isaac Gym applies in compute_observations()
ISAAC_LIN_VEL_SCALE = 2.0
ISAAC_ANG_VEL_SCALE = 0.25
ISAAC_DOF_POS_SCALE = 1.0
ISAAC_DOF_VEL_SCALE = 0.05
ISAAC_COMMANDS_SCALE = np.array([2.0, 2.0, 0.25])  # [lin_vel_x, lin_vel_y, ang_vel_yaw]


def unscale_lin_vel(lin_vel_scaled):
    """
    Unscale linear velocity from Isaac Gym format to GrandTour format.
    Isaac Gym: lin_vel * 2.0
    GrandTour: lin_vel (unscaled)
    """
    return lin_vel_scaled / ISAAC_LIN_VEL_SCALE


def unscale_ang_vel(ang_vel_scaled):
    """
    Unscale angular velocity from Isaac Gym format to GrandTour format.
    Isaac Gym: ang_vel * 0.25
    GrandTour: ang_vel (unscaled)
    """
    return ang_vel_scaled / ISAAC_ANG_VEL_SCALE


def unscale_commands(commands_scaled):
    """
    Unscale commands from Isaac Gym format to GrandTour format.
    Isaac Gym: commands * [2.0, 2.0, 0.25]
    GrandTour: commands (unscaled)
    """
    if isinstance(commands_scaled, torch.Tensor):
        return commands_scaled / torch.tensor(ISAAC_COMMANDS_SCALE, device=commands_scaled.device, dtype=commands_scaled.dtype)
    return commands_scaled / ISAAC_COMMANDS_SCALE


def unscale_joint_pos(joint_pos_scaled):
    """
    Convert joint positions from IsaacLab format to GrandTour format.

    IsaacLab: (dof_pos - isaac_default) * 1.0
    GrandTour: (dof_pos - gt_default) * 1.0

    Since both use scale=1.0, only the centering reference differs:
        gt_obs = isaac_obs + (isaac_default - gt_default)
    """
    isaac_default = build_default_dof_pos(use_grand_tour=False)
    gt_default = build_default_dof_pos(use_grand_tour=True)
    offset = isaac_default - gt_default

    if isinstance(joint_pos_scaled, torch.Tensor):
        offset_torch = torch.tensor(offset, device=joint_pos_scaled.device, dtype=joint_pos_scaled.dtype)
        return joint_pos_scaled + offset_torch

    return joint_pos_scaled + offset


def unscale_joint_vel(joint_vel_scaled):
    """
    Unscale joint velocities from Isaac Gym format to GrandTour format.
    Isaac Gym: dof_vel * 0.05
    GrandTour: joint_vel (unscaled)
    """
    return joint_vel_scaled / ISAAC_DOF_VEL_SCALE


def unscale_previous_actions(actions_scaled):
    """
    Previous actions stored by IsaacLab are already in GrandTour format — pass-through.

    make_actions_compatible() encodes actions as (target - gt_default) / 0.5 using GT
    defaults. IsaacLab stores the raw value passed to env.step(), so what's in the obs
    is already (target - gt_default) / 0.5, which is exactly the GT prev_actions format.
    No conversion needed.
    """
    return actions_scaled


def unscale_observations(obs_isaac, device="cpu"):
    """
    Convert observations from IsaacLab format to GrandTour format.

    IsaacLab and GrandTour use identical scaling for every observation term:
      lin_vel ×2.0, ang_vel ×0.25, commands ×[2,2,0.25], joint_vel ×0.05,
      projected_gravity unscaled, prev_actions (target-gt_default)/0.5.

    The ONLY difference is joint_pos centering:
      IsaacLab: (q - isaac_default) * 1.0
      GrandTour: (q - gt_default) * 1.0

    All other terms are passed through unchanged.

    Args:
        obs_isaac: Observations from IsaacLab (torch.Tensor or np.ndarray), 36 or 48 dims
        device: unused, kept for API compatibility

    Returns:
        Observations in GrandTour format (same type as input)
    """
    if isinstance(obs_isaac, torch.Tensor):
        obs_gt = obs_isaac.clone()
    else:
        obs_gt = obs_isaac.copy()

    # Fix joint_pos centering [12:24]: shift from isaac_default to gt_default reference
    obs_gt[..., 12:24] = unscale_joint_pos(obs_gt[..., 12:24])

    # All other terms are identical between IsaacLab and GrandTour — no conversion needed.

    return obs_gt
