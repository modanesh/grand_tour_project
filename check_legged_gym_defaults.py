"""
Helper script to verify whether legged_gym uses init_state.default_joint_angles
to build default_dof_pos (used in obs computation and robot reset).

Run on the server:
    python check_legged_gym_defaults.py

Prints:
  1. Where legged_gym is installed
  2. The relevant source lines from legged_robot.py
  3. Makes env and prints env.default_dof_pos before and after overriding
     init_state.default_joint_angles, to confirm the override works.
"""

import inspect
import importlib
import sys

# ── 1. Find legged_gym ────────────────────────────────────────────────────────
import legged_gym
print(f"legged_gym location: {legged_gym.__file__}")

# ── 2. Print relevant source from LeggedRobot ────────────────────────────────
try:
    from legged_gym.envs.base.legged_robot import LeggedRobot

    src = inspect.getsource(LeggedRobot._init_buffers)
    print("\n=== LeggedRobot._init_buffers (search for default_dof_pos) ===")
    for i, line in enumerate(src.splitlines()):
        if "default_dof_pos" in line or "default_joint_angles" in line:
            print(f"  {i:3d}: {line}")

    src2 = inspect.getsource(LeggedRobot.compute_observations)
    print("\n=== LeggedRobot.compute_observations (search for default_dof_pos) ===")
    for i, line in enumerate(src2.splitlines()):
        if "default_dof_pos" in line or "dof_pos" in line:
            print(f"  {i:3d}: {line}")

    src3 = inspect.getsource(LeggedRobot.reset_idx)
    print("\n=== LeggedRobot.reset_idx (search for default_dof_pos / default_joint_angles) ===")
    for i, line in enumerate(src3.splitlines()):
        if "default_dof_pos" in line or "default_joint_angles" in line or "dof_pos" in line.lower():
            print(f"  {i:3d}: {line}")

except Exception as e:
    print(f"Could not inspect LeggedRobot source: {e}")

# ── 3. Live test: does overriding init_state.default_joint_angles change env.default_dof_pos? ──
print("\n=== Live test ===")
try:
    import isaacgym  # must import before torch
    import torch
    from legged_gym.utils import get_args, task_registry

    args = get_args()
    args.task = "anymal_d_flat"
    args.headless = True
    args.num_envs = 1

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False

    # Make env with default (IG) settings
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ig_default_dof = env.default_dof_pos[0].cpu().tolist()
    print(f"env.default_dof_pos (IG defaults): {[round(x,4) for x in ig_default_dof]}")

    # Now override with GT defaults and remake
    from isaac_compatibility import gt_default_joint_angles, DOF_NAMES
    env_cfg2, _ = task_registry.get_cfgs(name=args.task)
    env_cfg2.env.num_envs = 1
    env_cfg2.terrain.num_rows = 1
    env_cfg2.terrain.num_cols = 1
    env_cfg2.terrain.curriculum = False
    env_cfg2.init_state.default_joint_angles = gt_default_joint_angles

    env2, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg2)
    gt_default_dof = env2.default_dof_pos[0].cpu().tolist()
    print(f"env.default_dof_pos (GT override): {[round(x,4) for x in gt_default_dof]}")

    changed = ig_default_dof != gt_default_dof
    print(f"\nOverride works: {changed}")
    if not changed:
        print("  -> default_dof_pos is NOT sourced from init_state.default_joint_angles.")
        print("     You will need to patch env.default_dof_pos manually after make_env().")
    else:
        print("  -> Overriding env_cfg.init_state.default_joint_angles before make_env() is sufficient.")

    # Print joint-by-joint comparison
    print(f"\n{'Joint':<12} {'IG default':>12} {'GT override':>12} {'match':>8}")
    for i, name in enumerate(DOF_NAMES):
        ig_val = round(ig_default_dof[i], 4)
        gt_val = round(gt_default_dof[i], 4)
        match = "OK" if abs(ig_val - gt_val) < 1e-4 else "CHANGED"
        print(f"{name:<12} {ig_val:>12} {gt_val:>12} {match:>8}")

except Exception as e:
    print(f"Live test failed: {e}")
    import traceback
    traceback.print_exc()
