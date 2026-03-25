"""
Script to print the joint configuration of the Anymal-D robot from IsaacLab.
"""

import numpy as np

# Joint names in URDF order (12 DOFs)
DOF_NAMES = [
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",  # Left Front leg
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",  # Right Front leg
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",  # Left Hind leg
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",  # Right Hind leg
]

# IsaacLab default joint angles (standing pose)
isaac_default_joint_angles = {
    # Hip Abduction/Adduction (HAA)
    "LF_HAA": 0.0,
    "LH_HAA": 0.0,
    "RF_HAA": -0.0,
    "RH_HAA": -0.0,
    # Hip Flexion/Extension (HFE)
    "LF_HFE": 0.4,
    "LH_HFE": -0.4,
    "RF_HFE": 0.4,
    "RH_HFE": -0.4,
    # Knee Flexion/Extension (KFE)
    "LF_KFE": -0.8,
    "LH_KFE": 0.8,
    "RF_KFE": -0.8,
    "RH_KFE": 0.8,
}

# Grand Tour default joint angles (from real robot data)
grand_tour_default_joint_angles = {
    "LF_HAA": -0.3818,
    "LF_HFE": 0.8445,
    "LF_KFE": -1.3450,
    "RF_HAA": 0.4191,
    "RF_HFE": 0.7747,
    "RF_KFE": -1.3316,
    "LH_HAA": -0.3890,
    "LH_HFE": -0.5935,
    "LH_KFE": 1.2910,
    "RH_HAA": 0.4031,
    "RH_HFE": -0.7371,
    "RH_KFE": 1.3671,
}

# Joint limits (typical values for Anymal-D)
joint_limits = {
    "HAA": {"lower": -0.8, "upper": 0.8},  # Hip Abduction/Adduction
    "HFE": {"lower": -1.5, "upper": 1.5},  # Hip Flexion/Extension
    "KFE": {"lower": -2.5, "upper": 2.5},  # Knee Flexion/Extension
}

# Scaling constants from IsaacLab
LIN_VEL_SCALE = 2.0  # Linear velocity scaling
ANG_VEL_SCALE = 0.25  # Angular velocity scaling
DOF_VEL_SCALE = 0.05  # Joint velocity scaling
obs_scale = 1.0  # Joint position scaling (after centering)
action_scale = 0.5  # Action scaling


def print_separator(char="=", length=70):
    print(char * length)


def print_joint_config():
    print("\n")
    print_separator("=")
    print("Anymal-D Robot Joint Configuration")
    print_separator("=")

    print("\n[ General Information ]")
    print(f"  - Total Degrees of Freedom: {len(DOF_NAMES)}")
    print("  - Number of Legs: 4")
    print("  - Joints per Leg: 3 (HAA, HFE, KFE)")
    print("  - Action Dimension: 12 (delta joint positions)")
    print("  - Observation Dimension: 36 (or 48 with prev actions)")

    print("\n" + "-" * 70)
    print("Joint Names and Order:")
    print("-" * 70)

    leg_names = [
        "Left Front (LF)",
        "Right Front (RF)",
        "Left Hind (LH)",
        "Right Hind (RH)",
    ]
    joint_types = [
        "HAA (Hip Abduction/Adduction)",
        "HFE (Hip Flexion/Extension)",
        "KFE (Knee Flexion/Extension)",
    ]

    for i, leg_name in enumerate(leg_names):
        print(f"\n  {leg_name}:")
        for j in range(3):
            idx = i * 3 + j
            print(f"    [{idx:2d}] {DOF_NAMES[idx]:8s} - {joint_types[j]}")

    print("\n" + "-" * 70)
    print("Default Joint Angles (IsaacLab - Standing Pose):")
    print("-" * 70)
    for leg_name in leg_names:
        leg_prefix = leg_name.split()[0][0] + leg_name.split()[1][0]  # LF, RF, LH, RH
        print(f"\n  {leg_name}:")
        for joint_type in ["HAA", "HFE", "KFE"]:
            joint_name = f"{leg_prefix}_{joint_type}"
            angle = isaac_default_joint_angles[joint_name]
            print(
                f"    {joint_name:8s}: {angle:7.4f} rad ({np.degrees(angle):7.2f} deg)"
            )

    print("\n" + "-" * 70)
    print("Default Joint Angles (Grand Tour - Real Robot):")
    print("-" * 70)
    for leg_name in leg_names:
        leg_prefix = leg_name.split()[0][0] + leg_name.split()[1][0]
        print(f"\n  {leg_name}:")
        for joint_type in ["HAA", "HFE", "KFE"]:
            joint_name = f"{leg_prefix}_{joint_type}"
            angle = grand_tour_default_joint_angles[joint_name]
            print(
                f"    {joint_name:8s}: {angle:7.4f} rad ({np.degrees(angle):7.2f} deg)"
            )

    print("\n" + "-" * 70)
    print("Joint Limits:")
    print("-" * 70)
    for joint_type, limits in joint_limits.items():
        print(
            f"  {joint_type}: [{limits['lower']:6.2f}, {limits['upper']:6.2f}] rad "
            f"([{np.degrees(limits['lower']):6.1f} deg, {np.degrees(limits['upper']):6.1f} deg])"
        )

    print("\n" + "-" * 70)
    print("Observation/Action Scaling Constants:")
    print("-" * 70)
    print(f"  Linear Velocity Scale:  {LIN_VEL_SCALE}")
    print(f"  Angular Velocity Scale: {ANG_VEL_SCALE}")
    print(f"  Joint Velocity Scale:   {DOF_VEL_SCALE}")
    print(f"  Joint Position Scale:   {obs_scale}")
    print(f"  Action Scale:           {action_scale}")

    print("\n" + "-" * 70)
    print("Observation Vector Structure (36 dims):")
    print("-" * 70)
    print("  [0:3]   - Linear velocity (body frame, scaled)")
    print("  [3:6]   - Angular velocity (body frame, scaled)")
    print("  [6:9]   - Projected gravity vector")
    print("  [9:12]  - Commands (vx, vy, yaw rate, scaled)")
    print("  [12:24] - Joint positions (centered & scaled)")
    print("  [24:36] - Joint velocities (scaled)")

    print("\n" + "-" * 70)
    print("Action Vector Structure (12 dims):")
    print("-" * 70)
    print("  Delta joint position targets (scaled by action_scale=0.5)")
    for i, dof_name in enumerate(DOF_NAMES):
        print(f"  [{i:2d}] - {dof_name}")

    print("\n")
    print_separator("=")
    print("End of Anymal-D Configuration")
    print_separator("=")
    print("\n")


def get_isaaclab_cfg_from_env():
    """
    Try to load the actual IsaacLab configuration from the environment.
    This requires IsaacLab to be installed.
    """
    try:
        from isaaclab_tasks.utils import parse_env_cfg
        import isaaclab_tasks

        print_separator("=")
        print("IsaacLab Environment Configuration")
        print_separator("=")

        task_name = "Isaac-Velocity-Flat-Anymal-D-v0"
        print(f"\nTask: {task_name}")

        # Parse the environment configuration
        env_cfg = parse_env_cfg(task_name, device="cpu", num_envs=1)

        print("\n[ Environment Configuration ]")
        print(f"  Scene:")
        if hasattr(env_cfg, "scene"):
            if hasattr(env_cfg.scene, "robot"):
                robot_cfg = env_cfg.scene.robot
                print(f"    Robot: {robot_cfg.__class__.__name__}")

                # Try to get joint information
                if hasattr(robot_cfg, "spawn"):
                    print(f"    Spawn config: {robot_cfg.spawn}")

        print("\n📋 Observation Configuration:")
        if hasattr(env_cfg, "observations"):
            print(
                f"  Corruption enabled: {getattr(env_cfg.observations, 'enable_corruption', 'N/A')}"
            )

        print("\n[ Action Configuration ]")
        if hasattr(env_cfg, "actions"):
            print(f"  Actions: {env_cfg.actions}")

        print("\n[ Rewards Configuration ]")
        if hasattr(env_cfg, "rewards"):
            reward_terms = list(env_cfg.rewards.__dict__.keys())
            print(f"  Reward terms: {reward_terms[:5]}...")

        return env_cfg

    except ImportError as e:
        print(f"\n[!] Could not import IsaacLab: {e}")
        print("   To get the full IsaacLab config, please install isaaclab-tasks:")
        print("   pip install isaaclab-tasks")
        return None
    except Exception as e:
        print(f"\n[!] Error loading IsaacLab config: {e}")
        return None


if __name__ == "__main__":
    # Print the joint configuration
    print_joint_config()

    # Try to get IsaacLab config if available
    print("\n")
    get_isaaclab_cfg_from_env()
