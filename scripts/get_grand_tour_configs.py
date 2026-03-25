"""
get_grand_tour_configs.py

Prints the label associated with each dimension of the Grand Tour observation
and action spaces (as built by build_dataset.py / build_offline_dataset()).

No Isaac Sim required – the structure is derived statically from the dataset
building code and scaling constants in isaac_compatibility.py.

Optionally loads an HDF5 dataset file to report per-dimension statistics.

Usage (from project root):
    python scripts/get_grand_tour_configs.py
    python scripts/get_grand_tour_configs.py --dataset offline_dataset_pp.hdf5
    python scripts/get_grand_tour_configs.py --dataset offline_dataset_pp.hdf5 --no-prev-actions
"""

import argparse
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from isaac_compatibility import (
    DOF_NAMES,
    LIN_VEL_SCALE,
    ANG_VEL_SCALE,
    DOF_VEL_SCALE,
    COMMANDS_SCALE,
    obs_scale,
    action_scale,
    grand_tour_default_joint_angles,
)

# ── Observation structure (matches build_offline_dataset in build_dataset.py) ─
#
#   [0:3]   base_lin_vel   * LIN_VEL_SCALE (2.0)          – body frame
#   [3:6]   base_ang_vel   * ANG_VEL_SCALE (0.25)          – body frame
#   [6:9]   projected_gravity (no scaling)
#   [9:12]  commands (vx, vy, yaw_rate) * COMMANDS_SCALE ([2.0, 2.0, 0.25])
#   [12:24] joint_pos  (joint_pos - grand_tour_default) * obs_scale (1.0)
#   [24:36] joint_vel  * DOF_VEL_SCALE (0.05)
#   [36:48] prev_actions (optional)  – same format as actions

_gt_defaults = [grand_tour_default_joint_angles[n] for n in DOF_NAMES]

OBS_TERMS = [
    # (term_name, sub_labels, scale_note)
    (
        "base_lin_vel",
        ["x", "y", "z"],
        f"× {LIN_VEL_SCALE} (body frame)",
    ),
    (
        "base_ang_vel",
        ["p", "q", "r"],
        f"× {ANG_VEL_SCALE} (body frame)",
    ),
    (
        "projected_gravity",
        ["x", "y", "z"],
        "no scaling",
    ),
    (
        "velocity_commands",
        ["vx", "vy", "yaw_rate"],
        f"× {list(COMMANDS_SCALE)}",
    ),
    (
        "joint_pos",
        DOF_NAMES,
        f"(joint_pos − GT_default) × {obs_scale}",
    ),
    (
        "joint_vel",
        DOF_NAMES,
        f"× {DOF_VEL_SCALE}",
    ),
]

OBS_TERMS_WITH_PREV = OBS_TERMS + [
    (
        "prev_actions",
        DOF_NAMES,
        f"(target − GT_default) / {action_scale}",
    ),
]

ACTION_TERMS = [
    (
        "joint_position_target",
        DOF_NAMES,
        f"(target − GT_default) / {action_scale}",
    ),
]


def _print_space(terms, title):
    print(f"\n  {title}")
    print(f"  {'Dim':>8}  {'Term':<25}  {'Sub-label':<12}  Scale / note")
    print(f"  {'-'*8}  {'-'*25}  {'-'*12}  {'-'*30}")
    idx = 0
    for term_name, sub_labels, scale_note in terms:
        for sub_i, label in enumerate(sub_labels):
            print(
                f"  {idx:>8}  {term_name:<25}  {label:<12}  "
                + (scale_note if sub_i == 0 else "")
            )
            idx += 1


def _print_stats(dataset_path, include_prev_actions):
    """Load HDF5 and print per-dimension mean/std/min/max."""
    try:
        import h5py
    except ImportError:
        print("\n  [!] h5py not installed – skipping dataset stats")
        return

    print(f"\n  Loading: {dataset_path}")
    with h5py.File(dataset_path, "r") as f:
        obs = f["observations"][:]
        actions = f["actions"][:]

    terms = OBS_TERMS_WITH_PREV if include_prev_actions else OBS_TERMS
    obs_dim_expected = sum(len(t[1]) for t in terms)

    print(f"  Dataset shape  – obs: {obs.shape}, actions: {actions.shape}")
    if obs.shape[-1] != obs_dim_expected:
        print(
            f"  [!] obs dim mismatch: got {obs.shape[-1]}, expected {obs_dim_expected}. "
            "Re-run with/without --no-prev-actions."
        )

    for name, arr, terms_list in [
        ("Observation", obs, terms),
        ("Action", actions, ACTION_TERMS),
    ]:
        print(f"\n  {name} statistics:")
        print(
            f"  {'Dim':>5}  {'Term':<22}  {'Label':<12}"
            f"  {'mean':>9}  {'std':>9}  {'min':>9}  {'max':>9}"
        )
        print(f"  {'-'*5}  {'-'*22}  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
        idx = 0
        for term_name, sub_labels, _ in terms_list:
            for sub_i, label in enumerate(sub_labels):
                if idx >= arr.shape[-1]:
                    break
                col = arr[:, idx]
                print(
                    f"  {idx:>5}  {term_name:<22}  {label:<12}"
                    f"  {col.mean():>9.4f}  {col.std():>9.4f}"
                    f"  {col.min():>9.4f}  {col.max():>9.4f}"
                )
                idx += 1


def main():
    parser = argparse.ArgumentParser(
        description="Print Grand Tour observation and action space labels"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to HDF5 dataset file for per-dimension statistics (optional)",
    )
    parser.add_argument(
        "--no-prev-actions",
        action="store_true",
        help="Treat obs as 36-dim (no previous actions term)",
    )
    args = parser.parse_args()

    include_prev = not args.no_prev_actions
    obs_terms = OBS_TERMS_WITH_PREV if include_prev else OBS_TERMS
    obs_dim = sum(len(t[1]) for t in obs_terms)
    act_dim = sum(len(t[1]) for t in ACTION_TERMS)

    print("\n" + "=" * 70)
    print("Grand Tour Dataset – Observation & Action Space")
    print("=" * 70)
    print(f"  Observation dims : {obs_dim} ({'with' if include_prev else 'without'} prev_actions)")
    print(f"  Action dims      : {act_dim}")
    print(f"  Joint order      : {DOF_NAMES}")

    print("\n" + "=" * 70)
    print("Grand Tour default joint angles (centering reference for joint_pos / actions):")
    print("=" * 70)
    print(f"  {'Joint':<10}  {'Default (rad)':>14}  {'Default (deg)':>14}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*14}")
    for name, val in zip(DOF_NAMES, _gt_defaults):
        print(f"  {name:<10}  {val:>14.4f}  {np.degrees(val):>14.2f}")

    print("\n" + "=" * 70)
    _print_space(obs_terms, "Observation space:")
    print("\n" + "=" * 70)
    _print_space(ACTION_TERMS, "Action space:")
    print("\n" + "=" * 70)

    if args.dataset:
        _print_stats(args.dataset, include_prev)
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
