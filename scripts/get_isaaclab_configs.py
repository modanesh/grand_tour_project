"""
get_isaaclab_configs.py

Launches the IsaacLab environment and prints the label associated with each
dimension of the observation space by introspecting the ObservationManager.

Usage (from project root, with the isaaclab conda env active):
    python scripts/get_isaaclab_configs.py
    python scripts/get_isaaclab_configs.py --task Isaac-Velocity-Flat-Anymal-D-v0
    python scripts/get_isaaclab_configs.py --task Isaac-Velocity-Flat-Anymal-D-v0 --num-envs 1 --device cpu
"""

import argparse
import sys
import os

# ── 1. Parse CLI args before Isaac Sim starts ────────────────────────────────
parser = argparse.ArgumentParser(description="Print IsaacLab observation space labels")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-Anymal-D-v0",
    help="IsaacLab task name",
)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

# ── 2. Launch Isaac Sim (must happen before any omni/isaac imports) ───────────
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# ── 3. Now safe to import IsaacLab modules ───────────────────────────────────
import torch
import isaaclab_tasks  # noqa: F401  registers all tasks
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

# ── 4. Build env cfg & create env ────────────────────────────────────────────
print(f"\nLoading task: {args.task}")
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)

# Disable observation corruption so we get clean shapes
if hasattr(env_cfg, "observations") and hasattr(
    env_cfg.observations, "enable_corruption"
):
    env_cfg.observations.enable_corruption = False

env = ManagerBasedRLEnv(cfg=env_cfg)

# ── 5. Introspect ObservationManager ─────────────────────────────────────────
obs_manager = env.observation_manager

print("\n" + "=" * 70)
print(f"Task:              {args.task}")
print(f"Device:            {args.device}")
print(f"Num envs:          {args.num_envs}")
print("=" * 70)

# group_obs_term_dim: dict[group_name, list[int]]  (dims per term, one entry per term)
# active_terms:       dict[group_name, list[ObsTermCfg]]  (term configs, same order)
if hasattr(obs_manager, "group_obs_term_dim"):
    raw_dims = obs_manager.group_obs_term_dim
elif hasattr(obs_manager, "_group_obs_term_dim"):
    raw_dims = obs_manager._group_obs_term_dim
else:
    raw_dims = None

def _get_term_name(term) -> str:
    """Extract name string from a term config or string."""
    if isinstance(term, str):
        return term
    return getattr(term, "name", str(term))

def _build_term_list(group_name: str) -> list:
    """
    Return list of (term_name, n_dims) pairs for a group,
    using active_terms for names and group_obs_term_dim for sizes.
    """
    dims = raw_dims[group_name]  # list[int]
    terms = obs_manager.active_terms.get(group_name, [])
    if len(terms) == len(dims):
        return [(_get_term_name(t), d) for t, d in zip(terms, dims)]
    # Shape mismatch – fall back to anonymous labels
    return [(f"term_{i}", d) for i, d in enumerate(dims)]

# ── Per-group, per-term breakdown ─────────────────────────────────────────────
def _print_group(group_name: str, named_terms: list):
    """Print one observation group with per-dimension labels."""
    print(f"\n  Group: '{group_name}'")
    print(f"  {'Dim':>8}  {'Term':<30}  {'Sub-index'}")
    print(f"  {'-'*8}  {'-'*30}  {'-'*20}")

    global_idx = 0
    for term_name, n_dims in named_terms:
        if n_dims == 1:
            print(f"  {global_idx:>8}  {term_name:<30}  [0]")
        else:
            sub_labels = _sub_labels(term_name, n_dims)
            for sub_i in range(n_dims):
                label = sub_labels[sub_i] if sub_i < len(sub_labels) else str(sub_i)
                print(f"  {global_idx + sub_i:>8}  {term_name:<30}  [{sub_i}] {label}")
        global_idx += n_dims


def _sub_labels(term_name: str, n):
    """
    Best-effort per-dimension labels for common IsaacLab observation terms.
    Falls back to numeric indices for unknown terms.
    """
    from isaac_compatibility import DOF_NAMES  # project-local

    n = int(n)  # guard against numpy int types
    xyz = ["x", "y", "z"]
    rpy = ["roll", "pitch", "yaw"]

    known = {
        # velocity / IMU terms
        "base_lin_vel":         xyz[:n],
        "base_ang_vel":         ["p", "q", "r"][:n],
        "projected_gravity":    xyz[:n],
        # commands
        "velocity_commands":    ["vx", "vy", "yaw_rate"][:n],
        # joint terms – 12-dim follow DOF_NAMES order
        "joint_pos":            DOF_NAMES[:n],
        "joint_pos_rel":        DOF_NAMES[:n],
        "joint_vel":            DOF_NAMES[:n],
        "joint_vel_rel":        DOF_NAMES[:n],
        "last_action":          DOF_NAMES[:n],
        "actions":              DOF_NAMES[:n],
        # height scanner (variable dims)
        "height_scan":          [f"h_{i}" for i in range(n)],
    }

    # Fuzzy match: check if any key is a substring of the term name
    for key, labels in known.items():
        if key in term_name.lower():
            return labels

    return [str(i) for i in range(n)]


if raw_dims is not None:
    for group_name in raw_dims:
        _print_group(group_name, _build_term_list(group_name))
else:
    # Fallback: reconstruct from active_terms + a reset observation
    print("\n  [!] group_obs_term_dim not found – falling back to shape introspection")

    obs_dict, _ = env.reset()
    for group_name, obs_tensor in obs_dict.items():
        n_total = obs_tensor.shape[-1]
        print(f"\n  Group: '{group_name}'  –  total dims: {n_total}")
        print(f"  {'Dim':>8}  {'Term':<30}  {'Sub-index'}")
        print(f"  {'-'*8}  {'-'*30}  {'-'*20}")

        # Try to read term shapes from active_terms
        if hasattr(obs_manager, "active_terms") and group_name in obs_manager.active_terms:
            idx = 0
            for term in obs_manager.active_terms[group_name]:
                term_name = term.name if hasattr(term, "name") else str(term)
                # Compute this term's contribution via the manager
                try:
                    term_dim = obs_manager.compute_group(group_name).shape[-1]
                except Exception:
                    term_dim = 1
                sub_labels = _sub_labels(term_name, term_dim)
                for sub_i in range(term_dim):
                    label = sub_labels[sub_i] if sub_i < len(sub_labels) else str(sub_i)
                    print(f"  {idx + sub_i:>8}  {term_name:<30}  [{sub_i}] {label}")
                idx += term_dim
        else:
            for i in range(n_total):
                print(f"  {i:>8}  (unknown)                       [{i}]")

# ── 6. Summary table ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Observation group totals:")
print("=" * 70)

obs_dict_reset, _ = env.reset()
for group_name, obs_tensor in obs_dict_reset.items():
    print(f"  {group_name:<30}  {obs_tensor.shape[-1]} dims")

# ── 7. Also print action space ────────────────────────────────────────────────
from isaac_compatibility import DOF_NAMES  # noqa: E402

print("\n" + "=" * 70)
print("Action space:")
print("=" * 70)
action_manager = getattr(env, "action_manager", None)
if action_manager is not None and hasattr(action_manager, "active_terms"):
    idx = 0
    for group_name, terms in action_manager.active_terms.items():
        print(f"\n  Group: '{group_name}'")
        for term in terms:
            term_name = term.name if hasattr(term, "name") else str(term)
            # Try to get dim from the term's data_info
            try:
                term_dim = term.data.shape[-1]
            except Exception:
                term_dim = len(DOF_NAMES)
            for sub_i in range(term_dim):
                label = DOF_NAMES[sub_i] if sub_i < len(DOF_NAMES) else str(sub_i)
                print(f"  {idx + sub_i:>8}  {term_name:<30}  [{sub_i}] {label}")
            idx += term_dim
else:
    print(f"  12 dims (joint position targets)")
    for i, name in enumerate(DOF_NAMES):
        print(f"  {i:>8}  joint_position_target            [{i}] {name}")

print("\n" + "=" * 70)

# ── 8. Cleanup ────────────────────────────────────────────────────────────────
env.close()
simulation_app.close()
