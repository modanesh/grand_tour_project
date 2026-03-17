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
    dims = raw_dims[group_name]  # list[int | tuple]
    # Normalise: (3,) → 3
    dims = [int(d[0]) if isinstance(d, tuple) else int(d) for d in dims]
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
    terms = action_manager.active_terms
    # active_terms may be a list or a dict
    if isinstance(terms, dict):
        terms = [t for group in terms.values() for t in group]
    print(f"  {'Dim':>8}  {'Term':<30}  {'Sub-index'}")
    print(f"  {'-'*8}  {'-'*30}  {'-'*20}")
    idx = 0
    for term in terms:
        term_name = term.name if hasattr(term, "name") else str(term)
        try:
            term_dim = term.data.shape[-1]
        except Exception:
            term_dim = len(DOF_NAMES)
        for sub_i in range(term_dim):
            label = DOF_NAMES[sub_i] if sub_i < len(DOF_NAMES) else str(sub_i)
            print(f"  {idx + sub_i:>8}  {term_name:<30}  [{sub_i}] {label}")
        idx += term_dim
else:
    print(f"  {'Dim':>8}  {'Term':<30}  {'Sub-index'}")
    print(f"  {'-'*8}  {'-'*30}  {'-'*20}")
    for i, name in enumerate(DOF_NAMES):
        print(f"  {i:>8}  joint_position_target            [{i}] {name}")

print("\n" + "=" * 70)

# ── 8. Scaling factors & default poses ───────────────────────────────────────
import dataclasses
from isaac_compatibility import (
    build_isaac_default_dof_pos, build_grand_tour_default_dof_pos,
)

def _action_scale_from_cfg(actions_cfg):
    """Return .scale from the first action term that has one."""
    if not dataclasses.is_dataclass(actions_cfg):
        return None
    for field in dataclasses.fields(actions_cfg):
        term = getattr(actions_cfg, field.name)
        if hasattr(term, "scale"):
            return term.scale
    return None

def _term_scale(term):
    """Return scale from a term object, trying .cfg.scale then .scale."""
    for getter in (lambda t: t.cfg.scale, lambda t: t.scale):
        try:
            return getter(term)
        except AttributeError:
            pass
    return None

def _build_obs_scale_map():
    """Build {term_name: scale} from obs_manager.active_terms."""
    scale_map = {}
    for terms in obs_manager.active_terms.values():
        for term in terms:
            name = term.name if hasattr(term, "name") else str(term)
            scale_map[name] = _term_scale(term)
    return scale_map

obs_scale_map = _build_obs_scale_map()
print("\nDEBUG obs_scale_map:", obs_scale_map)
# DEBUG: show raw attributes on first obs term
for _terms in obs_manager.active_terms.values():
    if _terms:
        _t = _terms[0]
        print(f"DEBUG first term type: {type(_t)}, attrs: {[a for a in dir(_t) if not a.startswith('__')]}")
        break
action_scale  = _action_scale_from_cfg(env_cfg.actions)
obs_scale     = obs_scale_map.get("joint_pos") or obs_scale_map.get("joint_pos_rel")
LIN_VEL_SCALE  = obs_scale_map.get("base_lin_vel")
ANG_VEL_SCALE  = obs_scale_map.get("base_ang_vel")
DOF_VEL_SCALE  = obs_scale_map.get("joint_vel") or obs_scale_map.get("joint_vel_rel")
COMMANDS_SCALE = obs_scale_map.get("velocity_commands")

print("\n" + "=" * 70)
print("Scaling factors (from IsaacLab env_cfg):")
print("=" * 70)
print(f"  {'action_scale':<30}  {action_scale}")
print(f"  {'obs_scale (joint_pos)':<30}  {obs_scale}")
print(f"  {'LIN_VEL_SCALE':<30}  {LIN_VEL_SCALE}")
print(f"  {'ANG_VEL_SCALE':<30}  {ANG_VEL_SCALE}")
print(f"  {'DOF_VEL_SCALE':<30}  {DOF_VEL_SCALE}")
_cs = COMMANDS_SCALE.tolist() if hasattr(COMMANDS_SCALE, "tolist") else COMMANDS_SCALE
print(f"  {'COMMANDS_SCALE (vx,vy,yaw)':<30}  {_cs}")

print("\n" + "=" * 70)
print("Default DOF positions (rad):")
print("=" * 70)
print(f"  {'Joint':<12}  {'Isaac Gym':>12}  {'Grand Tour':>12}  {'Delta':>12}")
print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")
isaac_defaults = build_isaac_default_dof_pos()
gt_defaults = build_grand_tour_default_dof_pos()
for i, name in enumerate(DOF_NAMES):
    delta = gt_defaults[i] - isaac_defaults[i]
    print(f"  {name:<12}  {isaac_defaults[i]:>12.4f}  {gt_defaults[i]:>12.4f}  {delta:>+12.4f}")

print("\n" + "=" * 70)

# ── 9. Cleanup ────────────────────────────────────────────────────────────────
env.close()
simulation_app.close()
