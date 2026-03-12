# TODO / Open Concerns

## Status Table

| Issue | Status | Priority |
|---|---|---|
| Shape mismatch (48→36D slice in `anymal_dataset.py`) | **Fixed** | — |
| Double normalization removed (`eval_isaac_v2.py`) | **Fixed** | — |
| `unscale_observations()` called before slicing (Fix E) | **Fixed** | — |
| Wandb config logged once per run | **Fixed** | — |
| Skip first rollout at epoch=0 | **Fixed** | — |
| `default_joint_angles` IG dict wrong for RF/LH joints (HFE+KFE) | **Fixed** | P0 |
| Initial pose distribution shift (IG default ≠ GT default) | **Fixed** | P1 |
| `gt_default_joint_angles` sign + magnitude in `offline_dataset.hdf5` | **Uncertain — unverified** | P2 |
| KFE defaults in GT (assumed same as IG ±0.8) | **Uncertain — unverified** | P2 |
| `offline_dataset_pp.hdf5` exact preprocessing | **Unknown** | P3 |
| Physics/contact sim-to-sim gap | **Unresolved** | P3 |

---

## P2 — GT Default Joint Angles (`isaac_compatibility.py`)

### Sign convention assumed from URDF (not verified against raw dataset)
`gt_default_joint_angles` now uses the ANYmal URDF mirrored convention
(left legs: positive HFE/KFE, right legs: negative) inferred from the live
Isaac Gym `env.default_dof_pos` inspection. This has not been verified
against `offline_dataset.hdf5` directly. To confirm, check the mean joint
positions of the raw dataset:
```python
import h5py, numpy as np
with h5py.File("offline_dataset.hdf5") as f:
    print(np.mean(f["observations"][:, 12:24], axis=0))  # joint pos dims
```
Expected (if convention is correct): LF_HFE≈+0.84, RF_HFE≈-0.84,
LH_HFE≈+0.59, RH_HFE≈-0.59.

### KFE defaults unverified for GT
`gt_default_joint_angles` assumes KFE magnitudes are the same as Isaac Gym
(±0.8 rad). These have not been derived from the Grand Tour dataset. The same
raw mean check above will reveal the actual GT KFE defaults.

---

## P3 — `offline_dataset_pp.hdf5` Format Unknown
The post-processed dataset format is unknown — no preprocessing script was
found. If `offline_dataset_pp.hdf5` is ever used again, the observation format
must be confirmed before deciding whether to apply `unscale_observations()`.
From `compare_dataset_stats.py`, joint position means are ~0, consistent with
centering around GT defaults, but other transforms may also be applied.

---

## Minor / Housekeeping

### `normalize` Parameter is Dead Code (`eval_isaac_v2.py`)
The `normalize` flag still loads dataset stats (`state_mean`, `state_std`) but
the normalization block was removed (Fix B). Either remove the parameter and
dataset loading entirely, or re-add intentionally.

### Single-Instance Obs Table Off-By-One (`eval_isaac_v2.py`)
`obs_isaacgym_format_scaled` is captured after `env.step()`, so it is one step
behind `obs_gt_format_unscaled`. The first Isaac Gym obs (from
`env.get_observations()` before the loop) is not captured in Isaac Gym format.
Minor, but worth being aware of when comparing the two tables side by side.
