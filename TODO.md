# TODO / Open Concerns

## Grand Tour Default Joint Angles (`isaac_compatibility.py`)

### KFE defaults unverified
`gt_default_joint_angles` assumes KFE joints are the same as Isaac Gym defaults:
```
LF_KFE: -0.8,  LH_KFE: 0.8,  RF_KFE: -0.8,  RH_KFE: 0.8
```
These were not derived from the Grand Tour dataset — they were assumed. Run
`compare_dataset_stats.py` on `offline_dataset_pp.hdf5` to verify the actual
mean joint positions at neutral stance, and update if needed.

### HFE defaults assumed from memory
The GT HFE values (`LF/RF_HFE: 0.84`, `LH/RH_HFE: -0.59`) came from prior
notes, not from fresh dataset analysis. Should be confirmed against the actual
dataset mean action/obs values.

## Fundamental Distribution Shift (Isaac Gym vs Grand Tour)

Isaac Gym initialises the robot at its own defaults (LF_HFE=0.4), not Grand
Tour defaults (LF_HFE=0.84). `make_actions_compatible` and
`unscale_previous_actions` must use IG defaults because Isaac Gym internally
applies `target = ig_default + action * action_scale`. Using GT defaults here
shifts every commanded position by `ig_default - gt_default` (~0.44 rad for
HFE), causing the robot to collapse immediately (reward=0).

`unscale_joint_pos` uses GT defaults as a pragmatic hack to shift dof_pos
observations into the GT reference frame (so they look like training data).
The correct long-term fix would be to set Isaac Gym's `default_dof_pos` in the
env config to match Grand Tour defaults, making the action and observation
reference frames consistent without any hacks.

## `normalize` Parameter is Now Dead Code (`eval_isaac_v2.py`)

The `normalize` flag in `OnlineEval.__init__` still loads dataset stats
(`state_mean`, `state_std`) but the normalization block in `eval_actor_isaac`
was removed (Fix B). Either:
- Remove the `normalize` parameter and the dataset loading entirely, or
- Decide if external normalization is ever needed and re-add intentionally.

## Wandb Config Logged Every Rollout (`eval_isaac_v2.py`)

`wandb.config.update(self._wandb_cfg)` is called at the start of every
`eval_actor_isaac` call (every `rollout_every` epochs). It's harmless but
wasteful. Add a `self._wandb_cfg_logged` flag to log only once:
```python
if wandb.run is not None and not self._wandb_cfg_logged:
    wandb.config.update(self._wandb_cfg, allow_val_change=True)
    self._wandb_cfg_logged = True
```

## Single-Instance Obs Table Off-By-One (`eval_isaac_v2.py`)

`obs_isaacgym_format_scaled` is captured after `env.step()`, so it is one step
behind `obs_gt_format_unscaled`. The very first Isaac Gym obs (from
`env.get_observations()` before the loop) is not captured in Isaac Gym format.
Minor, but worth being aware of when comparing the two tables side by side.
