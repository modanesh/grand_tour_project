from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym

from legged_gym.envs import *
from legged_gym.utils import (
    get_args,
    export_policy_as_jit,
    task_registry,
    Logger,
    export_policy_as_onnx,
)
import numpy as np
import torch

from tqdm import tqdm

from reward import rewards
from utils import compute_mean_std, load_hdf5_dataset
from grandtour_compatibility import unscale_observations
from isaac_compatibility import make_actions_compatible

from rich import print as rprint

task_name = "anymal_d_flat"
env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
"""
print(env_cfg)
print(dir(env_cfg))
print(env_cfg.env)
print(dir(env_cfg.env))
print(vars(env_cfg.env))
print(env_cfg.sim)
print(dir(env_cfg.sim))
print(env_cfg.init_state)
"""


print(dir(env_cfg.normalization))

rprint(f"action scale: {env_cfg.control.action_scale}")
rprint(f"joint angles:")
rprint(env_cfg.init_state.default_joint_angles)
rprint(f"normalization: {env_cfg.normalization.obs_scales.dof_pos}")
