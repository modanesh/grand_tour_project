from isaaclab.app import AppLauncher
from isaaclab_tasks.utils import parse_env_cfg

app_launcher = AppLauncher(headless=True)

TASK_NAME = "Isaac-Velocity-Flat-Anymal-D-v0"

env_cfg = parse_env_cfg(TASK_NAME, device="cuda:0", num_envs=2)
