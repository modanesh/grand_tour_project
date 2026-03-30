from isaaclab.app import AppLauncher
from argparse import ArgumentParser
import torch
import os
import cv2

parser = ArgumentParser()
parser.add_argument("--enable_cameras", action="store_true", default=True)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(
    headless=True,
    enable_cameras=True,
)
simulation_app = app_launcher.app

import isaaclab
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm

import matplotlib.pyplot as plt
import numpy as np
import isaaclab.envs.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import (
    AnymalDFlatEnvCfg,
)

print("isaaclab path:", isaaclab.__file__)
print("isaaclab version:", isaaclab.__version__)


# @configclass overrides the base class
@configclass
class AnymalDFlatCameraEnvCfg(AnymalDFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # add tiled camera to the existing scene config
        self.scene.tiled_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(-7.0, 0.0, 3.0),
                rot=(0.9945, 0.0, 0.1045, 0.0),
                convention="world",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 20.0),
            ),
            width=640,
            height=480,
        )

        # optional: nicer viewer pose
        self.viewer.eye = (7.0, 0.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.8)


        ########### OBSERVATION AND ACTION OVERRIDE #########
        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos,
            scale=1.0
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=mdp.joint_vel,
            scale=1.0
        )

        # position control instead of torque/effort control
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
        )

        # if the original config had another action term like joint_effort,
        # remove it so only joint position actions remain
        if hasattr(self.actions, "joint_effort"):
            self.actions.joint_effort = None
        if hasattr(self.actions, "torques"):
            self.actions.torques = None


env_cfg = AnymalDFlatCameraEnvCfg()
env_cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg=env_cfg)

obs, info = env.reset()

print(info)

# Create imgs directory if it doesn't exist
os.makedirs("imgs", exist_ok=True)

actions = torch.zeros_like(env.action_manager.action)
obs, rew, terminated, truncated, info = env.step(actions)


for i in range(100):
    actions = 0 * torch.ones_like(env.action_manager.action)
    actions[:, 8] = 0.5  # set first joint to 1.0
    actions[:, 9] = 0.5  # set second joint to 1.0
    actions[:, 10] = 0.5 # set third joint to 1.0
    actions[:, 11] = 0.5  # set fourth joint to 1.0


    if i == 0:
        print(f"actions shape: {actions.shape}")
    
    # actions = obs["policy"][0][12:24]
    # actions = actions.reshape(1, -1)
    obs, rew, terminated, truncated, info = env.step(actions)

    # target_pos = 1.0 * torch.ones_like(env.scene["robot"].data.joint_pos)
    # target_vel = torch.zeros_like(env.scene["robot"].data.joint_vel)
    

    # # Teleport the joints to the target state
    # env.scene["robot"].write_joint_state_to_sim(target_pos, target_vel)

    # obs, rew, terminated, truncated, info = env.step(torch.zeros_like(env.action_manager.action))


    if i%2==0:
        rgb = env.scene["tiled_camera"].data.output["rgb"]

        img = rgb[0].cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        print(f"step {i}: rgb shape = {rgb.shape}")
        # add green text in the top left corner of the image
        cv2.putText(
            img,
            f"step {i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),  # green (BGR)
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            f"obs: {str(obs['policy'][0][12:24])}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),  # green (BGR)
            2,
            cv2.LINE_8,
        )

        print(f"actions: {(actions[0])}")
        print(f"obs cmd: {obs['policy'][0][9:12]}")
        print(f"obs pos: {obs['policy'][0][12:24]}")
        print(f"obs vel: {obs['policy'][0][24:36]}")
        print(f"obs act: {obs['policy'][0][36:]}")
        print(f"rew: {rew}")
        print()

        cv2.imwrite(f"imgs/demo_anymal_d_flat_{i}.png", img)
    if terminated:
        break


# Print the order of joints the robot asset uses
print("Robot Joint Names:", env.scene["robot"].joint_names)

# # Print the indices the Action Manager is controlling
# print("Action Joint Indices:", env.action_manager._action_terms["joint_pos"].joint_ids)


# env.export_IO_descriptors(output_dir="./io_descriptors_output")


# io_descriptors = env.get_IO_descriptors()
# print(io_descriptors)

env.close()
simulation_app.close()
