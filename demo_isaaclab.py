from isaaclab.app import AppLauncher
from argparse import ArgumentParser
import torch
import os
import cv2
import tqdm

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

# read joint positions from ./anymal_state_odometry/pose_pos
import zarr

# grand tour reference
# Joint Naming 0-11: ['LF_HAA', 'LF_HFE', 'LF_KFE', 'RF_HAA', 'RF_HFE', 'RF_KFE', 'LH_HAA', 'LH_HFE', 'LH_KFE', 'RH_HAA', 'RH_HFE', 'RH_KFE']
# https://github.com/leggedrobotics/grand_tour_dataset/blob/main/examples_hugging_face/notebooks/explore.ipynb

grand_tour_ref_keys_order = ['LF_HAA', 'LF_HFE', 'LF_KFE', 'RF_HAA', 'RF_HFE', 'RF_KFE', 'LH_HAA', 'LH_HFE', 'LH_KFE', 'RH_HAA', 'RH_HFE', 'RH_KFE']
grand_tour_dict = dict()


keys_ = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
joint_positions_all = []
for key_idx, joint_name in zip(keys_, grand_tour_ref_keys_order):
    z = zarr.open(f"./data/anymal_state_actuator/{key_idx}_state_joint_position", mode="r")
    print(f"{joint_name} {key_idx}: joint_position shape:", z.shape) # shape is (nrows)
    grand_tour_dict[joint_name] = z[:]

isaac_lab_ref_keys_order = ['LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA', 'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE', 'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']

# concatenate as shape (12, nrows)
grand_tour_joint_positions = np.zeros((12, len(grand_tour_dict[grand_tour_ref_keys_order[0]])))
# concatenate in order of isaac lab order
for i, joint_name in enumerate(isaac_lab_ref_keys_order):
    grand_tour_joint_positions[i] = grand_tour_dict[joint_name]

print("grand_tour_joint_positions shape:", grand_tour_joint_positions.shape)
# print("joint_position dtype:", z.dtype)
# print("joint_position chunks:", z.chunks)
# joint_positions = z[:]
# print("joint_positions shape:", joint_positions.shape)
# print("joint_positions dtype:", joint_positions.dtype)
# print("joint_positions chunks:", joint_positions.chunks)
# print("joint_positions[0]:", joint_positions[0])
# raise KeyboardInterrupt


# # load hdf5 and store in variables
# import h5py

# grand_tour_observations = None
# grand_tour_actions = None
# grand_tour_next_observations = None
# grand_tour_rewards = None
# grand_tour_terminals = None

# with h5py.File("offline_dataset_pp.hdf5", "r") as f:
#     print("Keys:", list(f.keys()))
#     for key in f.keys():
#         print(f"{key}: {f[key].shape}")
#         if key == "observations":
#             grand_tour_observations = f[key][:]
#         elif key == "actions":
#             grand_tour_actions = f[key][:]
#         elif key == "next_observations":
#             grand_tour_next_observations = f[key][:]
#         elif key == "rewards":
#             grand_tour_rewards = f[key][:]
#         elif key == "terminals":
#             grand_tour_terminals = f[key][:]

# print("grand_tour_observations shape:", grand_tour_observations.shape)
# print("grand_tour_actions shape:", grand_tour_actions.shape)
# print("grand_tour_next_observations shape:", grand_tour_next_observations.shape)
# print("grand_tour_rewards shape:", grand_tour_rewards.shape)
# print("grand_tour_terminals shape:", grand_tour_terminals.shape)

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

cumulative_rewards = torch.zeros(env.num_envs, device=env.device)

obs, info = env.reset()

print(info)

# Create imgs directory if it doesn't exist
os.makedirs("imgs", exist_ok=True)

# actions = torch.zeros_like(env.action_manager.action)
# obs, rew, terminated, truncated, info = env.step(actions)

# cumulative reward -> tqdm pbar label dynamically
for i in tqdm.trange(2000, 5000, desc=f"Cumulative Reward: {cumulative_rewards[0].item()}"):
    actions = torch.zeros_like(env.action_manager.action)
    # actions[:, 8] = 0.25  # set first joint to 1.0
    # actions[:, 9] = 0.25  # set second joint to 1.0
    # actions[:, 10] = 0.25 # set third joint to 1.0
    # actions[:, 11] = 0.25  # set fourth joint to 1.0

    # override all joints with the grand tour actions
    curr_grand_tour_pos = grand_tour_joint_positions[:, i]
    actions[:, :] = torch.tensor(curr_grand_tour_pos, device=env.device, dtype=torch.float32)


    if i == 0:
        print(f"actions shape: {actions.shape}")
        print(f"step {i}: rgb shape = {env.scene['tiled_camera'].data.output['rgb'].shape}")

    # actions = obs["policy"][0][12:24]
    # actions = actions.reshape(1, -1)
    obs, rew, terminated, truncated, info = env.step(actions)
    cumulative_rewards += rew

    # Update the tqdm description with current cumulative reward
    # tqdm.write(f"Cumulative Reward: {cumulative_rewards[0].item()}")
    # target_pos = 1.0 * torch.ones_like(env.scene["robot"].data.joint_pos)
    # target_vel = torch.zeros_like(env.scene["robot"].data.joint_vel)
    

    # # Teleport the joints to the target state
    # env.scene["robot"].write_joint_state_to_sim(target_pos, target_vel)

    # obs, rew, terminated, truncated, info = env.step(torch.zeros_like(env.action_manager.action))


    pbar.set_description(f"Cumulative Reward: {cumulative_rewards[0].item()}")
    pbar.refresh()  # Force update display

    if i%1==0:
        rgb = env.scene["tiled_camera"].data.output["rgb"]

        img = rgb[0].cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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

        # cv2.putText(
        #     img,
        #     f"obs: {str(obs['policy'][0][12:24])}",
        #     (10, 60),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 255, 0),  # green (BGR)
        #     2,
        #     cv2.LINE_8,
        # )

        # cv2 puttext cumulative reward
        cv2.putText(
            img,
            f"cumulative reward: {round(cumulative_rewards[0].item(), 3)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),  # green (BGR)
            2,
            cv2.LINE_8,
        )

        # print(f"actions: {(actions[0])}")
        # print(f"obs cmd: {obs['policy'][0][9:12]}")
        # print(f"obs pos: {obs['policy'][0][12:24]}")
        # print(f"obs vel: {obs['policy'][0][24:36]}")
        # print(f"obs act: {obs['policy'][0][36:]}")
        # print(f"rew: {rew}")
        # print()

        cv2.imwrite(f"imgs/demo_anymal_d_flat_{i}.png", img)
    # if terminated:
    #     break


# make mp4 movie out of frames inline using cv2
import cv2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('demo_anymal_d_flat.mp4', fourcc, 30.0, (640, 480))
for i in range(2000, 5000):
    img = cv2.imread(f"imgs/demo_anymal_d_flat_{i}.png")
    out.write(img)
out.release()

# Print the order of joints the robot asset uses
print("Robot Joint Names:", env.scene["robot"].joint_names)
# Robot Joint Names: ['LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA', 'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE', 'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']

print("Cumulative Rewards:", cumulative_rewards)

# # Print the indices the Action Manager is controlling
# print("Action Joint Indices:", env.action_manager._action_terms["joint_pos"].joint_ids)


# env.export_IO_descriptors(output_dir="./io_descriptors_output")


# io_descriptors = env.get_IO_descriptors()
# print(io_descriptors)

env.close()
simulation_app.close()
