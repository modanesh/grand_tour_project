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
from sklearn.ensemble import RandomForestRegressor

print("isaaclab path:", isaaclab.__file__)
print("isaaclab version:", isaaclab.__version__)

# read joint positions from ./anymal_state_odometry/pose_pos
import zarr

CLASSIFIER = "random_forest"

# grand tour reference
# Joint Naming 0-11: ['LF_HAA', 'LF_HFE', 'LF_KFE', 'RF_HAA', 'RF_HFE', 'RF_KFE', 'LH_HAA', 'LH_HFE', 'LH_KFE', 'RH_HAA', 'RH_HFE', 'RH_KFE']
# https://github.com/leggedrobotics/grand_tour_dataset/blob/main/examples_hugging_face/notebooks/explore.ipynb

# load timestamps

# for anymal_state_actuator
z = zarr.open(f"./data/anymal_state_actuator/timestamp", mode="r")
print(f"anymal_state_actuator timestamps shape:", z.shape) # shape is (nrows)
grand_tour_actuator_timestamps = z[:]
print(f"anymal_state_actuator timestamps:", grand_tour_actuator_timestamps[:10])

# for anymal_state_odometry
z = zarr.open(f"./data/anymal_state_odometry/timestamp", mode="r")
print(f"anymal_state_odometry timestamps shape:", z.shape) # shape is (nrows)
grand_tour_odometry_timestamps = z[:]
print(f"anymal_state_odometry timestamps:", grand_tour_odometry_timestamps[:10])

# for command twist
z = zarr.open(f"./data/anymal_command_twist/timestamp", mode="r")
print(f"anymal_command_twist timestamps shape:", z.shape) # shape is (nrows)
grand_tour_command_timestamps = z[:]
print(f"anymal_command_twist timestamps:", grand_tour_command_timestamps[:10])

# load data

grand_tour_ref_keys_order = ['LF_HAA', 'LF_HFE', 'LF_KFE', 'RF_HAA', 'RF_HFE', 'RF_KFE', 'LH_HAA', 'LH_HFE', 'LH_KFE', 'RH_HAA', 'RH_HFE', 'RH_KFE']
grand_tour_dict_joint_positions = dict()
grand_tour_dict_joint_velocities = dict()


keys_ = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
joint_positions_all = []
for key_idx, joint_name in zip(keys_, grand_tour_ref_keys_order):
    z = zarr.open(f"./data/anymal_state_actuator/{key_idx}_state_joint_position", mode="r")
    print(f"{joint_name} {key_idx}: joint_position shape:", z.shape) # shape is (nrows)
    grand_tour_dict_joint_positions[joint_name] = z[:]
    
    z = zarr.open(f"./data/anymal_state_actuator/{key_idx}_state_joint_velocity", mode="r")
    print(f"{joint_name} {key_idx}: joint_velocity shape:", z.shape) # shape is (nrows)
    grand_tour_dict_joint_velocities[joint_name] = z[:]

z = zarr.open(f"./data/anymal_state_odometry/twist_lin", mode="r")
print(f"base_lin_vel shape:", z.shape) # shape is (nrows)
grand_tour_linear_velocities = z[:]

# load pose orientation from anymal state odometry
z = zarr.open(f"./data/anymal_state_odometry/pose_orien", mode="r")
print(f"pose_orientation shape:", z.shape) # shape is (nrows)
grand_tour_pose_orientation = z[:]
# convert quaternion to unit vector (xyz)
print(f"check that the quaternion norm is 1:", np.linalg.norm(grand_tour_pose_orientation[0]))
grand_tour_pose_orientation_xyz = grand_tour_pose_orientation[:, 1:] / np.linalg.norm(grand_tour_pose_orientation[:, 1:], axis=1, keepdims=True)
print(f"pose_orientation unit vector shape:", grand_tour_pose_orientation_xyz.shape)
print(f"check that the unit vector norm is 1:", np.linalg.norm(grand_tour_pose_orientation_xyz[0]))
print(f"first sample pose orientation xyz:", grand_tour_pose_orientation_xyz[0])
print(f"1000th sample pose orientation xyz:", grand_tour_pose_orientation_xyz[1000])


z = zarr.open(f"./data/anymal_state_odometry/twist_ang", mode="r")
print(f"base_ang_vel shape:", z.shape) # shape is (nrows)
grand_tour_angular_velocities = z[:]

z = zarr.open(f"./data/anymal_command_twist/linear", mode="r")
print(f"linear_velocity_commands shape:", z.shape) # shape is (nrows)
grand_tour_linear_velocity_commands = z[:]



isaac_lab_ref_keys_order = ['LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA', 'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE', 'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']

# concatenate as shape (12, nrows)
grand_tour_joint_positions = np.zeros((12, len(grand_tour_dict_joint_positions[grand_tour_ref_keys_order[0]])))
# concatenate in order of isaac lab order
for i, joint_name in enumerate(isaac_lab_ref_keys_order):
    grand_tour_joint_positions[i] = grand_tour_dict_joint_positions[joint_name]

grand_tour_joint_velocities = np.zeros((12, len(grand_tour_dict_joint_velocities[grand_tour_ref_keys_order[0]])))
# concatenate in order of isaac lab order
for i, joint_name in enumerate(isaac_lab_ref_keys_order):
    grand_tour_joint_velocities[i] = grand_tour_dict_joint_velocities[joint_name]

non_translated_grand_tour_joint_positions = grand_tour_joint_positions.copy()

grand_tour_joint_positions = grand_tour_joint_positions.T
grand_tour_joint_velocities = grand_tour_joint_velocities.T

print("grand_tour_angular_velocities shape:", grand_tour_angular_velocities.shape)
print("grand_tour_linear_velocities shape:", grand_tour_linear_velocities.shape)
print("grand_tour_joint_positions shape:", grand_tour_joint_positions.shape)
print("grand_tour_joint_velocities shape:", grand_tour_joint_velocities.shape)




# raise KeyboardInterrupt
"""
[INFO] Command Manager:  <CommandManager> contains 1 active terms.
+------------------------------------------------+
|              Active Command Terms              |
+-------+---------------+------------------------+
| Index | Name          |          Type          |
+-------+---------------+------------------------+
|   0   | base_velocity | UniformVelocityCommand |
+-------+---------------+------------------------+

[INFO] Recorder Manager:  <RecorderManager> contains 0 active terms.
+---------------------+
| Active Recorder Terms |
+-----------+---------+
|   Index   | Name    |
+-----------+---------+
+-----------+---------+

[INFO] Action Manager:  <ActionManager> contains 1 active terms.
+------------------------------------+
|  Active Action Terms (shape: 12)   |
+--------+-------------+-------------+
| Index  | Name        |   Dimension |
+--------+-------------+-------------+
|   0    | joint_pos   |          12 |
+--------+-------------+-------------+

[INFO] Observation Manager: <ObservationManager> contains 1 groups.
+---------------------------------------------------------+
| Active Observation Terms in Group: 'policy' (shape: (48,)) |
+-----------+---------------------------------+-----------+
|   Index   | Name                            |   Shape   |
+-----------+---------------------------------+-----------+
|     0     | base_lin_vel                    |    (3,)   |
|     1     | base_ang_vel                    |    (3,)   |
|     2     | projected_gravity               |    (3,)   |
|     3     | velocity_commands               |    (3,)   |
|     4     | joint_pos                       |   (12,)   |
|     5     | joint_vel                       |   (12,)   |
|     6     | actions                         |   (12,)   |
+-----------+---------------------------------+-----------+
"""


# full obs shape is (1, 57)
# [0:3] base_lin_vel (loaded from state odometry)
# [3:6] base_ang_vel (loaded from state odometry)
# [6:9] projected_gravity (xyz-unit-vector)
# [9:12] velocity_commands (loaded from command twist)
# [12:24] joint_pos (loaded from state actuator)
# [24:36] joint_vel (loaded from state actuator)


# naive implementation: interpolate all data to have total of 5000 points from start to end
grand_tour_linear_velocities_interpolated = np.linspace(grand_tour_linear_velocities[0], grand_tour_linear_velocities[-1], 5000)
grand_tour_angular_velocities_interpolated = np.linspace(grand_tour_angular_velocities[0], grand_tour_angular_velocities[-1], 5000)
grand_tour_pose_orientation_xyz_interpolated = np.linspace(grand_tour_pose_orientation_xyz[0], grand_tour_pose_orientation_xyz[-1], 5000)
grand_tour_linear_velocity_commands_interpolated = np.linspace(grand_tour_linear_velocity_commands[0], grand_tour_linear_velocity_commands[-1], 5000)
grand_tour_joint_positions_interpolated = np.linspace(grand_tour_joint_positions[0], grand_tour_joint_positions[-1], 5000)
grand_tour_joint_velocities_interpolated = np.linspace(grand_tour_joint_velocities[0], grand_tour_joint_velocities[-1], 5000)


print(f"shape of interploated linear velocity: {grand_tour_linear_velocities_interpolated.shape}")
print(f"shape of interploated angular velocity: {grand_tour_angular_velocities_interpolated.shape}")
print(f"shape of interploated pose orientation xyz: {grand_tour_pose_orientation_xyz_interpolated.shape}")
print(f"shape of interploated linear velocity commands: {grand_tour_linear_velocity_commands_interpolated.shape}")
print(f"shape of interploated joint positions: {grand_tour_joint_positions_interpolated.shape}")
print(f"shape of interploated joint velocities: {grand_tour_joint_velocities_interpolated.shape}")


raw_data = np.concatenate([
    grand_tour_linear_velocities_interpolated,
    grand_tour_angular_velocities_interpolated,
    grand_tour_pose_orientation_xyz_interpolated,
    grand_tour_linear_velocity_commands_interpolated,
    grand_tour_joint_positions_interpolated,
    grand_tour_joint_velocities_interpolated,
], axis=1)

X_data = raw_data[:-1]
Y_data = raw_data[1:, 12:24]  # Only predict next joint positions (12 dimensions)

print("X_data shape:", X_data.shape)
print("Y_data shape:", Y_data.shape)

# train diffuseloco on next joint prediction
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class GrandTourDataset(Dataset):
    def __init__(self, observations, next_observations):
        self.observations = torch.FloatTensor(observations)
        self.next_observations = torch.FloatTensor(next_observations)
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.next_observations[idx]

class DiffuseLocoModel(nn.Module):
    def __init__(self, input_dim=36, output_dim=12, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        
        self.diffusion_steps = 100
        self.noise_scheduler = nn.ModuleList([
            nn.Linear(hidden_dim//2 + 1, hidden_dim//2 + 1) for _ in range(self.diffusion_steps)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim//2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, t):
        encoded = self.encoder(x)
        t_embed = t.unsqueeze(-1).float()
        combined = torch.cat([encoded, t_embed], dim=-1)
        
        # Apply noise scheduler layers sequentially (simplified diffusion)
        for layer in self.noise_scheduler:
            combined = F.relu(layer(combined))
        
        return self.decoder(combined)

# Create dataset and dataloader
dataset = GrandTourDataset(X_data, Y_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

if CLASSIFIER == "random_forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_data[2000:3000], Y_data[2000:3000])
else:
    model = DiffuseLocoModel().to(device)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if CLASSIFIER == "random_forest":
    optimizer = None
    criterion = None
else:
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Training loop
    print("Starting DiffuseLoco training...")
    num_epochs = 2
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_obs, batch_next in dataloader:
            batch_obs = batch_obs.to(device)
            batch_next = batch_next.to(device)
            
            # Sample random diffusion timestep
            t = torch.randint(0, model.diffusion_steps, (batch_obs.shape[0],), device=device)
            
            # Forward pass
            pred_next = model(batch_obs, t)
            loss = criterion(pred_next, batch_next)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    print("DiffuseLoco training completed!")

    # Test the trained model
    model.eval()
    with torch.no_grad():
        test_obs = torch.FloatTensor(X_data[:5]).to(device)
        test_t = torch.zeros(5, device=device)
        pred_joints = model(test_obs, test_t)
        print("Sample predictions:", pred_joints.cpu().numpy())

    # Save the trained model
    torch.save(model.state_dict(), 'diffuseloco_model.pth')
    print("Model saved as 'diffuseloco_model.pth'")





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
for i in tqdm.trange(0,100, desc=f"Cumulative Reward: {cumulative_rewards[0].item()}"):
    actions = torch.zeros_like(env.action_manager.action)
    # actions[:, 8] = 0.25  # set first joint to 1.0
    # actions[:, 9] = 0.25  # set second joint to 1.0
    # actions[:, 10] = 0.25 # set third joint to 1.0
    # actions[:, 11] = 0.25  # set fourth joint to 1.0


    if CLASSIFIER == "random_forest":
        print(f"model: {model}")
        obs_first_36_features = obs["policy"][:, :36]
        # actions = model.predict(obs_first_36_features.cpu().numpy())
        # actions = torch.tensor(actions, device=env.device, dtype=torch.float32)

        # print(f"obs: {obs}")
        print(f"actions: {actions}")
        
    else:

        # override all joints with the grand tour actions
        curr_grand_tour_pos = non_translated_grand_tour_joint_positions[:, i]
        # run diffuseloco model inference obs -> model -> action
        curr_obs = obs["policy"].clone().detach()
        curr_obs = curr_obs.to(device=device, dtype=torch.float32)
        # Only use first 36 dimensions (exclude prev_actions)
        curr_obs = curr_obs[:, :36]
        curr_action = model(curr_obs, torch.tensor([0], device=device))
        curr_grand_tour_pos = curr_action.detach().cpu().numpy()[0]


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


    # pbar.set_description(f"Cumulative Reward: {cumulative_rewards[0].item()}")
    # pbar.refresh()  # Force update display

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
for i in range(0, 100):
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
