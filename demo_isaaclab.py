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
from sklearn.linear_model import LinearRegression

print("isaaclab path:", isaaclab.__file__)
print("isaaclab version:", isaaclab.__version__)

import zarr


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


from src.dataloader import GrandTourDataloader

dataloader = GrandTourDataloader()
X_data = dataloader.get_observations_isaac_lab_format()
Y_data = dataloader.get_actions_isaac_lab_format()
print("X_data shape:", X_data.shape)
print("Y_data shape:", Y_data.shape)


CLASSIFIER = "linear_regression"

if CLASSIFIER == "linear_regression":
    model = LinearRegression()
    model.fit(X_data[:], Y_data[:])
    print("RMSE: ", np.sqrt(np.mean((model.predict(X_data[:]) - Y_data[:]) ** 2)))
else:
    model = DiffuseLocoModel().to(device)


# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if CLASSIFIER == "linear_regression":
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
            t = torch.randint(
                0, model.diffusion_steps, (batch_obs.shape[0],), device=device
            )

            # Forward pass
            pred_next = model(batch_obs, t)
            loss = criterion(pred_next, batch_next)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

    print("DiffuseLoco training completed!")

    # Test the trained model
    model.eval()
    with torch.no_grad():
        test_obs = torch.FloatTensor(X_data[:5]).to(device)
        test_t = torch.zeros(5, device=device)
        pred_joints = model(test_obs, test_t)
        print("Sample predictions:", pred_joints.cpu().numpy())

    # Save the trained model
    torch.save(model.state_dict(), "diffuseloco_model.pth")
    print("Model saved as 'diffuseloco_model.pth'")


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
        self.observations.policy.joint_pos = ObsTerm(func=mdp.joint_pos, scale=1.0)
        self.observations.policy.joint_vel = ObsTerm(func=mdp.joint_vel, scale=1.0)

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
for i in tqdm.trange(0, 1000, desc=f"Cumulative Reward: {cumulative_rewards[0].item()}"):
    actions = torch.zeros_like(env.action_manager.action)
    # actions = torch.tensor(Y_data[2000:2001], device=env.device, dtype=torch.float32)
    # actions[:, 8] = 0.25  # set first joint to 1.0
    # actions[:, 9] = 0.25  # set second joint to 1.0
    # actions[:, 10] = 0.25 # set third joint to 1.0
    # actions[:, 11] = 0.25  # set fourth joint to 1.0

    if CLASSIFIER == "linear_regression":
        print(f"model: {model}")
        obs_first_36_features = obs["policy"][:, :36]
        actions_pred = model.predict(obs_first_36_features.cpu().numpy())
        actions_pred = torch.tensor(
            actions_pred, device=env.device, dtype=torch.float32
        )

        # print(f"model prediction: {model.predict(X_data[2000:2001])}")
        # print(f"Y_data: {Y_data[1000:1001]}")

        # print(f"actions_pred: {actions_pred}")

        actions = actions_pred

        # print(f"obs: {obs}")
        print(f"actions: {actions}")
        print()

    else:
        # override all joints with the grand tour actions
        curr_grand_tour_pos = non_translated_grand_tour_joint_positions[:, i]
        # run diffuseloco model inference obs -> model -> action
        curr_obs = obs["policy"].clone().detach()
        curr_obs = curr_obs.to(device=device, dtype=torch.float32)
        # Only use first 36 dimensions (exclude prev_actions)
        curr_obs = curr_obs[:, :36]
        curr_action = model.predict(curr_obs.cpu().numpy())
        curr_grand_tour_pos = curr_action[0]

        actions[:, :] = torch.tensor(
            curr_grand_tour_pos, device=env.device, dtype=torch.float32
        )

        if i == 0:
            print(f"actions shape: {actions.shape}")
            print(
                f"step {i}: rgb shape = {env.scene['tiled_camera'].data.output['rgb'].shape}"
            )

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

    if i % 1 == 0:
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

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("demo_anymal_d_flat.mp4", fourcc, 30.0, (640, 480))
for i in range(0, 1000):
    img = cv2.imread(f"imgs/demo_anymal_d_flat_{i}.png")
    out.write(img)
out.release()

# # Print the order of joints the robot asset uses
# print("Robot Joint Names:", env.scene["robot"].joint_names)
# # Robot Joint Names: ['LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA', 'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE', 'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']

print("Cumulative Rewards:", cumulative_rewards)


# print(f"sample joint position: {grand_tour_joint_positions[10000]}")

# # Print the indices the Action Manager is controlling
# print("Action Joint Indices:", env.action_manager._action_terms["joint_pos"].joint_ids)


# env.export_IO_descriptors(output_dir="./io_descriptors_output")


# io_descriptors = env.get_IO_descriptors()
# print(io_descriptors)

env.close()
simulation_app.close()
