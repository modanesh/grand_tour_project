from isaaclab.app import AppLauncher
from argparse import ArgumentParser
import torch
import os
import cv2
import tqdm
import math

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
import torch.nn as nn
import torch.optim as optim

print("isaaclab path:", isaaclab.__file__)
print("isaaclab version:", isaaclab.__version__)

import zarr

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--classifier", type=str, default="linear_regression")
args, _ = parser.parse_known_args()

CLASSIFIER = args.classifier


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


# Simple DDPM Model Implementation
class TransformerNoiseNet(nn.Module):
    """Transformer-based noise prediction network"""

    def __init__(self, input_dim, hidden_dim, action_dim, num_layers=4, num_heads=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection to predict noise for each action dimension
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim) where seq_len=1
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        x = self.transformer(x)  # (batch, seq_len, hidden_dim)
        x = self.output_proj(x)  # (batch, seq_len, action_dim)
        return x.squeeze(1)  # (batch, action_dim)


class SimpleDDPM(nn.Module):
    def __init__(
        self,
        obs_dim=36,
        action_dim=12,
        hidden_dim=256,
        num_layers=4,
        num_timesteps=100,
        use_transformer=False,
        num_heads=8,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_timesteps = num_timesteps
        self.use_transformer = use_transformer

        # Time embedding - using sinusoidal embedding for single timesteps
        self.time_embed_dim = hidden_dim

        if use_transformer:
            # Transformer-based network
            self.network = TransformerNoiseNet(
                input_dim=obs_dim + action_dim + hidden_dim,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                num_layers=num_layers,
                num_heads=num_heads,
            )
        else:
            # MLP network (original)
            layers = []
            layers.append(nn.Linear(obs_dim + action_dim + hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, action_dim))
            self.network = nn.Sequential(*layers)

        # Noise schedule - register as buffers so they move to GPU with model
        beta = torch.linspace(0.0001, 0.02, num_timesteps)
        alpha = 1.0 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)

    def sinusoidal_time_embedding(self, t):
        """Create sinusoidal time embedding for single timestep"""
        half_dim = self.time_embed_dim // 2
        t_normalized = t.float().unsqueeze(-1) / self.num_timesteps  # (batch, 1)

        # Create frequency multipliers using geometric progression
        freqs = torch.exp(
            torch.arange(0, half_dim, dtype=torch.float32, device=t.device)
            * -(math.log(10000.0) / half_dim)
        )  # (half_dim,)

        # Apply frequencies to normalized time
        args = t_normalized * freqs  # (batch, half_dim)

        emb = torch.cat(
            [torch.sin(args), torch.cos(args)], dim=-1
        )  # (batch, time_embed_dim)
        return emb

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t])[:, None, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t])[
            :, None, None
        ]

        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise

    def p_mean_variance(self, x, t):
        # Predict the mean and variance for the reverse process
        pred_noise = self.network(x, t)

        alpha_t = self.alpha[t][:, None, None]
        alpha_cumprod_t = self.alpha_cumprod[t][:, None, None]
        beta_t = self.beta[t][:, None, None]

        # Compute mean
        mean = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise
        )

        return mean

    def forward(self, x, t):
        # Concatenate observation and action
        # x shape: (batch, seq_len, obs_dim + action_dim) for training
        # For inference, x will be just observation
        if x.shape[-1] == self.obs_dim:
            # Inference mode - expand to match expected input
            batch_size = x.shape[0]
            dummy_action = torch.zeros(batch_size, 1, self.action_dim, device=x.device)
            x_expanded = torch.cat(
                [x.unsqueeze(1), dummy_action], dim=1
            )  # (batch, 1, obs+action)
        else:
            # Training mode
            x_expanded = x

        # Get time embedding
        t_embed = self.sinusoidal_time_embedding(t)

        # Concatenate time embedding with input
        combined_input = torch.cat(
            [x_expanded, t_embed.unsqueeze(1).expand(-1, x_expanded.shape[1], -1)],
            dim=-1,
        )

        # Pass through network
        if self.use_transformer:
            # Transformer expects (batch, seq_len, input_dim) and returns (batch, action_dim)
            pred_noise = self.network(combined_input)
        else:
            # MLP expects (batch, seq_len, input_dim) but returns (batch, seq_len, action_dim)
            pred_noise = self.network(combined_input)

        return pred_noise

    def sample(self, obs, num_inference_steps=20):
        """Sample action from observation using DDPM"""
        batch_size = obs.shape[0]
        device = obs.device

        # Start with pure noise
        action = torch.randn(batch_size, 1, self.action_dim, device=device)

        # Use fewer timesteps for inference
        timesteps = torch.linspace(
            self.num_timesteps - 1,
            0,
            num_inference_steps,
            dtype=torch.long,
            device=device,
        )

        for t in timesteps:
            t_batch = t.expand(batch_size)

            # Predict noise - concatenate obs and action for network input
            # obs: (batch, 36) -> (batch, 1, 36)
            # action: (batch, 1, 12) -> (batch, 1, 12)
            # combined: (batch, 1, 48) for network expecting obs_dim + action_dim
            obs_expanded = obs.unsqueeze(1)  # (batch, 1, 36)
            model_input = torch.cat([obs_expanded, action], dim=2)  # (batch, 1, 48)

            # Get time embedding and concatenate with input
            t_embed = self.sinusoidal_time_embedding(t_batch)  # (batch, hidden_dim)
            combined_input = torch.cat(
                [
                    model_input,
                    t_embed.unsqueeze(1).expand(-1, model_input.shape[1], -1),
                ],
                dim=-1,
            )

            with torch.no_grad():
                pred_noise = self.network(combined_input)
                # Transformer returns (batch, action_dim), ensure 3D for broadcasting
                if pred_noise.dim() == 2:
                    pred_noise = pred_noise.unsqueeze(1)  # (batch, 1, action_dim)

            # Compute mean
            alpha_t = self.alpha[t_batch][:, None, None]
            alpha_cumprod_t = self.alpha_cumprod[t_batch][:, None, None]
            beta_t = self.beta[t_batch][:, None, None]

            mean = (1 / torch.sqrt(alpha_t)) * (
                action - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise
            )

            # Add noise for all but final step
            if t > 0:
                noise = torch.randn_like(action)
                action = mean + torch.sqrt(beta_t) * noise
            else:
                action = mean

        return action.squeeze(1)  # Remove seq dimension


# CLASSIFIER = "linear_regression"

# Initialize device first
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if CLASSIFIER == "linear_regression":
    model = LinearRegression()
    model.fit(X_data[:], Y_data[:])
    print("RMSE: ", np.sqrt(np.mean((model.predict(X_data[:]) - Y_data[:]) ** 2)))
elif CLASSIFIER == "ddpm":
    model = SimpleDDPM(
        obs_dim=36,
        action_dim=12,
        hidden_dim=256,
        num_layers=4,
        num_timesteps=100,
        use_transformer=True,
        num_heads=8,
    ).to(device)
else:
    model = DiffuseLocoModel().to(device)
if CLASSIFIER == "linear_regression":
    optimizer = None
    criterion = None
elif CLASSIFIER == "ddpm":
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
else:
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Training loop
    print(f"Starting {CLASSIFIER} training...")
    num_epochs = 50
    model.train()

    # Convert data to tensors
    X_tensor = torch.FloatTensor(X_data).to(device)
    Y_tensor = torch.FloatTensor(Y_data).to(device)

    # Create simple dataset
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0

        if CLASSIFIER == "ddpm":
            for batch_obs, batch_action in train_loader:
                batch_size = batch_obs.shape[0]

                # Sample random diffusion timestep
                t = torch.randint(0, model.num_timesteps, (batch_size,), device=device)

                # Add noise to actions
                noise = torch.randn_like(batch_action)
                noisy_action = model.q_sample(batch_action.unsqueeze(1), t, noise)

                # Prepare input (concatenate obs and noisy action)
                model_input = torch.cat([batch_obs.unsqueeze(1), noisy_action], dim=1)

                # Forward pass - predict noise
                pred_noise = model.network(model_input, t)
                loss = criterion(pred_noise, noise.unsqueeze(1))

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
        else:
            # Original DiffuseLoco training
            for batch_obs, batch_next in train_loader:
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

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

    print(f"{CLASSIFIER} training completed!")

    # Test trained model
    model.eval()
    with torch.no_grad():
        test_obs = torch.FloatTensor(X_data[:5]).to(device)

        if CLASSIFIER == "ddpm":
            # Use DDPM sampling
            pred_joints = model.sample(test_obs, num_inference_steps=20)
            print("DDPM sample predictions:", pred_joints.cpu().numpy())
        else:
            # Original testing
            test_t = torch.zeros(5, device=device)
            pred_joints = model(test_obs, test_t)
            print("Sample predictions:", pred_joints.cpu().numpy())

    # Save trained model
    model_name = f"{CLASSIFIER}_model.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as '{model_name}'")


# @configclass overrides the base class
@configclass
class AnymalDFlatCameraEnvCfg(AnymalDFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # add tiled camera to the existing scene config
        self.scene.tiled_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(-12.0, 2.0, 3.0),
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
env_cfg.scene.num_envs = 8
env = ManagerBasedRLEnv(cfg=env_cfg)

cumulative_rewards = torch.zeros(env.num_envs, device=env.device)

obs, info = env.reset()

print(info)

# Create imgs directory if it doesn't exist
os.makedirs("imgs", exist_ok=True)

# actions = torch.zeros_like(env.action_manager.action)
# obs, rew, terminated, truncated, info = env.step(actions)

# cumulative reward -> tqdm pbar label dynamically
for i in tqdm.trange(
    0, 1000, desc=f"Cumulative Reward: {cumulative_rewards[0].item()}"
):
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

        actions = actions_pred
        print(f"actions: {actions}")
        print()

    elif CLASSIFIER == "ddpm":
        print(f"model: {model}")
        obs_first_36_features = obs["policy"][:, :36]

        # Use DDPM sampling for action prediction
        with torch.no_grad():
            actions_pred = model.sample(obs_first_36_features, num_inference_steps=10)

        actions = actions_pred
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
