"""Model creation and training utilities."""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import torch.optim as optim
import math

# Import TransformerForDiffusion from diffusion_policy
import sys
import pathlib

sys.path.insert(
    0, str(pathlib.Path(__file__).parent.parent.parent / "diffusion_policy")
)
from diffusion_policy.model.diffusion.transformer_for_diffusion import (
    TransformerForDiffusion,
)


class DiffusionTransformerPolicy(nn.Module):
    """Diffusion policy using TransformerForDiffusion architecture.

    Uses encoder-decoder transformer with observation conditioning.
    Based on: https://github.com/real-stanford/diffusion_policy
    """

    def __init__(
        self,
        obs_dim=36,
        action_dim=12,
        horizon=8,
        n_layer=6,
        n_head=8,
        n_emb=256,
        num_timesteps=100,
        n_obs_steps=1,
        p_drop_emb=0.1,
        p_drop_attn=0.1,
        causal_attn=True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_timesteps = num_timesteps
        self.n_obs_steps = n_obs_steps

        # TransformerForDiffusion as the noise prediction network
        # obs_as_cond=True means observations are used as conditioning
        # separate_goal_conditioning=True treats last 3 dims (velocity commands) as goals
        self.model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=obs_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=True,
            obs_as_cond=True,
            n_cond_layers=0,
            separate_goal_conditioning=True,
        )

        # Noise schedule - register as buffers so they move to GPU with model
        beta = torch.linspace(0.0001, 0.02, num_timesteps)
        alpha = 1.0 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), alpha_cumprod[:-1]])
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alpha_cumprod)
        )
        self.register_buffer(
            "posterior_variance",
            beta * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod),
        )

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][
            :, None, None
        ]
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _prepare_cond(self, obs):
        """Restructure obs so velocity commands [21:24] are at the end as goals.

        Input: obs (batch, 36) or (batch, n_obs_steps, 36)
        Output: cond (batch, n_obs_steps, 36) with [obs[:21], obs[24:], obs[21:24]]
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)
        # obs shape: (batch, n_obs_steps, 36)
        # Extract: obs[:21] (dims 0-20), obs[24:] (dims 24-35), obs[21:24] (dims 21-23, the goal)
        obs_part1 = obs[..., :21]  # indices 0-20
        obs_part2 = obs[..., 24:]  # indices 24-35
        goal = obs[..., 21:24]  # indices 21-23 (velocity commands)
        # Concatenate: [observations, goals] -> goals at the end for separate_goal_conditioning
        cond = torch.cat([obs_part1, obs_part2, goal], dim=-1)
        return cond

    def forward(self, obs, action_traj, t):
        """Forward pass for training - predict noise."""
        cond = self._prepare_cond(obs)
        return self.model(sample=action_traj, timestep=t, cond=cond)

    def p_sample(self, action_traj, t, obs):
        """Single reverse diffusion step: p(x_{t-1} | x_t)"""
        batch_size = action_traj.shape[0]
        t_batch = torch.full(
            (batch_size,), t, device=action_traj.device, dtype=torch.long
        )
        cond = self._prepare_cond(obs)
        with torch.no_grad():
            pred_noise = self.model(sample=action_traj, timestep=t_batch, cond=cond)
        alpha_t = self.alpha[t]
        beta_t = self.beta[t]
        alpha_cumprod_t = self.alpha_cumprod[t]
        alpha_cumprod_prev_t = self.alpha_cumprod_prev[t]
        pred_x0 = (
            action_traj - torch.sqrt(1.0 - alpha_cumprod_t) * pred_noise
        ) / torch.sqrt(alpha_cumprod_t)
        pred_mean = torch.sqrt(alpha_cumprod_prev_t) * beta_t * pred_x0
        pred_mean = (
            pred_mean + torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev_t) * action_traj
        )
        pred_mean = pred_mean / (1.0 - alpha_cumprod_t)
        if t == 0:
            return pred_mean
        else:
            posterior_variance_t = self.posterior_variance[t]
            noise = torch.randn_like(action_traj)
            return pred_mean + torch.sqrt(posterior_variance_t) * noise

    def sample(self, obs, num_inference_steps=20):
        """Sample action trajectory from observation using DDPM."""
        batch_size = obs.shape[0]
        device = obs.device
        # obs shape: (batch, obs_dim) - will be processed by _prepare_cond in p_sample
        action_traj = torch.randn(
            batch_size, self.horizon, self.action_dim, device=device
        )
        timesteps = torch.linspace(
            self.num_timesteps - 1,
            0,
            num_inference_steps,
            dtype=torch.long,
            device=device,
        )
        for t in timesteps:
            action_traj = self.p_sample(action_traj, t.item(), obs)
        return action_traj

    def get_optimizer(self, learning_rate=1e-4, weight_decay=1e-3, betas=(0.9, 0.95)):
        """Get optimizer with weight decay configuration."""
        return self.model.configure_optimizers(
            learning_rate=learning_rate, weight_decay=weight_decay, betas=betas
        )


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
        horizon=1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_timesteps = num_timesteps
        self.use_transformer = use_transformer
        self.horizon = horizon

        # Time embedding - using sinusoidal embedding
        self.time_embed_dim = hidden_dim

        if use_transformer:
            # Transformer-based network - handles trajectory dimension
            self.network = TransformerNoiseNet(
                input_dim=obs_dim + action_dim + hidden_dim,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                horizon=horizon,
                num_layers=num_layers,
                num_heads=num_heads,
            )
        else:
            # Conv1d-based network for temporal sequences
            # Input: (batch, horizon, obs_dim + action_dim + hidden_dim)
            layers = []
            input_channels = obs_dim + action_dim + hidden_dim

            # Stack of 1D convolutions to process trajectory
            layers.append(
                nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1)
            )
            layers.append(nn.ReLU())
            for _ in range(num_layers - 1):
                layers.append(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
                )
                layers.append(nn.ReLU())
            layers.append(nn.Conv1d(hidden_dim, action_dim, kernel_size=3, padding=1))
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
        """Add noise to action trajectory at timestep t.

        Args:
            x_start: (batch, horizon, action_dim) - clean action trajectory
            t: (batch,) - diffusion timestep for each sample
            noise: optional noise tensor
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Handle dimensions: t is (batch,), need to reshape for broadcasting
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t])  # (batch,)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t])  # (batch,)

        # Reshape for broadcasting across horizon and action dims
        sqrt_alpha_cumprod = sqrt_alpha_cumprod[:, None, None]  # (batch, 1, 1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod[
            :, None, None
        ]  # (batch, 1, 1)

        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise

    def forward(self, obs, action_traj, t):
        """Forward pass for predicting noise in action trajectory.

        Args:
            obs: (batch, obs_dim) - observation
            action_traj: (batch, horizon, action_dim) - noisy action trajectory
            t: (batch,) - diffusion timestep

        Returns:
            pred_noise: (batch, horizon, action_dim) - predicted noise
        """
        batch_size = obs.shape[0]

        # Broadcast observation to match horizon
        # obs: (batch, obs_dim) -> (batch, horizon, obs_dim)
        obs_expanded = obs.unsqueeze(1).expand(-1, self.horizon, -1)

        # Concatenate obs and noisy action trajectory
        combined = torch.cat(
            [obs_expanded, action_traj], dim=-1
        )  # (batch, horizon, obs_dim + action_dim)

        # Get time embedding and expand to horizon
        t_embed = self.sinusoidal_time_embedding(t)  # (batch, time_embed_dim)
        t_embed = t_embed.unsqueeze(1).expand(
            -1, self.horizon, -1
        )  # (batch, horizon, time_embed_dim)

        # Concatenate with combined input
        combined_input = torch.cat(
            [combined, t_embed], dim=-1
        )  # (batch, horizon, obs_dim + action_dim + time_embed_dim)

        if self.use_transformer:
            # Transformer returns (batch, horizon, action_dim)
            pred_noise = self.network(combined_input)
        else:
            # Conv1d expects (batch, channels, length)
            combined_input = combined_input.transpose(
                1, 2
            )  # (batch, input_dim, horizon)
            pred_noise = self.network(combined_input)  # (batch, action_dim, horizon)
            pred_noise = pred_noise.transpose(1, 2)  # (batch, horizon, action_dim)

        return pred_noise

    def sample(self, obs, num_inference_steps=20):
        """Sample action trajectory from observation using DDPM.

        Args:
            obs: (batch, obs_dim) - observation
            num_inference_steps: number of diffusion steps for inference

        Returns:
            action_trajectory: (batch, horizon, action_dim) - action trajectory
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Start with pure noise for entire trajectory
        action_traj = torch.randn(
            batch_size, self.horizon, self.action_dim, device=device
        )

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

            with torch.no_grad():
                # Predict noise for entire trajectory at once
                pred_noise = self.forward(obs, action_traj, t_batch)

            # Compute mean for reverse process
            alpha_t = self.alpha[t_batch][:, None, None]
            alpha_cumprod_t = self.alpha_cumprod[t_batch][:, None, None]
            beta_t = self.beta[t_batch][:, None, None]

            mean = (1 / torch.sqrt(alpha_t)) * (
                action_traj - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise
            )

            # Add noise for all but final step
            if t > 0:
                noise = torch.randn_like(action_traj)
                action_traj = mean + torch.sqrt(beta_t) * noise
            else:
                action_traj = mean

        return action_traj  # (batch, horizon, action_dim)


def train_model(X_data, Y_data, model, exp_config, exp_name, classifier, device):
    """Train a model based on experiment config.

    Args:
        X_data: Training observations
        Y_data: Training actions
        model: Pre-initialized model to train
        exp_config: Experiment configuration dict
        exp_name: Experiment name
        classifier: Model type ("linear_regression", "ddpm", etc.)
        device: torch device

    Returns:
        tuple: (model, optimizer, criterion, training_stats)
    """
    # Get config values with defaults
    num_epochs = exp_config.get("n_epochs", 10) if exp_config else 10

    if classifier == "linear_regression":
        model.fit(X_data[:], Y_data[:])
        print("RMSE: ", np.sqrt(np.mean((model.predict(X_data[:]) - Y_data[:]) ** 2)))
        return model, None, None, {}

    optimizer = model.get_optimizer(learning_rate=1e-4, weight_decay=1e-3)
    criterion = nn.MSELoss()
    print(f"Starting {classifier} training for {exp_name}...")
    model.train()

    # Convert data to tensors
    X_tensor = torch.FloatTensor(X_data).to(device)
    Y_tensor = torch.FloatTensor(Y_data).to(device)

    # Create simple dataset
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0

        if classifier == "ddpm":
            for batch_obs, batch_action in train_loader:
                batch_size = batch_obs.shape[0]
                batch_obs = batch_obs.to(device)
                batch_action = batch_action.to(device)
                action_traj = batch_action.unsqueeze(1).expand(-1, model.horizon, -1)
                t = torch.randint(0, model.num_timesteps, (batch_size,), device=device)
                noise = torch.randn_like(action_traj)
                noisy_action_traj = model.q_sample(action_traj, t, noise)
                pred_noise = model(batch_obs, noisy_action_traj, t)
                loss = criterion(pred_noise, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        else:
            for batch_obs, batch_next in train_loader:
                batch_obs = batch_obs.to(device)
                batch_next = batch_next.to(device)
                t = torch.randint(
                    0, model.diffusion_steps, (batch_obs.shape[0],), device=device
                )
                pred_next = model(batch_obs, t)
                loss = criterion(pred_next, batch_next)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

    print(f"{classifier} training completed for {exp_name}!")

    # Calculate training stats
    total_training_steps = num_epochs * len(train_loader)
    dataset_size = len(X_data)
    training_stats = {
        "training/total_steps": total_training_steps,
        "training/num_epochs": num_epochs,
        "training/dataset_size": dataset_size,
        "training/batch_size": 64,
    }

    # Test and save
    model.eval()
    with torch.no_grad():
        test_obs = torch.FloatTensor(X_data[:5]).to(device)
        if classifier == "ddpm":
            pred_joints = model.sample(test_obs, num_inference_steps=20)
            print("DDPM sample predictions:", pred_joints.cpu().numpy())
        else:
            test_t = torch.zeros(5, device=device)
            pred_joints = model(test_obs, test_t)
            print("Sample predictions:", pred_joints.cpu().numpy())

    # Save with experiment name
    model_name = f"{classifier}_{exp_name}.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as '{model_name}'")

    return model, optimizer, criterion, training_stats


def create_model(classifier, exp_config, device):
    """Create a model based on classifier type and experiment config.

    Args:
        classifier: Model type ("linear_regression", "ddpm", etc.)
        exp_config: Experiment configuration dict
        device: torch device

    Returns:
        model: Initialized model
    """
    # Get config values with defaults
    horizon = exp_config.get("horizon", 8) if exp_config else 8
    n_layer = exp_config.get("n_layers", 6) if exp_config else 6
    n_head = exp_config.get("n_attn_heads", 8) if exp_config else 8

    if classifier == "linear_regression":
        return LinearRegression()
    elif classifier == "ddpm":
        model = DiffusionTransformerPolicy(
            obs_dim=36,
            action_dim=12,
            horizon=horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=256,
            num_timesteps=100,
            n_obs_steps=1,
            causal_attn=True,
        ).to(device)
        print(
            "Using DiffusionTransformerPolicy with TransformerForDiffusion architecture"
        )
        print(f"  horizon={horizon}, n_layer={n_layer}, n_head={n_head}")
        return model
    else:
        model = DiffuseLocoModel().to(device)
        return model
