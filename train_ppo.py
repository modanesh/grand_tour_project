"""
PPO training for ANYmal-D using Stable Baselines 3 with Isaac Lab 2.0.2.

Obs dim: 36 (first 36 features truncated from environment)
Action dim: 12 (ANYmal-D joint actions)
"""

from isaaclab.app import AppLauncher
import argparse
import torch
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    type=str,
    default="flat",
    choices=["flat", "warehouse"],
    help="Environment type",
)
parser.add_argument(
    "--actuation-mode",
    type=str,
    default="SEA",
    choices=["SEA", "PD", "Implicit"],
)
parser.add_argument("--num-envs", type=int, default=512, help="Number of parallel environments")
parser.add_argument("--total-steps", type=int, default=10_000_000, help="Total training steps")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--obs-dim", type=int, default=36, help="Policy observation dimension")
args, _ = parser.parse_known_args()

ENV_TYPE = args.env
ACTUATION_MODE = args.actuation_mode
NUM_ENVS = args.num_envs
TOTAL_STEPS = args.total_steps
DEVICE = args.device
HEADLESS = args.headless
OBS_DIM = args.obs_dim

# Launch app
app_launcher = AppLauncher(headless=HEADLESS, enable_cameras=False)
simulation_app = app_launcher.app

import isaaclab
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import (
    AnymalDFlatEnvCfg,
)

if ENV_TYPE == "warehouse":
    from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.warehouse_env_cfg import (
        AnymalDWarehouseEnvCfg,
    )

# ============================================================================
# OBSERVATION WRAPPER
# ============================================================================

class ObservationTruncationWrapper:
    """Wraps Isaac Lab env to truncate observations to OBS_DIM."""
    
    def __init__(self, env, obs_dim):
        self.env = env
        self.obs_dim = obs_dim
        self.device = env.device
    
    @property
    def observation_space(self):
        """Gym-compatible observation space."""
        from gymnasium import spaces
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )
    
    @property
    def action_space(self):
        """Gym-compatible action space."""
        from gymnasium import spaces
        action_shape = self.env.action_manager.action.shape[1:]
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=action_shape,
            dtype=np.float32
        )
    
    @property
    def num_envs(self):
        return self.env.num_envs
    
    def reset(self):
        """Reset and return truncated observations."""
        obs, info = self.env.reset()
        obs_truncated = obs["policy"][:, :self.obs_dim]
        return obs_truncated.cpu().numpy().astype(np.float32), info
    
    def step(self, action):
        """Step and return truncated observations."""
        # Convert numpy to tensor
        action_tensor = torch.from_numpy(action).to(self.device, dtype=torch.float32)
        
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action_tensor)
        
        # Truncate observations
        obs_truncated = obs["policy"][:, :self.obs_dim]
        
        # Convert to numpy
        obs_np = obs_truncated.cpu().numpy().astype(np.float32)
        reward_np = reward.cpu().numpy().astype(np.float32)
        terminated_np = terminated.cpu().numpy().astype(np.bool_)
        truncated_np = truncated.cpu().numpy().astype(np.bool_)
        
        return obs_np, reward_np, terminated_np, truncated_np, info
    
    def close(self):
        self.env.close()


# ============================================================================
# ENVIRONMENT CREATION
# ============================================================================

def create_isaac_env(num_envs: int) -> ManagerBasedRLEnv:
    """Create Isaac Lab environment."""
    
    if ENV_TYPE == "warehouse":
        BaseEnvCfg = AnymalDWarehouseEnvCfg
        print("Using warehouse environment")
    else:
        BaseEnvCfg = AnymalDFlatEnvCfg
        print("Using flat environment")
    
    # Create config
    env_cfg = BaseEnvCfg()
    env_cfg.scene.num_envs = num_envs
    
    # Set control frequency
    CONTROLLER_FREQUENCY = 30  # Hz
    DECIMATION = 10
    PHYSICS_FREQUENCY = CONTROLLER_FREQUENCY * DECIMATION
    
    env_cfg.sim.dt = 1 / PHYSICS_FREQUENCY
    env_cfg.decimation = DECIMATION
    
    print(f"Creating Isaac Lab environment with {num_envs} parallel environments")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Wrap for observation truncation
    env = ObservationTruncationWrapper(env, obs_dim=OBS_DIM)
    
    return env


# ============================================================================
# VECTORIZED ENVIRONMENT WRAPPER
# ============================================================================

def create_vectorized_env(num_envs: int):
    """Create vectorized environment compatible with Stable Baselines 3."""
    
    from stable_baselines3.common.vec_env import VecEnv
    
    class IsaacLabVecEnv(VecEnv):
        """Vectorized environment wrapper for Isaac Lab."""
        
        def __init__(self, env):
            self.env = env
            self.reward_monitor = RewardMonitorCallback()
            super().__init__(
                num_envs=env.num_envs,
                observation_space=env.observation_space,
                action_space=env.action_space,
            )
            self.obs, _ = self.env.reset()
            self.reward_monitor.init_callback(self)
        
        def step_async(self, actions):
            self.actions = actions
        
        def step_wait(self):
            obs, reward, terminated, truncated, info = self.env.step(self.actions)
            # SB3 expects dones = terminated | truncated
            dones = terminated | truncated
            self.obs = obs
            
            # Track rewards
            self.reward_monitor.update(reward, dones)
            
            # Convert info to list of dicts (one per environment)
            # SB3 expects infos to be a list of dictionaries
            infos = [{} for _ in range(self.num_envs)]
            
            return obs, reward, dones, infos
        
        def reset(self):
            self.obs, _ = self.env.reset()
            return self.obs
        
        def close(self):
            self.env.close()
        
        def seed(self, seed=None):
            pass
        
        def get_attr(self, attr_name, indices=None):
            return [getattr(self.env, attr_name)]
        
        def set_attr(self, attr_name, value, indices=None):
            setattr(self.env, attr_name, value)
        
        def env_method(self, method_name, *args, **kwargs):
            return [getattr(self.env, method_name)(*args, **kwargs)]
        
        def env_is_wrapped(self, wrapper_type, indices=None):
            """Check if environment is wrapped with a specific wrapper."""
            return False
    
    isaac_env = create_isaac_env(num_envs)
    return IsaacLabVecEnv(isaac_env)


# ============================================================================
# CUSTOM CALLBACK
# ============================================================================

class RewardMonitorCallback:
    """Monitor and display episode rewards during training."""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = None
        self.current_lengths = None
        self.num_envs = None
    
    def init_callback(self, model):
        """Initialize callback with model info."""
        self.num_envs = model.env.num_envs
        self.current_rewards = np.zeros(self.num_envs)
        self.current_lengths = np.zeros(self.num_envs)
    
    def update(self, rewards, dones):
        """Update reward tracking."""
        self.current_rewards += rewards
        self.current_lengths += 1
        
        for i in range(self.num_envs):
            if dones[i]:
                self.episode_rewards.append(self.current_rewards[i])
                self.episode_lengths.append(self.current_lengths[i])
                self.current_rewards[i] = 0
                self.current_lengths[i] = 0
    
    def get_stats(self):
        """Get current reward statistics."""
        if not self.episode_rewards:
            return None
        
        recent = self.episode_rewards[-100:]
        return {
            "mean": np.mean(recent),
            "std": np.std(recent),
            "min": np.min(recent),
            "max": np.max(recent),
        }


# ============================================================================
# TRAINING WITH REWARD MONITORING
# ============================================================================

def train_ppo():
    """Train PPO using Stable Baselines 3."""
    
    print("\n" + "="*70)
    print("PPO TRAINING WITH STABLE BASELINES 3")
    print("="*70)
    
    # Create vectorized environment
    print(f"\n[1] Creating environment...")
    env = create_vectorized_env(NUM_ENVS)
    
    print(f"\n[2] Environment ready!")
    print(f"    Observation space: {env.observation_space}")
    print(f"    Action space: {env.action_space}")
    print(f"    Num envs: {NUM_ENVS}")
    print(f"    Device: {DEVICE}")
    
    # Import Stable Baselines 3
    print(f"\n[3] Importing Stable Baselines 3...")
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    
    # Custom callback to display rewards
    class RewardLogCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.last_print_step = 0
        
        def _on_step(self) -> bool:
            # Print reward stats every 5000 steps
            if self.num_timesteps - self.last_print_step >= 5000:
                stats = self.model.env.reward_monitor.get_stats()
                if stats:
                    print(f"[Step {self.num_timesteps:,}] Reward: "
                          f"mean={stats['mean']:.2f}, "
                          f"std={stats['std']:.2f}, "
                          f"min={stats['min']:.2f}, "
                          f"max={stats['max']:.2f}")
                self.last_print_step = self.num_timesteps
            return True
    
    # Create PPO agent
    print(f"\n[4] Creating PPO agent...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        n_steps=64,  # Rollout length per environment
        batch_size=4096,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,  # Entropy coefficient for exploration
        vf_coef=0.5,  # Value function loss coefficient
        max_grad_norm=0.5,
        target_kl=None,
        tensorboard_log="./logs/",
        device=DEVICE,
        verbose=1,
    )
    
    # Checkpointing
    print(f"\n[5] Setting up checkpointing...")
    from stable_baselines3.common.callbacks import CheckpointCallback
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="anymal_ppo_sb3",
        verbose=0,
    )
    
    reward_callback = RewardLogCallback()
    
    # Train
    print(f"\n[6] Starting training for {TOTAL_STEPS:,} steps...")
    print("="*70 + "\n")
    
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=[checkpoint_callback, reward_callback],
        progress_bar=True,
    )
    
    # Save final model
    print(f"\n[7] Training complete! Saving model...")
    model.save("anymal_ppo_sb3_final")
    
    final_stats = env.reward_monitor.get_stats()
    if final_stats:
        print(f"\n✓ Final episode rewards:")
        print(f"  Mean: {final_stats['mean']:.2f}")
        print(f"  Std: {final_stats['std']:.2f}")
        print(f"  Min: {final_stats['min']:.2f}")
        print(f"  Max: {final_stats['max']:.2f}")
    
    print(f"\n✓ Model saved to: anymal_ppo_sb3_final.zip")
    print(f"  Checkpoints saved to: ./checkpoints/")
    print(f"  Logs saved to: ./logs/")
    print(f"  Total episodes: {len(env.reward_monitor.episode_rewards)}")
    
    env.close()


if __name__ == "__main__":
    try:
        train_ppo()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()