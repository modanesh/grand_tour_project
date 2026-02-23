from align_data import align_data, load_aligned_data
from build_dataset import build_offline_dataset, save_offline_dataset_hdf5, episode_returns, DatasetConfig
import numpy as np
import json
from collections import OrderedDict

print("\nBuilding Offline Dataset...")
topics = [
            "anymal_state_odometry",
            "anymal_state_state_estimator",
            "anymal_imu",
            "anymal_state_actuator",
            "anymal_command_twist",
            #"hdr_front",
            #"hdr_left",
            #"hdr_right"
        ]

# if aligned data has already been instantiated, then can just load the aligned data.
aligned, mission_timesteps = align_data(topics=topics, save_aligned=False)
#aligned = load_aligned_data("aligned_data.zarr")

output_file = "aligned_dataset_shapes.txt"
with open(output_file, "w") as f:
    for sensor in topics:
        f.write(f"{sensor}:\n")
        keys = sorted(aligned["sensors"][sensor].keys())
        for key in keys:
            shape = aligned["sensors"][sensor][key].shape
            dtype = type(aligned["sensors"][sensor][key])
            f.write(f"-->    {key} {shape} {dtype}\n")
        f.write("------------------------------\n\n")

print(f"Sensor info written to {output_file}")
ds_cfg = DatasetConfig()
# set config here
ds_cfg.scale_lin_vel = True
ds_cfg.scale_ang_vel = False
ds_cfg.scale_commands = True
ds_cfg.scale_joint_pos = True
ds_cfg.scale_joint_vel = True
ds_cfg.scale_actions = True


dataset, episode_sums_total = build_offline_dataset(aligned,ds_cfg)
save_offline_dataset_hdf5(dataset)

# --- NEW: Generate mission metadata sidecar ---
# Compute cumulative timestep offsets per mission (raw, before obs[:-1] shift)
# Note: obs[:-1] only removes the very last timestep, so episode boundaries are unaffected
cumulative = 0
mission_offsets = OrderedDict()
for mission_name, n_steps in mission_timesteps.items():
    mission_offsets[mission_name] = cumulative
    cumulative += n_steps

# Map episode index -> mission (episode k starts at timestep k*1000 in the flat dataset)
total_timesteps = cumulative - 1   # after obs[:-1] shift
n_episodes = (total_timesteps + 999) // 1000
episode_to_mission = {}
mission_to_episodes = {m: [] for m in mission_timesteps}

for ep_idx in range(n_episodes):
    ep_start_ts = ep_idx * 1000
    for mission_name, offset in mission_offsets.items():
        mission_end = offset + mission_timesteps[mission_name]
        if offset <= ep_start_ts < mission_end:
            episode_to_mission[ep_idx] = mission_name
            mission_to_episodes[mission_name].append(ep_idx)
            break

metadata = {
    "missions": list(mission_timesteps.keys()),
    "mission_timesteps": dict(mission_timesteps),
    "episode_to_mission": {str(k): v for k, v in episode_to_mission.items()},
    "mission_to_episodes": {k: v for k, v in mission_to_episodes.items() if v},
    "total_episodes": n_episodes,
    "episode_length": 1000,
}
with open("mission_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved mission_metadata.json ({len(mission_timesteps)} missions, {n_episodes} episodes)")

# Write dataset info to file and print
with open(output_file, "a") as f:
    f.write("\nBuilding Offline Dataset...\n")
    f.write("\nOffline Dataset:\n")
    for k in list(dataset.keys()):
        info = f"{k}: {type(dataset[k])} {dataset[k].shape}\n"
        print(info.strip())
        f.write(info)
    
    ep_ret = episode_returns(dataset["rewards"], dataset["terminals"])
    total_episodes = f"Total episodes: {len(ep_ret)}\n"
    median_return = f"Median Episode Return: {np.median(ep_ret)}\n"
    
    print(total_episodes.strip())
    print(median_return.strip())
    
    f.write(total_episodes)
    f.write(median_return)
    
    # Compute and print per-term average episode rewards
    T_total = len(dataset["rewards"])
    num_episodes = len(ep_ret)
    avg_episode_length = T_total / num_episodes if num_episodes > 0 else 0
    
    print("\n" + "="*60)
    print("Per-Term Average Episode Rewards:")
    print("="*60)
    f.write("\n" + "="*60 + "\n")
    f.write("Per-Term Average Episode Rewards:\n")
    f.write("="*60 + "\n")
    
    for term_name, term_total in sorted(episode_sums_total.items()):
        # Adjust for the fact that we computed on full length but dataset is shifted
        term_total_adjusted = term_total  # Already computed on full length
        avg_per_episode = term_total_adjusted / num_episodes if num_episodes > 0 else 0
        avg_per_step = term_total_adjusted / T_total if T_total > 0 else 0
        
        info = f"{term_name:25s}: {avg_per_episode:10.4f} per episode, {avg_per_step:10.6f} per step\n"
        print(info.strip())
        f.write(info)
    
    print("="*60)
    f.write("="*60 + "\n")