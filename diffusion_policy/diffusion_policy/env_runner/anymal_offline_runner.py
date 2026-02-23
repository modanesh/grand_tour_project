import numpy as np
import torch
import h5py
import json
from collections import deque
from tqdm import tqdm
from diffusion_policy.env_runner.base_runner import BaseLowdimRunner

class AnymalOfflineRunner(BaseLowdimRunner):
    def __init__(self,
                 output_dir,
                 dataset_path,
                 mission_metadata_path,
                 test_missions,
                 n_obs_steps=2,
                 rte_window_size=50,
                 normalize=False):
        super().__init__(output_dir)
        self.dataset_path = dataset_path
        self.mission_metadata_path = mission_metadata_path
        self.test_missions = set(test_missions)
        self.n_obs_steps = n_obs_steps
        self.rte_window_size = rte_window_size

    @torch.no_grad()
    def run(self, policy):
        policy.eval()
        device = next(policy.parameters()).device

        # Load data
        with h5py.File(self.dataset_path, 'r') as f:
            all_obs = f['observations'][:]     # (T, 36)
            all_actions = f['actions'][:]      # (T, 12)
            terminals = f['terminals'][:]      # (T,)

        with open(self.mission_metadata_path) as f:
            metadata = json.load(f)
        ep_to_mission = metadata['episode_to_mission']

        # Reconstruct episode start/end indices
        terminal_idxs = np.where(terminals)[0]
        episode_ends = list(terminal_idxs + 1)
        if episode_ends[-1] < len(all_obs):
            episode_ends.append(len(all_obs))
        episode_starts = [0] + episode_ends[:-1]

        # Collect ATE/RTE per mission
        per_mission_ate = {}
        per_mission_rte = {}

        # Count test missions for progress bar
        test_episode_count = sum(1 for ep_idx, _ in enumerate(zip(episode_starts, episode_ends))
                                  if ep_to_mission.get(str(ep_idx), '') in self.test_missions)

        pbar = tqdm(total=test_episode_count, desc='Evaluating missions', unit='episode')

        for ep_idx, (ep_start, ep_end) in enumerate(zip(episode_starts, episode_ends)):
            mission = ep_to_mission.get(str(ep_idx), '')
            if mission not in self.test_missions:
                continue

            pbar.update(1)
            pbar.set_description(f'Evaluating {mission}')

            gt_obs = all_obs[ep_start:ep_end]       # (L, 36)
            gt_actions = all_actions[ep_start:ep_end]  # (L, 12)
            L = len(gt_obs)

            obs_buffer = deque(maxlen=self.n_obs_steps)
            pred_actions = []

            for t in range(L):
                obs_buffer.append(gt_obs[t])
                # Pad buffer if not full yet
                while len(obs_buffer) < self.n_obs_steps:
                    obs_buffer.appendleft(gt_obs[0])

                obs_arr = np.stack(list(obs_buffer))  # (n_obs_steps, 36)
                obs_tensor = torch.FloatTensor(obs_arr).unsqueeze(0).to(device)
                result = policy.predict_action({'obs': obs_tensor})
                pred_action = result['action'][0, 0].cpu().numpy()  # (12,) first step
                pred_actions.append(pred_action)

            pred_actions = np.array(pred_actions)     # (L, 12)
            errors = np.linalg.norm(pred_actions - gt_actions, axis=-1)  # (L,)

            # ATE
            ate = float(np.mean(errors))

            # RTE: mean of per-window errors
            W = self.rte_window_size
            window_ates = [
                float(np.mean(errors[i:i+W]))
                for i in range(0, L - W + 1, W)
            ]
            rte = float(np.mean(window_ates)) if window_ates else ate

            if mission not in per_mission_ate:
                per_mission_ate[mission] = []
                per_mission_rte[mission] = []
            per_mission_ate[mission].append(ate)
            per_mission_rte[mission].append(rte)

        pbar.close()

        # Aggregate across missions
        all_ates = [v for vals in per_mission_ate.values() for v in vals]
        all_rtes = [v for vals in per_mission_rte.values() for v in vals]
        agg_ate = float(np.mean(all_ates)) if all_ates else 0.0
        agg_rte = float(np.mean(all_rtes)) if all_rtes else 0.0

        result = {
            'test_ATE': agg_ate,
            'test_RTE': agg_rte,
            'test_score': -agg_ate,   # for checkpoint manager (higher = better)
            'per_mission_ATE': {m: float(np.mean(v)) for m, v in per_mission_ate.items()},
            'per_mission_RTE': {m: float(np.mean(v)) for m, v in per_mission_rte.items()},
        }
        return result
