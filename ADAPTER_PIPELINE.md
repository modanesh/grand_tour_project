# Unified Output Decoder Training Pipeline

## Quick Start

```bash
python diffuseloco_with_adapter.py --config-name=anymal_diffusion_policy num_cycles=5
```

This runs a complete 3-phase cycle, 5 times:
1. **Phase 1**: Train policy on Grand Tour data
2. **Phase 2**: Fine-tune output decoder on Isaac Gym expert data
3. **Phase 3**: Evaluate in Isaac Gym
4. Loop back to Phase 1

---

## Architecture Overview

```
Isaac Gym Observation (36D)
           ↓
    [Frozen Policy]
    (trained on GT data)
           ↓
   Policy Output (12D)
           ↓
   [Output Decoder] ← TRAINED HERE (Phase 2)
   (learns mapping to expert actions)
           ↓
Isaac Gym Action (12D)
           ↓
    [Isaac Gym Env]
```

**Key Points:**
- Policy is frozen (no weight updates)
- Only the lightweight output decoder is trained
- Decoder learns: policy_output → expert_actions
- No observation space adaptation needed

---

## Prerequisites

### 1. Data Files
- **Grand Tour training data**: Must be available (same as `diffuseloco.py` setup)
- **Isaac Gym expert data**: `expert_dataset.hdf5`
  - Should contain: `observations` (N, 36) and `actions` (N, 12)
  - This is raw Isaac Gym data (no scaling/normalization)

### 2. Environment
```bash
# Make sure dependencies are installed
pip install torch hydra-core omegaconf tqdm h5py

# Isaac Gym (for evaluation)
# Follow Isaac Gym installation instructions
```

### 3. Configuration
Your Hydra config (`anymal_diffusion_policy.yaml`) should define:
- Grand Tour data loader
- Policy architecture
- Training hyperparameters
- `num_cycles` (number of training cycles)

---

## Component Architecture

### Three Main Components

```
┌─────────────────────────────────────────────────────────┐
│  Grand Tour Policy (diffuseloco.py output)              │
│  ✓ Trained on Grand Tour mission data                   │
│  ✓ Frozen during Phase 2                                │
│  ✓ Input: Isaac Gym observations (36D)                  │
│  ✓ Output: Policy outputs (12D)                         │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Output Decoder (adapter.py)                            │
│  ✓ Lightweight MLP (12D → 256D → 256D → 12D)            │
│  ✓ ONLY component trained in Phase 2                    │
│  ✓ Learns: policy_output → expert_actions              │
│  ✓ Input: Policy outputs (12D)                          │
│  ✓ Output: Isaac Gym actions (12D)                      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Isaac Gym Environment (eval_isaac_v2.py)               │
│  ✓ Task: anymal_d_flat                                  │
│  ✓ normalize=False (critical!)                          │
│  ✓ Input: Actions (12D)                                 │
│  ✓ Output: Reward, observations, done signal            │
└─────────────────────────────────────────────────────────┘
```

### Phase 2 Training Loop

```
Expert Dataset (HDF5)
├── observations: (N, 36)  ← Raw Isaac Gym obs
└── actions: (N, 12)       ← Expert actions (target)

For each batch:
  obs[i] → Policy.act_inference() → policy_output[i]
                                         ↓
                    Decoder(policy_output[i]) → pred_actions[i]
                                         ↓
                    MSE_Loss(pred_actions[i], actions[i])
                                         ↓
                    Backward pass (update decoder only)
```

---

## Detailed Setup

### Step 1: Prepare Expert Dataset

Convert your Isaac Gym trajectories to HDF5:

```python
import h5py
import numpy as np

# Collect Isaac Gym trajectories
observations = []  # List of obs arrays from Isaac Gym
actions = []       # List of action arrays

# Save to HDF5
with h5py.File('expert_dataset.hdf5', 'w') as f:
    f.create_dataset('observations', data=np.array(observations))
    f.create_dataset('actions', data=np.array(actions))

print(f"Saved {len(observations)} trajectories")
```

### Step 2: Verify Directory Structure

```
grand_tour_project/
├── diffuseloco_with_adapter.py          ← Main script (RUN THIS)
├── adapter.py                           ← OutputDecoder module
├── eval_isaac_v2.py                     ← Evaluation script
├── isaac_compatibility.py               ← Isaac Gym utilities
├── grandtour_compatibility.py           ← GT utilities
├── diffusion_policy/                    ← Diffusion policy library
├── expert_dataset.hdf5                  ← Your expert data
└── diffusion_policy/config/
    └── anymal_diffusion_policy.yaml     ← Hydra config
```

### Step 3: Verify Hydra Config

Make sure your config includes:
```yaml
num_cycles: 5  # Number of training cycles

# Grand Tour training config
gt_workspace:
  _target_: "diffusion_policy.workspace.base_workspace.BaseWorkspace"
  # ... your GT training config

# Output Decoder config (or will use defaults)
decoder:
  expert_data_path: "expert_dataset.hdf5"
  policy_output_dim: 12      # Policy outputs 12D actions
  action_dim: 12             # Isaac Gym expects 12D actions
  batch_size: 32
  num_epochs: 10
  learning_rate: 1e-3
  hidden_dim: 256

# Evaluation config
eval:
  task_name: "anymal_d_flat"
  seed: 0
  normalize: False           # CRITICAL: Keep False for decoder training
  include_prev_actions: False
```

---

## Running the Pipeline

### Basic Command

```bash
python diffuseloco_with_adapter.py --config-name=anymal_diffusion_policy
```

### With Custom Parameters

```bash
python diffuseloco_with_adapter.py \
  --config-name=anymal_diffusion_policy \
  num_cycles=3 \
  decoder.batch_size=64 \
  decoder.num_epochs=5 \
  eval.seed=42
```

### Overriding Config Values

```bash
# Change number of cycles
python diffuseloco_with_adapter.py --config-name=anymal_diffusion_policy num_cycles=10

# Change decoder hyperparameters
python diffuseloco_with_adapter.py --config-name=anymal_diffusion_policy \
  decoder.learning_rate=2e-3 \
  decoder.num_epochs=20 \
  decoder.batch_size=64

# Change expert data path
python diffuseloco_with_adapter.py --config-name=anymal_diffusion_policy \
  decoder.expert_data_path=/path/to/custom_expert.hdf5
```

---

## What Happens During Execution

### Cycle 1 Example Output

```
============================================================
UNIFIED ADAPTER TRAINING PIPELINE
============================================================

############################################################
## CYCLE 1/5
############################################################

============================================================
[Cycle 0] PHASE 1: Training on Grand Tour Data
============================================================

Starting GT policy training (Cycle 0)...
[Training progress...]
✓ GT policy training complete

============================================================
[Cycle 0] PHASE 2: Fine-tuning Output Decoder on Expert Isaac Gym Data
============================================================

Loading expert data from: expert_dataset.hdf5
Expert observations shape: (10000, 36)
Expert actions shape: (10000, 12)
Training decoder for 10 epochs with batch size 32
Epoch 1/10 | Loss: 0.234567 | LR: 1.00e-03
Epoch 2/10 | Loss: 0.156234 | LR: 1.00e-03
...
✓ Output decoder fine-tuning complete

============================================================
[Cycle 0] PHASE 3: Evaluating in Isaac Gym
============================================================

────────────────────────────────────
Cycle 0 Evaluation Results:
────────────────────────────────────
  Average reward: 0.8234
  Episodes completed: 10
  Average episode length: 145.50
────────────────────────────────────

Checkpoint saved: outputs/anymal_diffusion_policy/.../checkpoints/cycle_0.ckpt
```

### Checkpoints Saved

After each cycle:
- `cycle_0.ckpt` - Full state from cycle 1
- `cycle_1.ckpt` - Full state from cycle 2
- `latest.ckpt` - Most recent checkpoint

Each checkpoint contains:
- Policy weights (frozen, for reference)
- Output decoder weights (the trained component)
- Evaluation history
- Config

---

## Monitoring Progress

### View Real-time Logs

```bash
# In separate terminal
tail -f outputs/*/anymal_diffusion_policy/*.log
```

### Check Evaluation History

```python
import torch

ckpt = torch.load('outputs/.../checkpoints/latest.ckpt')
eval_history = ckpt['eval_history']

for record in eval_history:
    print(f"Cycle {record['cycle']}: reward={record['eval_score']:.4f}")
```

---

## Troubleshooting

### Decoder loss not decreasing

**Problem**: Decoder loss stays high or increases
- Check expert data quality (observations and actions should be raw Isaac Gym format)
- Verify policy is working (test policy.act_inference on sample obs)
- Increase learning rate: `decoder.learning_rate=2e-3`
- Train longer: `decoder.num_epochs=20`
- Check dimensions: policy outputs 12D, decoder outputs 12D

### Evaluation score poor

**Problem**: Robot doesn't walk well in Isaac Gym
- Expert data may not cover enough behaviors
- Decoder needs more training epochs: `decoder.num_epochs=20`
- Check `normalize=False` in config (critical!)
- Verify expert data is from `anymal_d_flat` task
- Check policy is outputting reasonable values

### Out of memory

**Problem**: CUDA out of memory error
- Reduce batch size: `decoder.batch_size=16`
- Decoder is lightweight, so this is rarely the issue
- Use CPU: Add `device=cpu` to config (slower but works)

### Policy checkpoint not found

**Problem**: "GT workspace doesn't have 'policy' attribute"
- Verify your GT training config creates a `policy` object
- Check Hydra config defines the policy correctly

---

## Advanced Usage

### Resume From Checkpoint

```python
# In Python
import torch
from diffuseloco_with_adapter import AdapterTrainingWorkspace
from omegaconf import OmegaConf

ckpt = torch.load('outputs/.../checkpoints/cycle_3.ckpt')
cfg = OmegaConf.create(ckpt['config'])

workspace = AdapterTrainingWorkspace(cfg)
workspace.cycle = ckpt['cycle']  # Resume from cycle 3
# Continue from here...
```

---

## Expected Behavior

### Phase 1 (Grand Tour Training)
- Policy learns locomotion on Grand Tour simulator
- Training follows your standard `diffuseloco.py` procedure
- Produces a pre-trained policy checkpoint

### Phase 2 (Output Decoder Fine-tuning)
- Decoder loss should decrease steadily (e.g., 0.5 → 0.1)
- Only decoder weights update, policy is frozen
- Takes ~2-5 minutes per epoch (depends on dataset size, very lightweight)

### Phase 3 (Evaluation)
- Robot walks in Isaac Gym with adapted observations
- `anymal_d_flat` task - flat terrain locomotion
- Expected reward: 0.5-1.0 (task-dependent)
- 10 episodes per eval = ~2-3 minutes

### Loop Back
- Cycle repeats with updated policy + adapter
- Performance should improve over cycles (usually)

---

## File Organization

```
outputs/
├── anymal_diffusion_policy/
│   ├── 2026-02-24/
│   │   └── 10-30-45/
│   │       ├── logs/
│   │       │   └── train.log
│   │       └── checkpoints/
│   │           ├── cycle_0.ckpt
│   │           ├── cycle_1.ckpt
│   │           ├── cycle_2.ckpt
│   │           └── latest.ckpt
│   └── config.yaml
```

---

## Common Commands

```bash
# Full pipeline, 5 cycles
python diffuseloco_with_adapter.py --config-name=anymal_diffusion_policy num_cycles=5

# Quick test (1 cycle only)
python diffuseloco_with_adapter.py --config-name=anymal_diffusion_policy num_cycles=1

# Long training (20 cycles, different seed)
python diffuseloco_with_adapter.py --config-name=anymal_diffusion_policy num_cycles=20 eval.seed=42

# Custom expert data
python diffuseloco_with_adapter.py --config-name=anymal_diffusion_policy \
  decoder.expert_data_path=path/to/custom.hdf5

# Aggressive decoder training
python diffuseloco_with_adapter.py --config-name=anymal_diffusion_policy \
  decoder.learning_rate=5e-3 \
  decoder.num_epochs=20 \
  decoder.batch_size=64
```

---

## FAQ

**Q: Why is normalization turned off?**
A: The decoder trains on raw Isaac Gym observations and expert actions. Normalization would change the observation scale before the policy processes them, causing a mismatch with training.

**Q: Can I use a different Isaac Gym task?**
A: Yes, change `eval.task_name: "anymal_d_flat"` to any other task name supported by Isaac Gym.

**Q: What if training stops mid-cycle?**
A: The last completed checkpoint is saved. You can resume by loading that checkpoint and continuing.

**Q: How long does one cycle take?**
A: Depends on your hardware and config:
- Phase 1 (GT training): ~30-60 minutes
- Phase 2 (Decoder): ~5-10 minutes (very fast, lightweight MLP)
- Phase 3 (Eval): ~5 minutes
- **Total per cycle: ~45-75 minutes**

**Q: Can I parallelize cycles?**
A: Not easily - they're sequential by design. Each cycle depends on the previous policy.

---

## Support

If you encounter issues:
1. Check logs in `outputs/.../logs/`
2. Verify expert data format with `h5py`
3. Test individual phases separately
4. Check compatibility files (`isaac_compatibility.py`, `grandtour_compatibility.py`)

