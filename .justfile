# example
# just run_inference_diffuseloco_50hz_full stationary 50

set dotenv-load := true

# read WANDB API KEY from .env
# MAKE sure that there is a .env file in the same directory as the .justfile
WANDB_KEY_VALUE := env("LAURENCE_WANDB_API_KEY", "")
WANDB_ENTITY_VALUE := env("WANDB_ENTITY", "")

# full mix diffuseloco checkpoints
DIFFUSELOCO_30HZ_FULL_CHECKPOINT  := "../diffuseloco-fork/DiffuseLoco/outputs/2026-04-22/13-27-46/checkpoints/latest.ckpt"
DIFFUSELOCO_50HZ_FULL_CHECKPOINT  := "../diffuseloco-fork/DiffuseLoco/outputs/2026-04-25/00-52-23/checkpoints/latest.ckpt"
DIFFUSELOCO_20HZ_FULL_CHECKPOINT  := "../diffuseloco-fork/DiffuseLoco/outputs/2026-04-25/22-49-03/checkpoints/latest.ckpt`"

# diffuseloco 30hz trained on subsamples
DIFFUSELOCO_30HZ_FLAT_CHECKPOINT  := "../diffuseloco-fork/DiffuseLoco/outputs/2026-04-27/13-50-48/checkpoints/latest.ckpt"
DIFFUSELOCO_30HZ_ROUGH_CHECKPOINT := "../diffuseloco-fork/DiffuseLoco/outputs/2026-04-27/01-09-41/checkpoints/latest.ckpt"

quickstart:
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_30HZ_FULL_CHECKPOINT}} --velocity-preset forward_only_slow --controller-frequency 30

# user can define velocity preset with --velocity-preset
# user can define controller frequency with --controller-frequency
# [arg("velocity_preset", long)]
# [arg("controller_frequency", long)]
# [arg("diffuseloco_checkpoint", long)]
run_inference_diffuseloco_30hz_full velocity_preset="forward_only_slow" controller_frequency="30":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_30HZ_FULL_CHECKPOINT}} --checkpoint-nickname "30hz_full" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}}

run_inference_diffuseloco_50hz_full velocity_preset="forward_only_slow" controller_frequency="50":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_50HZ_FULL_CHECKPOINT}} --checkpoint-nickname "50hz_full" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}}

run_inference_diffuseloco_20hz_full velocity_preset="forward_only_slow" controller_frequency="20":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_20HZ_FULL_CHECKPOINT}} --checkpoint-nickname "20hz_full" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}}

run_inference_diffuseloco_30hz_flat velocity_preset="forward_only_slow" controller_frequency="30":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_30HZ_FLAT_CHECKPOINT}} --checkpoint-nickname "30hz_flat" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}}

run_inference_diffuseloco_30hz_rough velocity_preset="forward_only_slow" controller_frequency="30":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_30HZ_ROUGH_CHECKPOINT}} --checkpoint-nickname "30hz_rough" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}}

run_isaaclab_anymal_d_quickstart:
    #!/bin/bash
    conda activate isaaclab-2-0-2
    cd IsaacLab
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 4096 --headless