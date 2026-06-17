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

# IsaacLab offline generated
DIFFUSELOCO_50HZ_ISAACLAB_MAY_14_CHECKPOINT := "../diffuseloco-fork/DiffuseLoco/outputs/2026-05-14/16-44-42/checkpoints/latest.ckpt"

DIFFUSELOCO_50HZ_ISAACLAB_MAY_20_CHECKPOINT := "../diffuseloco-fork/DiffuseLoco/outputs/2026-05-20/18-38-08/checkpoints/latest.ckpt"

DIFFUSELOCO_30HZ_GRANDTOUR_GOAL_CONDITION := "../diffuseloco-fork/DiffuseLoco/outputs/2026-05-18/17-28-51/checkpoints/latest.ckpt"

DIFFUSELOCO_50HZ_GRANDTOUR_UPSAMPLING := "../diffuseloco-fork/DiffuseLoco/outputs/2026-05-25/17-45-06/checkpoints/latest.ckpt"

DIFFUSELOCO_50HZ_GRANDTOUR_GOAL_CONDITION := "../diffuseloco-fork/DiffuseLoco/outputs/2026-06-10/17-59-54/checkpoints/latest.ckpt"

DIFFUSELOCO_50HZ_GRANDTOUR_OFFSET_ONE := "../diffuseloco-fork/DiffuseLoco/outputs/2026-06-10/19-09-09/checkpoints/latest.ckpt"

DIFFUSELOCO_50HZ_GRANDTOUR_OFFSET_ONE_GRAVITY_FIX := "../diffuseloco-fork/DiffuseLoco/outputs/2026-06-10/23-37-55/checkpoints/latest.ckpt"

DIFFUSELOCO_50HZ_GRANDTOUR_OFFSET_ONE_GRAVITY_FIX_V2 := "../diffuseloco-fork/DiffuseLoco/outputs/2026-06-11/00-22-59/checkpoints/latest.ckpt"

DIFFUSELOCO_50HZ_ISAACLAB_OFFSET_ONE := "../diffuseloco-fork/DiffuseLoco/outputs/2026-06-11/14-20-27/checkpoints/latest.ckpt"

quickstart:
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_30HZ_FULL_CHECKPOINT}} --velocity-preset forward_only_slow --controller-frequency 30

# user can define velocity preset with --velocity-preset
# user can define controller frequency with --controller-frequency
# [arg("velocity_preset", long)]
# [arg("controller_frequency", long)]
# [arg("diffuseloco_checkpoint", long)]
run_inference_diffuseloco_30hz_full velocity_preset="forward_only_slow" controller_frequency="30" env="flat":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_30HZ_FULL_CHECKPOINT}} --checkpoint-nickname "30hz_full" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}} --env {{env}}

run_inference_diffuseloco_50hz_full velocity_preset="forward_only_slow" controller_frequency="50" env="flat":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_50HZ_FULL_CHECKPOINT}} --checkpoint-nickname "50hz_full" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}} --env {{env}}

run_inference_diffuseloco_20hz_full velocity_preset="forward_only_slow" controller_frequency="20" env="flat":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_20HZ_FULL_CHECKPOINT}} --checkpoint-nickname "20hz_full" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}} --env {{env}}

run_inference_diffuseloco_30hz_flat velocity_preset="forward_only_slow" controller_frequency="30" env="flat":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_30HZ_FLAT_CHECKPOINT}} --checkpoint-nickname "30hz_flat" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}} --env {{env}}

run_inference_diffuseloco_30hz_rough velocity_preset="forward_only_slow" controller_frequency="30" env="rough":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_30HZ_ROUGH_CHECKPOINT}} --checkpoint-nickname "30hz_rough" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}} --env {{env}}

run_inference_diffuseloco_50hz_isaaclab_may_14 velocity_preset="forward_only_slow" controller_frequency="50" env="flat":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_50HZ_ISAACLAB_MAY_14_CHECKPOINT}} --checkpoint-nickname "50hz_isaaclab_may_14" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}} --env {{env}}

run_inference_diffuseloco_50hz_isaaclab_may_20 velocity_preset="forward_only_slow" controller_frequency="50" env="flat":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_50HZ_ISAACLAB_MAY_20_CHECKPOINT}} --checkpoint-nickname "50hz_isaaclab_may_20" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}} --env {{env}}

run_inference_diffuseloco_30hz_grandtour_goal_condition velocity_preset="stationary" controller_frequency="30" env="flat":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_30HZ_GRANDTOUR_GOAL_CONDITION}} --checkpoint-nickname "30hz_grandtour_goal_condition" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}} --env {{env}}

run_inference_diffuseloco_50hz_grandtour_upsampling velocity_preset="forward_only_slow" controller_frequency="50" env="flat":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_50HZ_GRANDTOUR_UPSAMPLING}} --checkpoint-nickname "50hz_grandtour_upsampling" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}} --env {{env}}

run_inference_diffuseloco_50hz_grandtour_goal_condition velocity_preset="forward_only_slow" controller_frequency="50" env="flat":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_50HZ_GRANDTOUR_GOAL_CONDITION}} --checkpoint-nickname "50hz_grandtour_goal_condition" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}} --env {{env}}

run_inference_diffuseloco_50hz_grandtour_offset_one velocity_preset="default" controller_frequency="50" env="flat":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_50HZ_GRANDTOUR_OFFSET_ONE}} --checkpoint-nickname "50hz_grandtour_offset_one" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}} --env {{env}}

run_inference_diffuseloco_50hz_grandtour_offset_one_gravity_fix velocity_preset="default" controller_frequency="50" env="flat":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_50HZ_GRANDTOUR_OFFSET_ONE_GRAVITY_FIX}} --checkpoint-nickname "50hz_grandtour_offset_one_gravity_fix" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}} --env {{env}}

run_inference_diffuseloco_50hz_grandtour_offset_one_gravity_fix_v2 velocity_preset="forward_only_slow" controller_frequency="50" env="flat":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_50HZ_GRANDTOUR_OFFSET_ONE_GRAVITY_FIX_V2}} --checkpoint-nickname "50hz_grandtour_offset_one_gravity_fix_v2" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}} --env {{env}}

run_inference_diffuseloco_50hz_isaaclab_offset_one velocity_preset="default" controller_frequency="50" env="flat":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python demo_serve.py --checkpoint {{DIFFUSELOCO_50HZ_ISAACLAB_OFFSET_ONE}} --checkpoint-nickname "50hz_isaaclab_offset_one" --velocity-preset {{velocity_preset}} --controller-frequency {{controller_frequency}} --env {{env}}

run_isaaclab_anymal_d_quickstart:
    #!/bin/bash
    conda activate isaaclab-2-0-2
    cd IsaacLab
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 4096 --max_iterations 2000 --headless \
        --logger wandb \
        ++actions.joint_pos.scale=1.0 \
        ++actions.joint_pos.use_default_offset=false \
        ++actions.joint_pos.offset=0.0 \
        ++observations.policy.enable_corruption=false
