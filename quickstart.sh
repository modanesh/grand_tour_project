# activate conda env
conda activate isaaclab-2-0-2 # (or the env that you named)

# configure diffuse_policy lib path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/diffusion_policy"

# the config file is located at
# ./diffusion_policy/diffusion_policy/config/anymal_diffusion_policy_isaaclab.yaml
python diffuseloco.py --config-name=anymal_diffusion_policy_isaaclab
