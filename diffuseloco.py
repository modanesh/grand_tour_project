"""
Usage:
Training:
python diffuseloco.py --config-name=anymal_diffusion_policy

For the IsaacLab quickstart:

```bash

# activate conda env
conda activate isaaclab-2-0-2 (or the env that you named)

# configure diffuse_policy lib path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/diffusion_policy"

# the config file is located at
# ./diffusion_policy/diffusion_policy/config/anymal_diffusion_policy_isaaclab.yaml 
python diffuseloco.py --config-name=anymal_diffusion_policy_isaaclab
```

"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

try:
    from isaacgym.torch_utils import *
except:
    print("Isaac Gym Not Installed")
    
import torch
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True

import hydra
from omegaconf import OmegaConf
import pathlib

from diffusion_policy.workspace.base_workspace import BaseWorkspace


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "diffusion_policy", "config")),
    config_name="anymal_diffusion_policy.yaml"
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

