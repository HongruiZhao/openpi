import numpy as np
from openpi.policies import libero_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

def main():
    # Load the config for pi05_libero
    config = _config.get_config("pi05_libero")
    
    # Specify the checkpoint directory. Since we already downloaded it, maybe_download will return the local path.
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")

    print(f"Loading policy from {checkpoint_dir}...")
    # Create a trained policy.
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    # Run inference on a dummy LIBERO example.
    print("Running inference on dummy example...")
    example = libero_policy.make_libero_example()
    result = policy.infer(example)

    # Delete the policy to free up memory.
    del policy

    actions = result["actions"]
    print("Inference successful!")
    print("Actions shape:", actions.shape)
    print("First few actions:\n", actions[0])

if __name__ == "__main__":
    main()
