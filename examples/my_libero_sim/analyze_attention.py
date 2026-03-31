import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange
def visualize(args):
    npz_path = args.npz_path
    max_step = args.max_step
    
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    
    # prefix_attention: (N_calls, Layers, B, K, G, T_prefix, S_prefix)
    # step_attention: (N_calls, Layers, B, K, G, T_suffix, S_step)
    prefix_attention = data["prefix_attention"].squeeze()
    step_attention = data["step_attention"].squeeze()
    epochs = data["epochs"]
    steps = data["steps"]

    N_calls = step_attention.shape[0]
    num_layers = step_attention.shape[1]
    num_heads = step_attention.shape[2]
    t_suffix = step_attention.shape[3]
    s_step = step_attention.shape[4]
    t_prefix = prefix_attention.shape[-1]

    print(f"N_calls: {N_calls}, num_layers: {num_layers}, num_heads: {num_heads}")
    print(f"T_suffix: {t_suffix}, S_step: {s_step}, T_prefix: {t_prefix}")

    # num_visual_tokens: Pi0 models typically use 3 image slots (base, left wrist, right wrist).
    # Each image has 256 tokens in PaliGemma/SigLIP (224x224 with patch size 14 -> 16x16=256).
    # So 3 * 256 = 768 visual tokens.
    if t_prefix >= 768:
        num_visual_tokens = 768
    elif t_prefix >= 512:
        num_visual_tokens = 512
    elif t_prefix >= 256:
        num_visual_tokens = 256
    else:
        num_visual_tokens = 0
        
    print(f"Detected {num_visual_tokens} visual tokens out of {t_prefix} prefix tokens.")

    avg_step_attention = np.mean(step_attention, axis=(1,2)) # (num_steps, T_suffix, S_step)

    unique_epochs = np.unique(epochs)
    for epoch in tqdm(unique_epochs):
        sim_steps = np.where(epochs == epoch)[0] 
        fig, ax = plt.subplots()
        fig.suptitle(f"Epoch {epoch}")
        
        attn = avg_step_attention[sim_steps] 
        attn = rearrange(attn, 'sim_steps  T_suffix  S_step -> (sim_steps T_suffix) S_step')
        
        visual_attn = attn[:,0:num_visual_tokens].sum(axis=-1)
        text_attn = attn[:,num_visual_tokens:t_prefix].sum(axis=-1)
        act_attn = attn[:,t_prefix:].sum(axis=-1)

        if max_step > 0:
            visual_attn = visual_attn[:max_step]
            text_attn = text_attn[:max_step]
            act_attn = act_attn[:max_step]  
        ax.plot(visual_attn, label="Visual")
        ax.plot(text_attn, label="Text")
        ax.plot(act_attn, label="Action/proprioception")
    
        ax.set_xlabel("Step")
        ax.set_ylabel("Attention Weight")
        ax.legend()
            
        plt.tight_layout()
        out_path = npz_path.replace(".npz", f"_epoch_{epoch}.png")
        plt.savefig(out_path, dpi=300)
        print(f"Saved plot to {out_path}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and visualize attention weights from policy inference.")
    parser.add_argument("--npz_path", type=str, default='examples/my_libero_sim/videos/pi05_libero_libero_goal_0_bowl_cabinet_analysis.npz',
                         help="Path to the .npz file containing attention weights.")
    parser.add_argument("--max_step", type=int, default=0)
    args = parser.parse_args()
    
    if not os.path.exists(args.npz_path):
        print(f"Error: File {args.npz_path} not found.")
    else:
        visualize(args)
