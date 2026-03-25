# Diffusion Policy Implementation Analysis

## 1. Noise Scheduler Implementation

The noise scheduler in Diffusion Policy is typically implemented using the `DDPMScheduler` from the `diffusers` library. It manages both the forward diffusion process (adding noise) and the reverse diffusion process (removing noise).

### How $\epsilon^k$ is obtained

In the training phase, $\epsilon^k$ (the noise at step $k$) is sampled from a standard normal distribution. The forward process corrupted version of the clean action trajectory $x_0$ at step $k$ is calculated as:

$$x_k = \sqrt{\bar{\alpha}_k} x_0 + \sqrt{1 - \bar{\alpha}_k} \epsilon$$

where $\bar{\alpha}_k$ is the cumulative product of the noise schedule parameters. The model $\epsilon_\theta$ is then trained to predict this noise $\epsilon$ given the noisy sample $x_k$, the timestep $k$, and the conditioning observations $cond$:

$$\text{Loss} = \| \epsilon - \epsilon_\theta(x_k, k, cond) \|^2$$

During inference, the model predicts the noise residual $\epsilon_\theta$, which is then used by the scheduler to perform a reverse diffusion step from $x_k$ to $x_{k-1}$:

$$x_{k-1} = \frac{1}{\sqrt{\alpha_k}} \left( x_k - \frac{1 - \alpha_k}{\sqrt{1 - \bar{\alpha}_k}} \epsilon_\theta(x_k, k, cond) \right) + \sigma_k z, \quad z \sim \mathcal{N}(0, I)$$

### Relevant Codes (`diffusion_policy/policy/diffusion_unet_image_policy.py`)

```python
# Forward process (Training)
noise = torch.randn(trajectory.shape, device=trajectory.device)
timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device).long()
noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

# Reverse process (Inference)
for t in scheduler.timesteps:
    model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)
    trajectory = scheduler.step(model_output, t, trajectory).prev_sample
```

---

## 2. Attention Implementation

Attention is implemented in the `TransformerForDiffusion` class using an Encoder-Decoder architecture.

### Token Structure

The attention mechanism operates on three types of tokens:
1.  **Action Tokens**: A sequence of $T$ tokens representing the action trajectory.
2.  **Observation (Image) Tokens**: A sequence of $n\_obs\_steps$ tokens, where each token is a feature vector encoded from observations (images + low-dim state).
3.  **Diffusion Iteration Token**: A single token representing the current diffusion step $k$, obtained via sinusoidal embedding.

### Attention Process and Equations

The attention process follows the standard Multi-Head Attention formula:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

In the Transformer architecture:
- **Encoder**: Processes the concatenation of the time token and observation tokens:
  $$\text{Memory} = \text{TransformerEncoder}([token_{time}, token_{obs,1}, \dots, token_{obs,n\_obs\_steps}])$$
- **Decoder**: Uses the action tokens as Query ($Q$) and the Encoder output as Key ($K$) and Value ($V$) for cross-attention. It also performs causal self-attention on the action tokens.

### Attention Mask Implementation

Two masks are used to maintain temporal causality:
1.  **Self-Attention Mask (`mask`)**: A causal mask for action tokens.
    $$M_{i,j} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$
2.  **Memory Mask (`memory_mask`)**: A cross-attention mask between actions and the observation/time tokens.
    $$M_{t, s} = \begin{cases} 0 & \text{if } t \ge s-1 \\ -\infty & \text{if } t < s-1 \end{cases}$$
    where $t$ is the action index and $s$ is the memory index (starting with the time token at $s=0$). This ensures that an action at time $t$ can only attend to observations up to time $t$.

### Relevant Codes (`diffusion_policy/model/diffusion/transformer_for_diffusion.py`)

```python
# Mask Generation
if causal_attn:
    sz = T
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    self.register_buffer("mask", mask)
    
    if time_as_cond and obs_as_cond:
        S = T_cond
        t, s = torch.meshgrid(torch.arange(T), torch.arange(S), indexing='ij')
        mask = t >= (s-1) # Causal dependence: action t attends to obs s
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        self.register_buffer('memory_mask', mask)

# Forward Pass
x = self.decoder(
    tgt=action_embeddings,
    memory=encoder_output,
    tgt_mask=self.mask,
    memory_mask=self.memory_mask
)
```


* Use your MCP tools to search through the repo https://github.com/real-stanford/diffusion_policy.git. Answer me two questions:
    * How is the noise scheduler implemented/ how $\epsilon^k$ is obtained. Show me the codes and represent the method using equations 
    * How is attention implemented? Do we perform attention on image token, action tokens, and diffusion iteration token? How is the attention mask implemented? Show me the codes and represent the attention process using equations. 
* Write down your answer to this file.