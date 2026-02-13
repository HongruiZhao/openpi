# OpenPI Configurations (π₀.₅ and others)

## π₀.₅ (pi05) Models
For the π₀.₅ (pi05) model family, the `config_name` represents a combination of the model architecture and the target robot/task platform.

- **pi05_libero**: Configures π₀.₅ for LIBERO simulation (observation transforms, state normalization, etc.).
- **pi05_aloha**: For Aloha (Trossen) bimanual robots.
- **pi05_droid**: For Franka Panda robots (DROID dataset).

## Available Configs Summary
@src/openpi/training/config.py
| Category | Config Names |
| :--- | :--- |
| **Aloha (Real/Sim)** | `pi0_aloha`, `pi05_aloha`, `pi0_aloha_towel`, `pi0_aloha_sim`, `pi05_aloha_pen_uncap` |
| **DROID (Franka)** | `pi0_droid`, `pi05_droid`, `pi0_fast_droid`, `pi05_droid_finetune` |
| **LIBERO (Sim)** | `pi0_libero`, `pi05_libero`, `pi0_fast_libero`, `pi0_fast_libero_low_mem_finetune` |
| **Debugging** | `debug`, `debug_pi05`, `debug_restore` |

## Role of TrainConfig
A `TrainConfig` (and its associated `DataConfig`) coordinates both training and inference:

1. **Model Architecture:** Architecture type (π₀, π₀.₅, π₀-FAST), `action_dim`, `action_horizon` (chunk size), and language backbone.
2. **Data Transforms:**
   - **Normalization:** Loads precomputed `norm_stats` (z-score or quantile) for specific sensors/actuators.
   - **Robot Logic:** Platform-specific logic (e.g., absolute to delta action conversion).
   - **Image Processing:** Camera standardization (e.g., 224x224 resizing).
3. **Tokenization:** Language prompt conversion.
4. **Training Details:** Hyperparameters like learning rate, optimizer, batch size, and weight freezing (LoRA).
5. **Weight Loading:** Default checkpoint path for initialization.
