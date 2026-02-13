
# Download

To run inference, two main types of assets are required: model checkpoints and auxiliary model files (like tokenizers).

## 1. Model Checkpoints
* **What:** Pre-trained or fine-tuned weights (e.g., `pi05_libero`, `pi0_fast_droid`) and normalization statistics.
* **How:**  Command:
```bash
uv run python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))"
```
* **Where:** Stored in `~/.cache/openpi/openpi-assets/checkpoints/<model_name>`. `OPENPI_DATA_HOME`: Set this to override the default cache directory (`~/.cache/openpi`).
* **Size:** 11.6 GB.

## 2. Auxiliary Assets (PaliGemma Tokenizer)
* **What:** The `paligemma_tokenizer.model` file.
* **Why:** The model's language backbone (PaliGemma) requires this SentencePiece model to convert text prompts into tokens. While the checkpoint contains weights, it does *not* always include the tokenizer model, which is fetched from a separate Google-internal or public bucket.
* **How:** If the automated download fails (due to GCS permissions), it can be manually downloaded:
  ```bash
  mkdir -p ~/.cache/openpi/big_vision
  curl -L -o ~/.cache/openpi/big_vision/paligemma_tokenizer.model https://huggingface.co/openvla/openvla-7b/resolve/main/tokenizer.model
  ```
* **Where:** Stored in `~/.cache/openpi/big_vision/paligemma_tokenizer.model`.
* **Size:** 500 kB.


# Running Inference

## Dummy inference 
* Once the checkpoint and tokenizer are available:
```bash
uv run python examples/my_inference.py
```
* GPU VRAM for `pi05_libero`: 24.7 GB.
* Input to pi05 (LIBERO):
  * `observation/state`: (8,) -> End-effector position (3D: x, y, z), Orientation (4D: qx, qy, qz, qw), and Gripper (1D: position).
  * `observation/image`: (224, 224, 3) -> Base camera RGB.
  * `observation/wrist_image`: (224, 224, 3) -> Wrist camera RGB.
  * `prompt`: Language instruction (e.g., "pick up the bowl").
* Output of pi05 (LIBERO):
  * `actions`: (10, 7) -> Sequence of 10 action steps, each 7D: Relative translation (3D: dx, dy, dz), Relative rotation (3D: roll, pitch, yaw deltas), and Absolute gripper (1D: -1.0 for closed, 1.0 for open).

## LIBERO sim 
* You can run the inference script with default parameters:
  ```bash
  uv run examples/libero_sim/inference.py
  ```

* To run a specific task or change the checkpoint:
  ```bash
  uv run examples/libero_sim/inference.py --task_id 5 --task_suite_name libero_spatial
  ```
  * available tasks: [LIBERO_SPATIAL, LIBERO_OBJECT, LIBERO_GOAL, LIBERO_90, LIBERO_10]