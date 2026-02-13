# openpi - Gemini Context

Openpi is an open-source framework by Physical Intelligence for training and deploying Vision-Language-Action (VLA) models for robotics. It supports flow-based (π₀, π₀.₅) and autoregressive (π₀-FAST) models.

## Project Overview

- **Core Technology:** JAX/Flax for primary models, with recent support for PyTorch.
- **Models:**
  - **π₀:** Flow-based VLA.
  - **π₀-FAST:** Autoregressive VLA with FAST action tokenizer.
  - **π₀.₅:** Upgraded π₀ with better generalization and knowledge insulation.
- **Architecture:**
  - `src/openpi/models`: Model definitions (JAX).
  - `src/openpi/models_pytorch`: Model definitions (PyTorch).
  - `src/openpi/policies`: Policy wrappers for different robot platforms (ALOHA, DROID, LIBERO).
  - `src/openpi/training`: Training loops, data loaders, and configuration.
  - `scripts/`: Entry points for training, inference, and serving.

## Building and Running

### Environment Setup
The project uses `uv` for dependency management.

```bash
# Initialize submodules
git submodule update --init --recursive

# Install dependencies (requires GIT_LFS_SKIP_SMUDGE=1 for LeRobot)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Key Commands
- **Compute Norm Stats:** `uv run scripts/compute_norm_stats.py --config-name <config_name>`
- **Training (JAX):** `uv run scripts/train.py <config_name> --exp-name <name>`
- **Training (PyTorch):** `uv run scripts/train_pytorch.py <config_name> --exp-name <name>`
- **Serving Policy:** `uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=<path>`

## Development Conventions

- **Configuration:** Uses `src/openpi/training/config.py` for defining experiments.
- **Data:** Uses LeRobot for dataset handling. Data conversion scripts are in `examples/`.
- **Typing:** Uses `jaxtyping` and `beartype` for runtime type checking in JAX code.
- **Code Style:** Adheres to `ruff` for linting and formatting (config in `pyproject.toml`).
- **Mixed Precision:** JAX training defaults to bfloat16 activations. PyTorch supports full bfloat16 or float32.

## Key Files & Directories

- `src/openpi/training/config.py`: Central registry for training configurations.
- `src/openpi/policies/policy_config.py`: Factory for creating policies from checkpoints.
- `src/openpi/transforms.py`: Data transformation and normalization logic.
- `examples/`: Robot-specific implementation examples and data conversion scripts.
- `packages/openpi-client`: Client-side library for interacting with policy servers.
