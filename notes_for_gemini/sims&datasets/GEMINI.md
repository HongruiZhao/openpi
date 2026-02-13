# Meta-World

Meta-World is an open-source benchmark designed for multi-task and meta-reinforcement learning in robotic manipulation. It provides a suite of continuous control environments based on the Gymnasium API.

## Key Features
- **Robotic Manipulation:** Focuses on diverse manipulation tasks using the **Sawyer robot arm**.
- **Gymnasium Integration:** Fully compatible with the Gymnasium API for environment interaction and vectorization.
- **Benchmark Suites:**
    - **Multi-Task (MT1, MT10, MT50):** Designed for learning multiple tasks simultaneously. MT10 and MT50 include task IDs in observations.
    - **Meta-Learning (ML1, ML10, ML45):** Designed for testing few-shot adaptation to new goals or entirely new tasks.
- **Execution Modes:** Supports both synchronous (`sync`) and asynchronous (`async`) vector environments for scalability.
- **Customization:** Users can create custom benchmarks by selecting specific task sets.

## Supported Robots
- **Sawyer:** All current Meta-World environments use a simulated Sawyer robot arm for manipulation tasks.

## Available Tasks (50 total)
Meta-World includes 50 distinct robotic manipulation tasks. Below is a categorized list:

### Common Tasks (MT10)
- `reach-v3`, `push-v3`, `pick-place-v3`, `door-open-v3`, `drawer-open-v3`, `drawer-close-v3`, `button-press-topdown-v3`, `peg-insert-side-v3`, `window-open-v3`, `window-close-v3`.

### Meta-Learning Splits (ML10)
- **Train:** MT10 tasks (with minor variations, e.g., `sweep-v3` and `basketball-v3` replace others).
- **Test:** `drawer-open-v3`, `door-close-v3`, `shelf-place-v3`, `sweep-into-v3`, `lever-pull-v3`.

### Full Task List (V3)
1. **Basic:** `reach-v3`, `push-v3`, `pick-place-v3`.
2. **Objects:** `basketball-v3`, `soccer-v3`, `bin-picking-v3`, `box-close-v3`, `hand-insert-v3`.
3. **Controls:** `button-press-v3`, `button-press-topdown-v3`, `dial-turn-v3`, `lever-pull-v3`, `faucet-open-v3`, `faucet-close-v3`.
4. **Furniture/Home:** `door-open-v3`, `door-close-v3`, `drawer-open-v3`, `drawer-close-v3`, `window-open-v3`, `window-close-v3`, `shelf-place-v3`.
5. **Tools:** `hammer-v3`, `disassemble-v3`, `assembly-v3`, `peg-insert-side-v3`, `peg-unplug-side-v3`, `stick-push-v3`, `stick-pull-v3`.
6. **And more:** Various wall-obstructed and directional variants (e.g., `reach-wall-v3`, `handle-press-side-v3`).

## Installation & Documentation
- **Install:** `pip install metaworld`
- **Docs:** [metaworld.farama.org](https://metaworld.farama.org)
- **Repo:** [Farama-Foundation/Metaworld](https://github.com/Farama-Foundation/Metaworld)

# LIBERO

LIBERO (Benchmarking Knowledge Transfer for Lifelong Robot Learning) is a simulation framework designed to study how robots can transfer and accumulate knowledge across a wide variety of tasks. It focuses on both declarative knowledge (objects/spatial relations) and procedural knowledge (behaviors).

## Key Features
- **Procedural Generation:** Includes a pipeline capable of generating a theoretically infinite number of manipulation tasks.
- **Task Suites (130 Tasks):**
    - **LIBERO-Spatial:** Focuses on transferring knowledge about spatial relationships.
    - **LIBERO-Object:** Focuses on object properties and categories.
    - **LIBERO-Goal:** Focuses on diverse language-defined goals.
    - **LIBERO-100:** A large-scale suite of 100 tasks requiring transfer of entangled knowledge.
- **Lifelong Learning Focus:** Specifically designed to evaluate how policies adapt to new tasks sequentially (LIBERO-90 for pretraining, LIBERO-10 for downstream evaluation).
- **Control:** Primarily supports imitation learning and lifelong learning algorithms due to the difficulty of sparse-reward reinforcement learning in these environments.

## Supported Robots
- **Franka Panda:** Uses a simulated Franka Emika Panda robot arm for all manipulation tasks.

## Installation & Documentation
- **Install:**
  To run LIBERO simulation in the current environment, you may need to install the following dependencies:
  ```bash
  uv pip install robosuite==1.4.1 bddl easydict gymnasium shimmy hydra-core thop
  ```
  *Note: `robosuite==1.4.1` is specifically required as newer versions may have breaking structural changes for LIBERO.*
- **Docs:** [lifelong-robot-learning.github.io/LIBERO](https://lifelong-robot-learning.github.io/LIBERO/)
- **Repo:** [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

# Dataset Comparison: DROID, ALOHA, and LIBERO

Comparison of robots, DoF, and manipulation types across the primary datasets supported in OpenPI.

| Dataset | Robot Used | DoF (Control) | Tasks Included | Bi-manipulation |
| :--- | :--- | :--- | :--- | :--- |
| **DROID** | Franka Emika Panda | 8 (7-DoF arm + 1-DoF gripper) | Generalist table-top manipulation (pick-and-place, etc.) across 100+ diverse real-world environments. | No (Single Arm) |
| **ALOHA** | 2x Trossen ViperX 300 | 14 (2x [6-DoF arm + 1-DoF gripper]) | Bimanual skills: folding towels, opening tupperware, uncapping pens, and kitchen-related tasks. | **Yes** |
| **LIBERO** | Simulated Franka Panda | 7 or 8 (Depending on suite) | 130 simulated tasks across 4 suites: Spatial (spatial relations), Object (object properties), Goal (language goals), and Long-horizon. | No (Single Arm) |

