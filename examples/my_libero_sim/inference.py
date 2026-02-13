import argparse
import collections
import logging
import math
import os
import pathlib
import re
import sys

import imageio
import numpy as np
import tqdm
import yaml
from PIL import Image

# Automatically add LIBERO to PYTHONPATH
libero_path = str(pathlib.Path(__file__).parent.parent.parent / "third_party" / "libero")
if libero_path not in sys.path:
    sys.path.insert(0, libero_path)

# Automatically set LIBERO_CONFIG_PATH if not set
if "LIBERO_CONFIG_PATH" not in os.environ:
    config_path = pathlib.Path("/tmp/libero_config")
    config_path.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_path)
    
    config_file = config_path / "config.yaml"
    if not config_file.exists():
        repo_root = pathlib.Path(__file__).parent.parent.parent.absolute()
        libero_root = repo_root / "third_party" / "libero" / "libero" / "libero"
        with open(config_file, "w") as f:
            f.write(f"""benchmark_root: {libero_root}
bddl_files: {libero_root}/bddl_files
init_files: {libero_root}/init_files
datasets: {repo_root}/third_party/libero/datasets
assets: {libero_root}/assets
""")


# Import LIBERO
try:
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError:
    print(f"LIBERO not found at {libero_path}. Please check the path.")
    raise

# Import OpenPI
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _setup_logging(video_out_path):
    log_file = pathlib.Path(video_out_path) / "inference.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    if root_logger.getEffectiveLevel() > logging.INFO:
        root_logger.setLevel(logging.INFO)
    return log_file, file_handler, root_logger


def _load_policy(config_name, checkpoint_dir):
    logging.info(f"Loading policy {config_name} from {checkpoint_dir}")
    config = _config.get_config(config_name)
    checkpoint_path = download.maybe_download(checkpoint_dir)
    return policy_config.create_trained_policy(config, checkpoint_path)


def _init_libero(args):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = args.get("task_suite_name", "libero_10")
    task_id = args.get("task_id", 0)
    
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    
    seed = args.get("seed", 7)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
    
    if args.get("task_description"):
        task_description = args["task_description"]
        
    return env, task_description, initial_states, task_suite_name, task_id


def _run_loop(env, policy, task_description, initial_states, args):
    num_steps_wait = args.get("num_steps_wait", 10)
    max_steps = args.get("max_steps", 500)
    resize_size = args.get("resize_size", 224)
    
    env.reset()
    obs = env.set_init_state(initial_states[0])
    
    action_plan = collections.deque()
    replay_images = []
    success = False
    
    pbar = tqdm.tqdm(total=max_steps + num_steps_wait)
    for t in range(max_steps + num_steps_wait):
        if t < num_steps_wait:
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            pbar.update(1)
            continue
            
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        replay_images.append(img)
        
        if not action_plan:
            state = np.concatenate((
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ))
            
            def resize(img):
                return np.array(Image.fromarray(img).resize((resize_size, resize_size), Image.BILINEAR))

            input_data = {
                "observation/image": resize(img),
                "observation/wrist_image": resize(wrist_img),
                "observation/state": state,
                "prompt": str(task_description),
            }
            
            action_chunk = policy.infer(input_data)["actions"]
            action_plan.extend(action_chunk)
            
        action = action_plan.popleft()
        obs, _, done, _ = env.step(action.tolist())
        
        if done:
            success = True
            logging.info("Task successful!")
            break
        pbar.update(1)
        
    pbar.close()
    return success, replay_images


def _save_video(video_out_path, replay_images, task_description, success, checkpoint_name):
    sanitized = re.sub(r'[^\w\s-]', '', task_description).strip().lower().replace(' ', '_')
    video_name = f"libero_{checkpoint_name}_{sanitized}_{'success' if success else 'failure'}.mp4"
    video_path = pathlib.Path(video_out_path) / video_name
    logging.info(f"Saving video to {video_path}")
    imageio.mimwrite(video_path, replay_images, fps=10)


def _finalize_logging(file_handler, root_logger, log_file, video_out_path, success, task_description, task_suite_name, task_id, checkpoint_name):
    file_handler.flush()
    root_logger.removeHandler(file_handler)
    file_handler.close()
    
    try:
        sanitized = re.sub(r'[^\w\s-]', '', task_description).strip().lower().replace(' ', '_')
        log_name = f"libero_{checkpoint_name}_{sanitized}_{'success' if success else 'failure'}.log"
    except (NameError, TypeError):
        log_name = f"libero_{checkpoint_name}_{task_suite_name}_task{task_id}_{'success' if success else 'failure'}.log"
            
    dest_path = pathlib.Path(video_out_path) / log_name
    if dest_path.exists():
        dest_path.unlink()
    log_file.rename(dest_path)


def run_inference(args: dict):
    config_name = args.get("config_name", "pi05_libero")
    checkpoint_dir = args.get("checkpoint_dir", "gs://openpi-assets/checkpoints/pi05_libero")
    checkpoint_name = os.path.basename(checkpoint_dir.rstrip("/"))
    video_out_path = args.get("video_out_path", "examples/libero_sim/videos")
    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)

    log_file, file_handler, root_logger = _setup_logging(video_out_path)
    success = False
    task_description = None
    task_suite_name = args.get("task_suite_name", "libero_10")
    task_id = args.get("task_id", 0)

    try:
        policy = _load_policy(config_name, checkpoint_dir)
        env, task_description, initial_states, task_suite_name, task_id = _init_libero(args)
        
        logging.info(f"Task: {task_description}")
        success, replay_images = _run_loop(env, policy, task_description, initial_states, args)
        
        _save_video(video_out_path, replay_images, task_description, success, checkpoint_name)
        env.close()
    finally:
        _finalize_logging(file_handler, root_logger, log_file, video_out_path, success, task_description, task_suite_name, task_id, checkpoint_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(pathlib.Path(__file__).parent / "config.yaml"),
        help="Path to the YAML config file",
    )
    cmd_args = parser.parse_args()
    
    args = {}
    config_path = pathlib.Path(cmd_args.config)
    if config_path.exists():
        with open(config_path) as f:
            args = yaml.safe_load(f)
    else:
        logging.warning(f"Config file {cmd_args.config} not found. Using default values.")
            
    run_inference(args)
