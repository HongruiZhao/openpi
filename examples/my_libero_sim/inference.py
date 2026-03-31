import argparse
import collections
import logging
import logging.handlers as handlers
import math
import os
import pathlib
import re
import sys

import imageio
import numpy as np
from tqdm import trange 
import yaml
from PIL import Image
from pythonjsonlogger import jsonlogger
import warnings


# Automatically add LIBERO to PYTHONPATH
libero_path = str(pathlib.Path(__file__).parent.parent.parent / "third_party" / "libero")
if libero_path not in sys.path:
    sys.path.insert(0, libero_path)

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


logger = logging.getLogger("inference")
def _setup_logging(video_out_path:str, log_name:str):
    log_file = pathlib.Path(video_out_path) / (log_name + ".json")
    logger.setLevel(logging.INFO)
    
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    file_handler = handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=2
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def _load_policy(config_name, checkpoint_dir, analysis_flag):
    logger.info(f"Loading policy {config_name} from {checkpoint_dir}")
    config = _config.get_config(config_name)
    checkpoint_path = download.maybe_download(checkpoint_dir)
    return policy_config.create_trained_policy(config, checkpoint_path, analysis_flag=analysis_flag)


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
    num_epoch = args.get("epoch", 5)
    replay_images = []
    replay_analysis = []

    for epoch in trange(num_epoch):
        env.reset()
        obs = env.set_init_state(initial_states[epoch])
        
        action_plan = collections.deque()
        for t in trange(max_steps + num_steps_wait):
            if t < num_steps_wait:
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
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
                
                infer_result = policy.infer(input_data)
                action_chunk = infer_result["actions"]
                action_plan.extend(action_chunk)

                # Store attention if it exists
                if "prefix_attention" in infer_result:
                    replay_analysis.append({
                        "epoch": epoch,
                        "step": t,
                        "prefix_attention": infer_result["prefix_attention"],
                        "step_attention": infer_result["step_attention"],
                    })
                
            action = action_plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            
            if done:
                logger.info(f"Task successful! time: {t},epoch:{epoch}")
                break
        
    return replay_images, replay_analysis


def _save_video(video_out_path: str, log_name: str, 
                replay_images: list[np.typing.ArrayLike]):
    video_path = pathlib.Path(video_out_path) / (log_name + '.mp4')
    logger.info(f"Saving video to {video_path}")
    imageio.mimwrite(video_path, replay_images, fps=240)


def _save_analysis(video_out_path: str, log_name: str, 
                   replay_analysis: list[dict]):
    if not replay_analysis:
        return
    analysis_path = pathlib.Path(video_out_path) / (log_name + '_analysis.npz')
    logger.info(f"Saving analysis to {analysis_path}")
    # Convert list of dicts to a single dict of arrays for efficient saving.
    # We convert attention weights to float32 to ensure they are correctly saved and 
    # loaded by NumPy, avoiding issues with bfloat16 (which often loads as |V2).
    save_dict = {
        "prefix_attention": np.array([x["prefix_attention"] for x in replay_analysis]).astype(np.float32),
        "step_attention": np.array([x["step_attention"] for x in replay_analysis]).astype(np.float32),
        "epochs": np.array([x["epoch"] for x in replay_analysis]),
        "steps": np.array([x["step"] for x in replay_analysis]),
    }
    np.savez_compressed(analysis_path, **save_dict)


def run_inference(args: dict):
    config_name = args.get("config_name", "pi05_libero")
    checkpoint_dir = args.get("checkpoint_dir", "gs://openpi-assets/checkpoints/pi05_libero")
    checkpoint_name = os.path.basename(checkpoint_dir.rstrip("/"))
    video_out_path = args.get("video_out_path", "examples/my_libero_sim/videos")
    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)
    task_suite_name = args.get("task_suite_name", "libero_10")

    analysis_flag = args.get('analysis', {})

    try:
        env, task_description, initial_states, _, task__id = _init_libero(args)
        log_name =  checkpoint_name + "_" + task_suite_name + "_" + str(task__id) + "_" + task_description
        _setup_logging(video_out_path, log_name)
        
        policy = _load_policy(config_name, checkpoint_dir, analysis_flag)
        logger.info(f"Task: {task_description}")
        replay_images, replay_analysis = _run_loop(env, policy, task_description, initial_states, args)
        
        _save_video(video_out_path, log_name, replay_images)
        _save_analysis(video_out_path, log_name, replay_analysis)
        env.close()
    except Exception as e:
        logger.exception(f"An error occurred: {e}", extra={"error": str(e)})
    finally:
        logger.info('program finished!')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(pathlib.Path(__file__).parent / "configs/inference_config.yaml"),
        help="Path to the YAML config file",
    )
    cmd_args = parser.parse_args()
    
    args = {}
    config_path = pathlib.Path(cmd_args.config)
    if config_path.exists():
        with open(config_path) as f:
            args = yaml.safe_load(f)
    else:
        logger.warning(f"Config file {cmd_args.config} not found. Using default values.")
            
    run_inference(args)
