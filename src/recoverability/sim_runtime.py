"""Shared, project-local RoboTwin setup. Import after selecting CUDA_VISIBLE_DEVICES."""
from copy import deepcopy
import importlib
import os
from pathlib import Path
import random
import sys


def configure(root: Path, gpu: str = '1') -> Path:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', gpu)
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(root / 'cache/torch_extensions'))
    os.environ.setdefault('XDG_CACHE_HOME', str(root / 'cache/xdg'))
    os.environ.setdefault('EVENTVLA_RAY_DENOISER', 'none')
    os.environ.setdefault('EVENTVLA_RT_SPP', '128')
    sim = root / 'third_party/EventVLA/RoboTwin-Mem'
    sys.path.insert(0, str(sim))
    os.chdir(sim)
    return sim


def make_env(root: Path, task: str, seed: int, *, task_class=None, **overrides):
    import yaml
    sim = root / 'third_party/EventVLA/RoboTwin-Mem'
    cfg = yaml.safe_load((sim / 'task_config/demo_clean.yml').read_text())
    camera = yaml.safe_load((sim / 'task_config/_camera_config.yml').read_text())
    robot = sim / 'assets/embodiments/aloha-agilex'
    robot_cfg = yaml.safe_load((robot / 'config.yml').read_text())
    cfg.update(task_name=task, task_config='demo_clean', render_freq=0,
               save_data=False, collect_data=False, eval_video_log=False,
               eval_mode=True, dual_arm_embodied=True,
               left_robot_file=str(robot), right_robot_file=str(robot),
               left_embodiment_config=deepcopy(robot_cfg),
               right_embodiment_config=deepcopy(robot_cfg),
               head_camera_h=camera[cfg['camera']['head_camera_type']]['h'],
               head_camera_w=camera[cfg['camera']['head_camera_type']]['w'],
               save_path=str(root / 'runs/diagnostics'),
               log_file=str(root / 'logs/diagnostic_task.log'))
    cfg.update(overrides)
    if task_class is None:
        task_class = getattr(importlib.import_module('envs.' + task), task)
    random.seed(seed)
    env = task_class()
    env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **cfg)
    return env


def close_env(env):
    """Join only this environment's own planner workers."""
    robot = getattr(env, 'robot', None)
    try:
        env.close_env()
    finally:
        if robot is not None:
            for side in ('left', 'right'):
                conn = getattr(robot, side + '_conn', None)
                if conn is not None:
                    conn.close()
                proc = getattr(robot, side + '_proc', None)
                if proc is not None:
                    if proc.is_alive():
                        proc.terminate()
                    proc.join(timeout=5)


def public_observation(obs):
    """Copy a strict whitelist; simulator poses/oracle metadata never cross this API."""
    import numpy as np
    return {
        'images': {view: np.array(obs['observation'][view]['rgb'], copy=True)
                   for view in ('head_camera', 'left_camera', 'right_camera')},
        'proprioception': np.array(obs['joint_action']['vector'], copy=True),
    }
