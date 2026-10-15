"""Exercise the original simulator; expert success is not VLA policy success."""
import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / 'third_party/EventVLA/RoboTwin-Mem'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT / 'cache/torch_extensions'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / 'cache/xdg'))
sys.path.insert(0, str(SIM))


def describe(value):
    if isinstance(value, dict):
        return {k: describe(v) for k, v in value.items()}
    if hasattr(value, 'shape'):
        return dict(shape=list(value.shape), dtype=str(value.dtype))
    if isinstance(value, (tuple, list)):
        return dict(type=type(value).__name__, length=len(value))
    return type(value).__name__


def main():
    import numpy as np
    import yaml
    from PIL import Image
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='cover_blocks_hard')
    p.add_argument('--seed', type=int, default=100000)
    p.add_argument('--expert', action='store_true')
    args = p.parse_args()
    os.chdir(SIM)
    report = ROOT / 'reports' / ('smoke_sim_' + args.task + '.json')
    outcome = dict(task=args.task, seed=args.seed, kind='privileged_expert' if args.expert else 'environment_only',
                   status='running', pid=os.getpid(), stage='imports')

    def update(**values):
        outcome.update(values)
        report.write_text(json.dumps(outcome, indent=2))
        print(json.dumps(values), flush=True)

    env = None
    start = time.monotonic()
    try:
        cfg = yaml.safe_load((SIM / 'task_config/demo_clean.yml').read_text())
        camera = yaml.safe_load((SIM / 'task_config/_camera_config.yml').read_text())
        robot = SIM / 'assets/embodiments/aloha-agilex'
        robot_cfg = yaml.safe_load((robot / 'config.yml').read_text())
        cfg.update(task_name=args.task, task_config='demo_clean', render_freq=0,
                   save_data=False, collect_data=False, eval_video_log=False,
                   eval_mode=True, dual_arm_embodied=True,
                   left_robot_file=str(robot), right_robot_file=str(robot),
                   left_embodiment_config=deepcopy(robot_cfg),
                   right_embodiment_config=deepcopy(robot_cfg),
                   head_camera_h=camera[cfg['camera']['head_camera_type']]['h'],
                   head_camera_w=camera[cfg['camera']['head_camera_type']]['w'],
                   save_path=str(ROOT / 'runs/smoke_sim'),
                   log_file=str(ROOT / 'logs/smoke_sim_task.log'))
        module = importlib.import_module('envs.' + args.task)
        env = getattr(module, args.task)()
        update(stage='setup_demo')
        env.setup_demo(now_ep_num=0, seed=args.seed, is_test=True, **cfg)
        update(stage='get_obs')
        obs = env.get_obs()
        update(observation_schema=describe(obs))
        cameras = obs['observation']
        sample_path = ROOT / 'reports' / ('smoke_' + args.task + '_observation.npz')
        np.savez_compressed(sample_path, head=cameras['head_camera']['rgb'],
                            left=cameras['left_camera']['rgb'],
                            right=cameras['right_camera']['rgb'],
                            state=obs['joint_action']['vector'])
        update(observation_file=str(sample_path),
               denoiser=os.environ.get('EVENTVLA_RAY_DENOISER', 'oidn'))

        def save_rgb(value):
            if not isinstance(value, dict):
                return False
            if 'rgb' in value:
                rgb = np.asarray(value['rgb'])
                if rgb.ndim == 3 and rgb.shape[-1] >= 3:
                    if rgb.dtype != np.uint8:
                        rgb = np.clip(rgb * 255 if rgb.max() <= 1 else rgb, 0, 255).astype(np.uint8)
                    dest = ROOT / 'reports' / ('smoke_' + args.task + '.png')
                    Image.fromarray(rgb[..., :3]).save(dest)
                    update(render_image=str(dest))
                    return True
            return any(save_rgb(v) for v in value.values())

        if not save_rgb(cameras['head_camera']):
            raise RuntimeError('No RGB observation was produced')
        if args.expert:
            update(stage='expert_rollout')
            env.play_once()
            update(expert_plan_success=bool(env.plan_success),
                   expert_task_success=bool(env.check_success()))
        update(status='complete', stage='finished', elapsed_seconds=time.monotonic()-start)
    except Exception:
        update(status='failed', error=traceback.format_exc(), elapsed_seconds=time.monotonic()-start)
        raise
    finally:
        if env is not None and hasattr(env, 'scene'):
            env.close_env()


if __name__ == '__main__':
    main()
