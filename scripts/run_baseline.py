"""Run the original closed-loop evaluator with an isolated local policy server."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
import yaml

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'third_party/EventVLA/EventVLA'
SIM = ROOT / 'third_party/EventVLA/RoboTwin-Mem'


def stop_group(proc):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='cover_blocks_hard')
    p.add_argument('--episodes', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--memory-policy', choices=['upstream', 'fifo', 'uniform'], default='upstream')
    p.add_argument('--history-capacity', type=int, default=None,
                   help='Override event-memory capacity in both model and client configuration')
    p.add_argument('--timeout-seconds', type=int, default=3600)
    args = p.parse_args()
    if args.episodes < 1:
        p.error('--episodes must be positive')
    if args.history_capacity is not None and args.history_capacity < 0:
        p.error('--history-capacity must be non-negative')
    smoke = json.loads((ROOT / 'reports/smoke_policy.json').read_text())
    if smoke['status'] != 'complete':
        raise RuntimeError('Strict checkpoint and inference smoke must pass first')
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + args.task
    run = ROOT / 'runs' / run_id
    run.mkdir()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    shared_runtime = ROOT / 'checkpoints/runtime/EventVLA-RoboTwin-MeM'
    runtime = run / 'checkpoint_runtime'
    (runtime / 'final_model').mkdir(parents=True)
    config = yaml.safe_load((shared_runtime / 'config.yaml').read_text())
    if args.history_capacity is not None:
        config['framework']['memory_buffer']['qwen_memory_injection']['max_keyframe_images'] = args.history_capacity
        config['datasets']['vla_data']['keyframe_image_memory']['max_keyframes'] = args.history_capacity
    history_capacity = config['framework']['memory_buffer']['qwen_memory_injection']['max_keyframe_images']
    (runtime / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (runtime / 'dataset_statistics.json').write_bytes((shared_runtime / 'dataset_statistics.json').read_bytes())
    ckpt = runtime / 'final_model/pytorch_model.pt'
    ckpt.symlink_to((shared_runtime / 'final_model/pytorch_model.pt').resolve())
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_HOME='/usr/local/cuda-12.8',
               TORCH_EXTENSIONS_DIR=str(ROOT / 'cache/torch_extensions'),
               XDG_CACHE_HOME=str(ROOT / 'cache/xdg'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               EVENTVLA_RAY_DENOISER='none', EVENTVLA_RT_SPP='128',
               EVENTVLA_EVAL_EPISODES=str(args.episodes),
               EVENTVLA_PYTHON=str(ROOT / 'envs/policy/bin/python'),
               ROBOTWIN_MEM_PYTHON=str(ROOT / 'envs/sim/bin/python'),
               ROBOTWIN_MEM_ROOT=str(SIM), POLICY_CKPT_PATH=str(ckpt),
               HOST='127.0.0.1', PORT=str(port), UNNORM_KEY='robotwin_mem',
               AUTO_START_POLICY_SERVER='0', SERVER_READY_TIMEOUT='600',
               ROBOTWIN_EVAL_VIDEO_TMPDIR=str(run / 'video_tmp'),
               PYTHONPATH=str(ROOT / 'src'), PYTHONUNBUFFERED='1',
               PATH='/usr/local/cuda-12.8/bin:' + env.get('PATH', ''))
    if args.memory_policy != 'upstream':
        env['RECOVERABILITY_MEMORY_POLICY'] = args.memory_policy
        env['RECOVERABILITY_MEMORY_SEED'] = str(args.seed)
    else:
        env.pop('RECOVERABILITY_MEMORY_POLICY', None)
    state = dict(status='starting', run_id=run_id, pid=os.getpid(), task=args.task,
                 requested_episodes=args.episodes, seed=args.seed, physical_gpu=args.gpu,
                 checkpoint=str(ckpt), memory_policy=args.memory_policy,
                 history_capacity=history_capacity,
                 attention='sdpa', denoiser='none', ray_samples=128,
                 benchmark_caveat='Local compatibility configuration; not an exact published-score reproduction')

    def update(**values):
        state.update(values)
        (run / 'status.json').write_text(json.dumps(state, indent=2))
        (ROOT / 'reports/latest_baseline.json').write_text(json.dumps(state, indent=2))
        print(json.dumps(values), flush=True)

    server = client = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with (run / 'server.log').open('w') as log:
            server = subprocess.Popen(['bash', str(CODE / 'examples/RoboTwin-Mem/eval_files/run_policy_server.sh'),
                                       str(ckpt), str(args.gpu), str(port)], cwd=CODE, env=env,
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        update(status='loading_policy', server_pid=server.pid, port=port)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError('Policy server exited; see server.log')
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError('Policy server did not become ready within 600 seconds')
        update(status='evaluating')
        cmd = ['bash', str(CODE / 'examples/RoboTwin-Mem/eval_files/eval.sh'),
               args.task, 'demo_clean', run_id, str(args.seed), str(args.gpu)]
        with (run / 'evaluation.log').open('w') as log:
            client = subprocess.Popen(cmd, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        update(client_pid=client.pid)
        rc = client.wait(timeout=args.timeout_seconds)
        if rc:
            raise RuntimeError('Evaluation exited with code ' + str(rc))
        results = list((SIM / 'eval_result' / args.task).rglob('_result.txt'))
        results = [str(f) for f in results if run_id in str(f)]
        if not results:
            raise RuntimeError('Evaluator exited without a result file')
        update(status='complete', result_files=results, exit_code=rc)
    except BaseException:
        update(status='failed', error=traceback.format_exc())
        raise
    finally:
        stop_group(client)
        stop_group(server)


if __name__ == '__main__':
    main()
