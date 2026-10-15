"""Serial native rollouts and matched-parent controls on one explicitly selected GPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
from run_baseline import stop_group

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--seeds', type=int, nargs='+', default=[100000, 100001])
    p.add_argument('--max-actions', type=int, default=300)
    p.add_argument('--conditions', nargs='+', choices=['native', 'position', 'color', 'native_repeat'],
                   default=['native', 'position', 'color', 'native_repeat'])
    a = p.parse_args()
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_native_controls')
    run.mkdir()
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               CUDA_HOME='/usr/local/cuda-12.8', TORCH_EXTENSIONS_DIR=str(ROOT / 'cache/torch_extensions'),
               XDG_CACHE_HOME=str(ROOT / 'cache/xdg'), PYTHONPATH=str(ROOT / 'src'), PYTHONUNBUFFERED='1')
    names = ['scripts/native_control_worker.py', 'scripts/run_native_controls.py', 'scripts/seeded_policy_server.py',
             'src/recoverability/sim_runtime.py', 'src/recoverability/vla_intervention.py',
             'third_party/EventVLA/EventVLA/examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py']
    hashes = {}
    for name in names:
        content = (ROOT / name).read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
        dest = run / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    state = dict(status='loading', run_dir=str(run), pid=os.getpid(), gpu=a.gpu, port=port,
                 source_hashes=hashes, episodes=[], formal_training_eligible=False,
                 protocol='Native rollout then replay identical native parent; cache cleared equally; language only interventions.')
    def save():
        text = json.dumps(state, indent=2)
        (run / 'status.json').write_text(text)
        (ROOT / 'reports/latest_native_controls.json').write_text(text)
    server = child = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    try:
        with (run / 'server.log').open('w') as log:
            server = subprocess.Popen([str(ROOT / 'envs/policy/bin/python'), '-u', str(ROOT / 'scripts/seeded_policy_server.py'),
                                       '--port', str(port), '--report', str(run / 'server_ready.json')],
                                      cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state['server_pid'] = server.pid
        save()
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if server.poll() is not None: raise RuntimeError('Policy server failed')
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=1): break
            except OSError: time.sleep(1)
        else: raise TimeoutError('Policy server readiness timeout')
        for seed in a.seeds:
            parent = run / f'{seed}_rollout'
            for condition in ['rollout'] + a.conditions:
                if condition != 'rollout' and not (parent / 'parent.json').exists():
                    state.setdefault('skipped', []).append(dict(seed=seed, condition=condition, reason='Native prefix failed'))
                    save()
                    continue
                out = run / f'{seed}_{condition}'
                job = dict(seed=seed, condition=condition, output=str(out), gpu=a.gpu, port=port,
                           max_actions=a.max_actions, source=None if condition == 'rollout' else str(parent))
                jobpath = out.with_suffix('.job.json')
                jobpath.write_text(json.dumps(job, indent=2))
                state.update(status='evaluating', current_job=str(jobpath))
                save()
                with out.with_suffix('.log').open('w') as log:
                    child = subprocess.Popen([str(ROOT / 'envs/sim/bin/python'), '-u', str(ROOT / 'scripts/native_control_worker.py'), str(jobpath)],
                                             cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        if child.wait(timeout=2400): raise RuntimeError('Worker failed: ' + str(out))
                    finally:
                        stop_group(child)
                        child = None
                result = json.loads((out / 'result.json').read_text())
                state['episodes'].append({k: result[k] for k in ('native_success', 'intended_cover_actuated', 'actions_executed',
                    'parent_match', 'events', 'action_hash', 'final_physics_hash', 'forbidden_calls') if k in result} |
                    dict(seed=seed, condition=condition, result_file=str(out / 'result.json'), model_requests=len(result['request_audits'])))
                save()
        state.update(status='complete')
        save()
    except BaseException:
        state.update(status='failed', error=traceback.format_exc())
        save()
        raise
    finally:
        stop_group(child)
        stop_group(server)


if __name__ == '__main__':
    main()
