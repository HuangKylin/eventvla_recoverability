"""Bounded serial safe-boundary controls with resource monitoring and a lock."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
from resume_native_controls import atomic_json, resource_sample
from run_baseline import stop_group

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[100000, 100001])
    args = parser.parse_args()
    lock = (ROOT / 'runs/native_controls_resume.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial = resource_sample(args.gpu)
    gpu = next(g for g in initial['gpus'] if int(g['index']) == args.gpu)
    if initial['alarms'] or float(gpu['memory_MiB']) > 2048 or float(gpu['utilization']) > 10:
        raise RuntimeError('GPU or system is not available: ' + json.dumps(initial))
    old = json.loads(args.previous.read_text())
    sources = {e['seed']: str(Path(e['result_file']).parent) for e in old['episodes'] if e['condition'] == 'rollout'}
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_native_boundary_controls')
    run.mkdir()
    names = ['scripts/native_boundary_worker.py', 'scripts/run_native_boundary_controls.py',
             'src/recoverability/native_boundary.py', 'scripts/seeded_policy_server.py',
             'scripts/resume_native_controls.py', 'scripts/native_snapshot_audit.py',
             'src/recoverability/sim_runtime.py', 'src/recoverability/vla_intervention.py',
             'third_party/EventVLA/EventVLA/examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py',
             'third_party/EventVLA/RoboTwin-Mem/envs/cover_blocks_hard.py']
    hashes = {}
    for name in names:
        content = (ROOT / name).read_bytes(); hashes[name] = hashlib.sha256(content).hexdigest()
        dest = run / 'source' / name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(content)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    state = dict(status='loading', run_dir=str(run), pid=os.getpid(), port=port, gpu=args.gpu,
                 started_utc=datetime.now(timezone.utc).isoformat(), source_hashes=hashes,
                 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), episodes=[],
                 seeds=args.seeds, conditions=['prepare', 'native', 'native_repeat', 'position_other'],
                 max_actions=300, max_parent_extension=220, formal_training_eligible=False,
                 protocol='Finish native chunk; closed/released parent; language-only paired controls; no deletion or training.')
    def save():
        atomic_json(run / 'status.json', state); atomic_json(ROOT / 'reports/latest_native_boundary_controls.json', state)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               CUDA_HOME='/usr/local/cuda-12.8', TORCH_EXTENSIONS_DIR=str(ROOT / 'cache/torch_extensions'),
               XDG_CACHE_HOME=str(ROOT / 'cache/xdg'), PYTHONPATH=str(ROOT / 'src'), PYTHONUNBUFFERED='1')
    server = child = None; last_sample = 0.; alarms = 0
    resources = (run / 'resources.jsonl').open('a', buffering=1)
    def check():
        nonlocal last_sample, alarms
        if time.monotonic() - last_sample >= 10:
            sample = resource_sample(args.gpu); resources.write(json.dumps(sample)+'\n'); resources.flush()
            alarms = alarms + 1 if sample['alarms'] else 0; last_sample = time.monotonic()
            if alarms >= 3: raise RuntimeError('Resource guard: ' + repr(sample['alarms']))
        if server is not None and server.poll() is not None: raise RuntimeError('Policy server exited')
    def interrupt(signum, frame): raise KeyboardInterrupt('Signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
    save(); print(run, flush=True)
    try:
        with (run / 'server.log').open('w') as log:
            server = subprocess.Popen([str(ROOT / 'envs/policy/bin/python'), '-u', str(ROOT / 'scripts/seeded_policy_server.py'),
                '--port', str(port), '--report', str(run / 'server_ready.json')], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state['server_pid'] = server.pid; save(); deadline = time.monotonic()+600
        while time.monotonic() < deadline:
            check()
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=1): break
            except OSError: time.sleep(1)
        else: raise TimeoutError('Server readiness timeout')
        for seed in args.seeds:
            prepared = run / f'{seed}_prepare'
            for condition in state['conditions']:
                if condition != 'prepare' and not (prepared / 'parent.json').exists():
                    state.setdefault('skipped', []).append(dict(seed=seed, condition=condition, reason='No eligible boundary')); save(); continue
                output = run / f'{seed}_{condition}'
                job = dict(seed=seed, gpu=args.gpu, port=port, condition=condition, output=str(output),
                           source=sources[seed] if condition == 'prepare' else str(prepared), max_actions=300, max_extension=220)
                path = output.with_suffix('.job.json'); atomic_json(path, job)
                state.update(status='evaluating', current_job=str(path)); save()
                with output.with_suffix('.log').open('w') as log:
                    child = subprocess.Popen([str(ROOT / 'envs/sim/bin/python'), '-u', str(ROOT / 'scripts/native_boundary_worker.py'), str(path)],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    state['worker_pid'] = child.pid; save(); deadline = time.monotonic()+2400
                    while child.poll() is None:
                        check()
                        if time.monotonic() > deadline: raise TimeoutError(str(output))
                        time.sleep(2)
                    if child.returncode: raise RuntimeError('Worker failed: ' + str(output))
                stop_group(child); child = None
                result = json.loads((output/'result.json').read_text())
                if result['status'] != 'complete': raise RuntimeError('Incomplete result')
                row = {k: result[k] for k in ['parent_saved','first_lift_correct','first_reveal_correct',
                    'native_first_target_opened','native_failed','actions_executed','parent_match','action_hash',
                    'final_physics_hash','forbidden_calls'] if k in result}
                row.update(seed=seed, condition=condition, result_file=str(output/'result.json'), model_requests=len(result['request_audits']))
                state['episodes'].append(row); save()
        state.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
        state.pop('current_job', None); state.pop('worker_pid', None); save()
    except BaseException:
        state.update(status='failed', error=traceback.format_exc()); save(); raise
    finally:
        stop_group(child); stop_group(server); resources.close()
        state['process_cleanup_completed'] = True; save()


if __name__ == '__main__': main()
