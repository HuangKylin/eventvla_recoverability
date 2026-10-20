"""Run exactly the fixed native census serially, with durable progress and guards.

Paired branches require a separate, reviewed selection manifest; this entry
point never invents missing evidence labels or automatically starts training.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from recoverability.evidence_diagnostic import validate_native_jobs
from conservative_execution import load_profile, conservative_sample, rest_between_jobs
from resume_native_controls import atomic_json, resource_sample
from run_baseline import stop_group


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    contract = json.loads((ROOT/'configs/evidence_validity_diagnostic_v1.json').read_text())
    jobs = [(t,s) for t in contract['tasks'] for s in contract['development_seeds_per_task']]
    validate_native_jobs(jobs)
    if len(jobs) > contract['max_native_episodes']:
        raise ValueError('Native census exceeds frozen cap')
    if sha(ROOT/contract['native_task_list']) != contract['native_task_list_sha256']:
        raise ValueError('Official task list changed')
    profile = load_profile(ROOT)
    lock = (ROOT/'runs/native_controls_resume.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sample = conservative_sample(resource_sample(args.gpu), args.gpu, profile)
    gpu = next(g for g in sample['gpus'] if int(g['index']) == args.gpu)
    if sample['alarms'] or float(gpu['memory_MiB']) > 2048 or float(gpu['utilization']) > 10:
        raise RuntimeError('Selected GPU/system unavailable: '+json.dumps(sample))
    if shutil.disk_usage(ROOT).free < 20*1024**3:
        raise RuntimeError('Require at least 20 GiB free for exact observations and requests')
    names = ['scripts/evidence_diagnostic_worker.py', 'scripts/run_evidence_diagnostic.py',
             'scripts/seeded_policy_server.py', 'scripts/conservative_execution.py',
             'scripts/resume_native_controls.py', 'scripts/run_baseline.py',
             'src/recoverability/evidence_diagnostic.py', 'src/recoverability/snapshot.py',
             'src/recoverability/sim_runtime.py', 'src/recoverability/vla_intervention.py',
             'configs/evidence_validity_diagnostic_v1.json', 'configs/conservative_execution.json',
             'checkpoints/runtime/EventVLA-RoboTwin-MeM/config.yaml',
             'checkpoints/runtime/EventVLA-RoboTwin-MeM/dataset_statistics.json',
             'third_party/EventVLA/EventVLA/examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py',
             'third_party/EventVLA/RoboTwin-Mem/envs/_base_task.py',
             'third_party/EventVLA/RoboTwin-Mem/task_config/_eval_step_limit.yml']
    for task in contract['tasks']:
        names += [f'third_party/EventVLA/RoboTwin-Mem/envs/{task}.py',
                  f'third_party/EventVLA/RoboTwin-Mem/description/task_instruction/{task}.json']
    hashes = {name:sha(ROOT/name) for name in names}
    checkpoint = ROOT/'checkpoints/runtime/EventVLA-RoboTwin-MeM/final_model/pytorch_model.pt'
    checkpoint_sha = sha(checkpoint)
    preflight = dict(observed_utc=datetime.now(timezone.utc).isoformat(), resource=sample,
                     checkpoint=str(checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
                     source_hashes=hashes, gpu_jobs_started=0)
    atomic_json(ROOT/'reports/evidence_diagnostic_preflight.json', preflight)
    if args.preflight_only:
        print(json.dumps(preflight), flush=True)
        return
    run = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_evidence_native_census')
    run.mkdir()
    for name in names:
        dest = run/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes())
    inherited = []
    old = None
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if args.resume:
        old = json.loads((args.resume/'status.json').read_text())
        if old['source_hashes'] != hashes or old['checkpoint_sha256'] != checkpoint_sha:
            raise ValueError('Resume source/checkpoint changed')
        if old['boot_id'] == boot and not old.get('process_cleanup_completed'):
            raise ValueError('Previous batch may still be running')
        for row in old['episodes']:
            path = Path(row['result_file'])
            if sha(path) != row['result_sha256'] or json.loads(path.read_text())['status'] != 'complete':
                raise ValueError('Inherited completed result changed')
            inherited.append(dict(row, inherited_from=old['run_dir']))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    state = dict(status='loading', run_dir=str(run), pid=os.getpid(), gpu=args.gpu,
                 port=port, boot_id=boot, source_hashes=hashes, checkpoint_sha256=checkpoint_sha,
                 episodes=inherited, jobs=jobs, execution_profile=profile,
                 started_utc=datetime.now(timezone.utc).isoformat(),
                 formal_training_eligible=False, diagnostic_branches_started=0,
                 phase='native_census', process_cleanup_completed=False)
    if old:
        state['resumed_from'] = old['run_dir']
    def save():
        atomic_json(run/'status.json', state)
        atomic_json(ROOT/'reports/latest_evidence_native_census.json', state)
    child = server = None
    last_sample, alarms = 0., 0
    resources = (run/'resources.jsonl').open('a', buffering=1)
    def check():
        nonlocal last_sample, alarms
        if time.monotonic()-last_sample >= profile['monitor_interval_seconds']:
            item = conservative_sample(resource_sample(args.gpu), args.gpu, profile)
            resources.write(json.dumps(item)+'\n')
            resources.flush()
            os.fsync(resources.fileno())
            alarms = alarms+1 if item['alarms'] else 0
            last_sample = time.monotonic()
            if alarms >= profile['consecutive_alarm_limit']:
                raise RuntimeError('Resource guard: '+repr(item['alarms']))
        if server is not None and server.poll() is not None:
            raise RuntimeError('Policy server exited')
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               CUDA_HOME='/usr/local/cuda-12.8', TORCH_EXTENSIONS_DIR=str(ROOT/'cache/torch_extensions'),
               XDG_CACHE_HOME=str(ROOT/'cache/xdg'), PYTHONPATH=str(ROOT/'src'),
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               EVENTVLA_RAY_DENOISER='none', EVENTVLA_RT_SPP='128', TOKENIZERS_PARALLELISM='false')
    env.pop('RECOVERABILITY_MEMORY_POLICY', None)
    save()
    print(str(run), flush=True)
    try:
        with (run/'server.log').open('w') as log:
            server = subprocess.Popen([str(ROOT/'envs/policy/bin/python'), '-u',
                str(ROOT/'scripts/seeded_policy_server.py'), '--port', str(port),
                '--report', str(run/'server_ready.json')], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state['server_pid'] = server.pid
        save()
        deadline = time.monotonic()+600
        while not (run/'server_ready.json').exists():
            check()
            if time.monotonic()>deadline:
                raise TimeoutError('Server readiness')
            time.sleep(1)
        # Readiness file is written immediately before the listening loop.
        while True:
            check()
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=1):
                    break
            except OSError:
                if time.monotonic()>deadline:
                    raise TimeoutError('Server socket readiness')
                time.sleep(1)
        for task, seed in jobs:
            if any(r['task']==task and r['seed']==seed for r in inherited):
                continue
            if state['episodes']:
                state['status'] = 'inter_job_rest'
                save()
                rest_between_jobs(profile['inter_job_rest_seconds'], check)
            out = run/f'{task}_{seed}'
            job = dict(task=task, seed=seed, gpu=args.gpu, port=port, output=str(out), mode='native')
            jobpath = out.with_suffix('.job.json')
            atomic_json(jobpath, job)
            state.update(status='evaluating', current_job=str(jobpath))
            save()
            with out.with_suffix('.log').open('w') as log:
                child = subprocess.Popen([str(ROOT/'envs/sim/bin/python'), '-u',
                    str(ROOT/'scripts/evidence_diagnostic_worker.py'), str(jobpath)], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state['worker_pid'] = child.pid
                save()
                deadline = time.monotonic()+3600
                while child.poll() is None:
                    check()
                    if time.monotonic()>deadline:
                        raise TimeoutError('Worker wall-clock limit; not a policy failure')
                    heartbeat = out/'result.json'
                    if heartbeat.exists() and time.time()-heartbeat.stat().st_mtime>240:
                        raise TimeoutError('Worker progress missing for 240 s; not a policy failure')
                    time.sleep(2)
                if child.returncode:
                    raise RuntimeError('Worker failed: '+str(out))
            stop_group(child)
            child = None
            result = json.loads((out/'result.json').read_text())
            if result['status'] != 'complete':
                raise RuntimeError('Incomplete worker result')
            state['episodes'].append({k:result[k] for k in (
                'native_success', 'native_failed', 'native_step_limit_reached',
                'candidate_saved', 'candidate_step', 'actions_executed',
                'native_success_without_both_returns', 'elapsed_seconds') if k in result} |
                dict(task=task, seed=seed, result_file=str(out/'result.json'),
                     result_sha256=sha(out/'result.json'), requests=len(result['request_audits'])))
            save()
        state.update(status='complete', next_action='review actual failures before admitting paired branches')
        save()
    except BaseException:
        state.update(status='failed', error=traceback.format_exc())
        save()
        raise
    finally:
        stop_group(child)
        stop_group(server)
        state['process_cleanup_completed'] = True
        save()
        resources.close()


if __name__ == '__main__':
    main()
