"""Collect predeclared fresh parents; retain every failed or rejected attempt."""
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
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(410000,410020)))
    parser.add_argument('--resume', type=Path, help='Continue a cleaned-up failed batch in a new run directory')
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds))!=len(args.seeds) or not set(args.seeds)<=set(range(410000,410020)):
        raise ValueError('Use unique predeclared fresh seeds 410000..410019')
    lock = (ROOT / 'runs/native_controls_resume.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial = resource_sample(args.gpu)
    gpu = next(g for g in initial['gpus'] if int(g['index']) == args.gpu)
    if initial['alarms'] or float(gpu['memory_MiB']) > 2048 or float(gpu['utilization']) > 10:
        raise RuntimeError('GPU or system is not available: ' + json.dumps(initial))
    library=json.loads((ROOT/'reports/latest_shared_skill_library.json').read_text())['run_dir']+'/library.json'
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_native_parent_collection')
    run.mkdir()
    names = ['scripts/native_parent_worker.py', 'scripts/run_native_parent_collection.py',
             'scripts/native_boundary_worker.py', 'src/recoverability/camera_capture.py',
             'src/recoverability/fixed_skills.py', 'src/recoverability/cue_reader.py',
             'src/recoverability/native_items.py','src/recoverability/native_recovery.py',
             'src/recoverability/native_memory.py','src/recoverability/snapshot.py',
             'scripts/native_recovery_vla_worker.py','scripts/validate_fresh_parent_replay.py',
             'scripts/audit_native_parent_inputs.py',
             'src/recoverability/native_boundary.py', 'scripts/seeded_policy_server.py',
             'scripts/resume_native_controls.py', 'scripts/native_snapshot_audit.py',
             'src/recoverability/sim_runtime.py', 'src/recoverability/vla_intervention.py',
             'third_party/EventVLA/EventVLA/examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py',
             'third_party/EventVLA/RoboTwin-Mem/envs/cover_blocks_hard.py']
    hashes = {}
    for name in names:
        content = (ROOT / name).read_bytes(); hashes[name] = hashlib.sha256(content).hexdigest()
        dest = run / 'source' / name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(content)
    inherited=[]; resume_state=None
    if args.resume:
        resume_state=json.loads((args.resume.resolve()/'status.json').read_text())
        rebooted=bool(resume_state.get('boot_id') and resume_state['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        if not rebooted and (resume_state['status']!='failed' or not resume_state.get('process_cleanup_completed')):
            raise ValueError('Require a confirmed new boot or failed batch with confirmed cleanup')
        if resume_state['seeds']!=args.seeds:
            raise ValueError('Resume must preserve the original seed list and order')
        changed=[name for name,old in resume_state['source_hashes'].items() if hashes.get(name)!=old]
        if set(changed)-{'scripts/run_native_parent_collection.py'}:
            raise ValueError('Resume cannot change worker or audit code: '+repr(changed))
        for row in resume_state['episodes']:
            if json.loads(Path(row['result_file']).read_text())['status']!='complete':
                raise ValueError('Cannot inherit an incomplete episode')
            inherited.append(dict(row,inherited_from=resume_state['run_dir']))
        for row in inherited:
            if row['condition']=='prepare' and row.get('parent_saved') and not any(
                    e['seed']==row['seed'] and e['condition']=='native' for e in inherited):
                raise ValueError('This resume path requires a completed native branch for each saved inherited parent')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    state = dict(status='loading', run_dir=str(run), pid=os.getpid(), port=port, gpu=args.gpu,
                 started_utc=datetime.now(timezone.utc).isoformat(), source_hashes=hashes,
                 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), episodes=inherited,
                 seeds=args.seeds, conditions=['prepare','native'],
                 max_actions=1700, predeclared_seed_pool=list(range(410000,410020)), formal_training_eligible=False,
                 protocol='Fresh native rollouts; save first closed natural chunk boundary with four events. Retain all rejected attempts; no success filtering, interventions or training.')
    if resume_state:
        state.update(resumed_from=resume_state['run_dir'],inherited_episodes=len(inherited),
            resume_changes=changed,previous_boot_id=resume_state.get('boot_id'),
            resume_reason='Continue after a confirmed reboot' if rebooted else 'Use the simulation environment for the CPU input audit; preserve all completed policy outputs')
    def save():
        atomic_json(run / 'status.json', state); atomic_json(ROOT / 'reports/latest_native_parent_collection.json', state)
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
    def auxiliary(argv,log_path):
        nonlocal child
        with log_path.open('w') as log:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['worker_pid']=child.pid;save();deadline=time.monotonic()+600
            while child.poll() is None:
                check()
                if time.monotonic()>deadline:raise TimeoutError(str(log_path))
                time.sleep(2)
            if child.returncode:raise RuntimeError('Validation failed: '+str(log_path))
        stop_group(child);child=None
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
            existing_prepare=next((e for e in inherited if e['seed']==seed and e['condition']=='prepare'),None)
            if existing_prepare:prepared=Path(existing_prepare['result_file']).parent
            for condition in state['conditions']:
                if condition != 'prepare' and not (prepared / 'parent.json').exists():
                    state.setdefault('skipped', []).append(dict(seed=seed, condition=condition, reason='No eligible boundary')); save(); continue
                existing=next((e for e in inherited if e['seed']==seed and e['condition']==condition),None)
                if existing:
                    if condition=='prepare' and existing.get('parent_saved'):
                        validation=json.loads((prepared/'replay_validation.json').read_text())
                        if not validation.get('passed'):raise ValueError('Inherited replay validation is not passed')
                    if condition=='native':
                        state['stage']='audit_inherited_inputs';save()
                        auxiliary([str(ROOT/'envs/sim/bin/python'),'-u',str(ROOT/'scripts/audit_native_parent_inputs.py'),str(prepared)],prepared/'input_audit_resume.log')
                        existing['input_audit_passed']=json.loads((prepared/'input_audit/report.json').read_text())['status']=='verified';save()
                    continue
                output = run / f'{seed}_{condition}'
                job = dict(seed=seed, gpu=args.gpu, port=port, condition=condition, output=str(output),
                           max_actions=1700)
                if condition=='native':
                    validation=json.loads((prepared/'replay_validation.json').read_text())
                    if not validation.get('passed'):raise ValueError('Fresh parent replay not validated')
                    job.update(condition='keep',source=str(prepared),library=library,max_actions=300,fast_prefix=True)
                path = output.with_suffix('.job.json'); atomic_json(path, job)
                state.update(status='evaluating', current_job=str(path)); save()
                with output.with_suffix('.log').open('w') as log:
                    worker='native_parent_worker.py' if condition=='prepare' else 'native_recovery_vla_worker.py'
                    child = subprocess.Popen([str(ROOT / 'envs/sim/bin/python'), '-u', str(ROOT / 'scripts'/worker), str(path)],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    state['worker_pid'] = child.pid; save(); deadline = time.monotonic()+2400
                    while child.poll() is None:
                        check()
                        if time.monotonic() > deadline: raise TimeoutError(str(output))
                        heartbeat=output/'result.json'
                        if heartbeat.exists() and time.time()-heartbeat.stat().st_mtime>240:
                            raise TimeoutError('No progress for 240 seconds: '+str(output))
                        time.sleep(2)
                    if child.returncode: raise RuntimeError('Worker failed: ' + str(output))
                stop_group(child); child = None
                result = json.loads((output/'result.json').read_text())
                if result['status'] != 'complete': raise RuntimeError('Incomplete result')
                row = {k: result[k] for k in ['parent_saved','first_lift_correct','first_reveal_correct',
                    'native_first_target_opened','native_failed','actions_executed','parent_match','action_hash',
                    'final_physics_hash','forbidden_calls','rejection_reason','elapsed_seconds'] if k in result}
                row.update(seed=seed, condition=condition, result_file=str(output/'result.json'), model_requests=len(result['request_audits']))
                state['episodes'].append(row); save()
                if condition=='prepare' and result.get('parent_saved'):
                    state['stage']='validate_fresh_prefix';save()
                    auxiliary([str(ROOT/'envs/sim/bin/python'),'-u',str(ROOT/'scripts/validate_fresh_parent_replay.py'),
                               '--source',str(prepared),'--gpu',str(args.gpu)],prepared/'replay_validation.log')
                    row['replay_passed']=json.loads((prepared/'replay_validation.json').read_text()).get('passed',False);save()
                if condition=='native':
                    state['stage']='audit_fresh_inputs';save()
                    auxiliary([str(ROOT/'envs/sim/bin/python'),'-u',str(ROOT/'scripts/audit_native_parent_inputs.py'),str(prepared)],prepared/'input_audit.log')
                    row['input_audit_passed']=json.loads((prepared/'input_audit/report.json').read_text())['status']=='verified';save()
        state.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
        state.pop('current_job', None); state.pop('worker_pid', None); save()
        state.pop('stage',None);save()
    except BaseException:
        state.update(status='failed', error=traceback.format_exc()); save(); raise
    finally:
        stop_group(child); stop_group(server); resources.close()
        state['process_cleanup_completed'] = True; save()


if __name__ == '__main__': main()
