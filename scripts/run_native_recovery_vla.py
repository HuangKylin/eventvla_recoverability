"""Guarded serial native-memory diagnostics with isolated protocol manifests."""
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
    parser.add_argument('--previous', type=Path, default=ROOT/'reports/latest_native_boundary_controls.json')
    parser.add_argument('--seeds', type=int, nargs='+', default=[100000, 100001])
    parser.add_argument('--original-memory-probe',action='store_true',help='Separate exploratory control: execute recovery while retaining all original events')
    parser.add_argument('--continuation-probe',action='store_true',help='Separate matched time-only and passive-hold controls retaining original events')
    parser.add_argument('--item-pilot',action='store_true',help='Development-only keep / delete one / recover one labels')
    parser.add_argument('--content-probe',action='store_true',help='Constant-count candidate image substitution diagnostic')
    parser.add_argument('--candidate-indices',type=int,nargs='+',default=[0,1,2,3])
    args = parser.parse_args()
    if sum([args.original_memory_probe,args.continuation_probe,args.item_pilot,args.content_probe])>1:raise ValueError('Choose one diagnostic protocol')
    if not args.candidate_indices or len(set(args.candidate_indices))!=len(args.candidate_indices) or not set(args.candidate_indices)<=set(range(4)):
        raise ValueError('Candidate indices must be unique and within 0..3')
    lock = (ROOT / 'runs/native_controls_resume.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial = resource_sample(args.gpu)
    gpu = next(g for g in initial['gpus'] if int(g['index']) == args.gpu)
    if initial['alarms'] or float(gpu['memory_MiB']) > 2048 or float(gpu['utilization']) > 10:
        raise RuntimeError('GPU or system is not available: ' + json.dumps(initial))
    old = json.loads(args.previous.read_text())
    sources = {e['seed']: str(Path(e['result_file']).parent) for e in old['episodes'] if e['condition'] == 'prepare'}
    valid_seeds=(bool(args.seeds) and set(args.seeds)<=set(sources)) if args.original_memory_probe or args.continuation_probe or args.item_pilot or args.content_probe else (len(args.seeds)==2 and set(args.seeds)==set(sources))
    if old['status'] != 'complete' or not valid_seeds:
        raise ValueError('Need the two completed, validated safe parents')
    gate=json.loads((ROOT/'reports/latest_native_recovery_gate.json').read_text())
    if gate['status']!='complete' or not set(args.seeds)<={e['seed'] for e in gate['episodes'] if e.get('motion_gate_passed')}:
        raise ValueError('Both recovery motion/closure gates must pass first; coverage may be partial and is reported separately')
    for name in ['src/recoverability/native_recovery.py','src/recoverability/fixed_skills.py',
                 'src/recoverability/camera_capture.py','src/recoverability/sim_runtime.py']:
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=gate['source_hashes'][name]:
            raise ValueError('Recovery implementation changed after gate: '+name)
    library=json.loads((ROOT/'reports/latest_shared_skill_library.json').read_text())['run_dir']+'/library.json'
    fast_path=ROOT/'reports/fast_prefix_validation.json'
    fast=json.loads(fast_path.read_text()) if fast_path.exists() else {}
    fast_prefix=(fast.get('status')=='complete' and
                 set(args.seeds)<={e['seed'] for e in fast.get('episodes',[]) if e.get('passed')} and
                 fast.get('source_hash')==hashlib.sha256((ROOT/'scripts/validate_fast_prefix.py').read_bytes()).hexdigest())
    baseline=json.loads((ROOT/'reports/latest_native_memory_controls.json').read_text())
    keep_reference={e['seed']:json.loads(Path(e['result_file']).read_text()) for e in baseline['episodes'] if e['condition']=='keep'}
    for source in sources.values():
        if not (Path(source) / 'input_audit/report.json').exists():
            raise ValueError('Missing validated parent input audit')
    matched_ticks={}
    if args.continuation_probe:
        reference=json.loads((ROOT/'reports/latest_native_recovery_vla.json').read_text())
        if reference['status']!='complete':raise ValueError('Need completed recovery duration reference')
        matched_ticks={e['seed']:int(e['recovery_physics_steps']) for e in reference['episodes'] if e['condition']=='recover'}
        if not set(args.seeds)<=set(matched_ticks):raise ValueError('Missing matched recovery durations')
    stem='native_recovery_continuation_probe' if args.continuation_probe else ('native_recovery_original_probe' if args.original_memory_probe else 'native_recovery_vla')
    if args.item_pilot:stem='native_recovery_item_pilot'
    if args.content_probe:stem='native_recovery_content_probe'
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_'+stem)
    run.mkdir()
    names = ['scripts/native_recovery_vla_worker.py','scripts/run_native_recovery_vla.py',
             'src/recoverability/native_recovery.py','src/recoverability/fixed_skills.py',
             'src/recoverability/camera_capture.py','src/recoverability/snapshot.py',
             'src/recoverability/native_boundary.py','src/recoverability/native_memory.py','src/recoverability/native_items.py',
             'src/recoverability/cue_reader.py','scripts/seeded_policy_server.py','docs/native_recovery_protocol.md',
             'scripts/analyze_native_recovery.py','scripts/validate_fast_prefix.py',
             'scripts/resume_native_controls.py','scripts/native_boundary_worker.py',
             'src/recoverability/sim_runtime.py','src/recoverability/vla_intervention.py',
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
                 seeds=args.seeds, conditions=['time_only_original','hold_original'] if args.continuation_probe else (['recover_original'] if args.original_memory_probe else ['keep', 'recover', 'recover_discard']), parent_sources=sources, recovery_gate=gate['run_dir'],
                 matched_recovery_ticks=matched_ticks,
                 max_actions=300, formal_training_eligible=False,
                 fast_prefix=fast_prefix,fast_prefix_validation=fast if fast_prefix else None,
                 protocol='Public recovery then irreversible commit. Same physical sweep in recover/recover_discard. Bundle diagnostic only; no per-item labels.')
    if args.item_pilot:
        state.update(conditions=['keep']+[f'{condition}_item_{i}' for i in args.candidate_indices for condition in ['drop','recover']],
            candidate_indices=args.candidate_indices,
            protocol='Single-candidate K/D/R development pilot. Reuse one equivalent keep per parent. Public routing, one-slot recovery, frozen first-reveal endpoint. Not formal training data.')
    if args.content_probe:
        state.update(conditions=['keep']+[f'mask_item_{i}' for i in args.candidate_indices],candidate_indices=args.candidate_indices,
            protocol='Diagnostic only: keep four event positions/timestamps/metadata; replace one image with current cue-free head RGB. No physics/time intervention; no training labels.')
    def save():
        atomic_json(run / 'status.json', state); atomic_json(ROOT / ('reports/latest_'+stem+'.json'), state)
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
            for condition in state['conditions']:
                output = run / f'{seed}_{condition}'
                job = dict(seed=seed, gpu=args.gpu, port=port, condition=condition, output=str(output),
                           source=sources[seed], library=library, max_actions=300,fast_prefix=fast_prefix)
                if args.continuation_probe:job['matched_recovery_ticks']=matched_ticks[seed]
                path = output.with_suffix('.job.json'); atomic_json(path, job)
                state.update(status='evaluating', current_job=str(path)); save()
                with output.with_suffix('.log').open('w') as log:
                    child = subprocess.Popen([str(ROOT / 'envs/sim/bin/python'), '-u', str(ROOT / 'scripts/native_recovery_vla_worker.py'), str(path)],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    state['worker_pid'] = child.pid; save(); deadline = time.monotonic()+2400
                    while child.poll() is None:
                        check()
                        if time.monotonic() > deadline: raise TimeoutError(str(output))
                        heartbeat=output/'result.json'
                        if heartbeat.exists() and time.time()-heartbeat.stat().st_mtime>240:
                            raise TimeoutError('No experiment progress for 240 seconds; inspect faulthandler log: '+str(output))
                        time.sleep(2)
                    if child.returncode: raise RuntimeError('Worker failed: ' + str(output))
                stop_group(child); child = None
                result = json.loads((output/'result.json').read_text())
                if result['status'] != 'complete': raise RuntimeError('Incomplete result')
                row = {k: result[k] for k in ['parent_saved','first_lift_correct','first_reveal_correct',
                    'recovery_physics_steps','commit_step','native_first_target_opened','native_failed','actions_executed','parent_match','action_hash',
                    'final_physics_hash','forbidden_calls'] if k in result}
                row.update(seed=seed, condition=condition, result_file=str(output/'result.json'), model_requests=len(result['request_audits']))
                if condition=='keep':
                    ref=keep_reference[seed]
                    same_inputs=lambda x:[(a['step'],a['images']) for a in x['request_audits']]
                    row['reference_match']=dict(actions=result['action_hash']==ref['action_hash'],
                        physics=result['final_physics_hash']==ref['final_physics_hash'],inputs=same_inputs(result)==same_inputs(ref))
                    if not all(row['reference_match'].values()):
                        state['episodes'].append(row);save();raise RuntimeError('Keep does not reproduce the validated native baseline')
                state['episodes'].append(row); save()
        state.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
        state.pop('current_job', None); state.pop('worker_pid', None); save()
    except BaseException:
        state.update(status='failed', error=traceback.format_exc()); save(); raise
    finally:
        stop_group(child); stop_group(server); resources.close()
        state['process_cleanup_completed'] = True; save()


if __name__ == '__main__': main()
