"""Run the predeclared slot/RNG calibration after the current parent collection."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from resume_native_controls import atomic_json,resource_sample
from run_baseline import stop_group
from conservative_execution import load_profile,conservative_sample,rest_between_jobs

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from recoverability.slot_protocol import CONDITIONS,PROTOCOL


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--parents',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    args=parser.parse_args()
    execution_profile=load_profile(ROOT)
    for path in (args.parents,args.after):
        if not path.resolve().is_relative_to(ROOT):raise ValueError('Manifest must be inside this project')
    old=json.loads(args.parents.read_text())
    if old['status']!='complete' or not old.get('process_cleanup_completed'):raise ValueError('Need completed calibration parents')
    seeds=[410000,410001]
    sources={e['seed']:Path(e['result_file']).parent for e in old['episodes'] if e['condition']=='prepare' and e.get('parent_saved')}
    references={e['seed']:json.loads(Path(e['result_file']).read_text()) for e in old['episodes'] if e['condition']=='native'}
    if not set(seeds)<=set(sources) or not set(seeds)<=set(references):raise ValueError('Missing predeclared parent')
    for seed in seeds:
        if json.loads((sources[seed]/'input_audit/report.json').read_text())['status']!='verified':raise ValueError('Missing input audit')
        if not json.loads((sources[seed]/'replay_validation.json').read_text())['passed']:raise ValueError('Missing prefix replay validation')
    queue_lock=(ROOT/'runs/native_slot_probe_queue.lock').open('a+')
    fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_slot_probe')
    run.mkdir()
    names=['scripts/run_native_slot_probe.py','scripts/native_slot_probe_worker.py','src/recoverability/slot_protocol.py',
           'scripts/conservative_execution.py','configs/conservative_execution.json',
           'scripts/seeded_policy_server.py','scripts/resume_native_controls.py','scripts/run_baseline.py',
           'scripts/native_boundary_worker.py','src/recoverability/sim_runtime.py','src/recoverability/camera_capture.py',
           'src/recoverability/fixed_skills.py','src/recoverability/cue_reader.py','src/recoverability/native_boundary.py',
           'src/recoverability/native_items.py','src/recoverability/native_memory.py','src/recoverability/native_recovery.py',
           'src/recoverability/snapshot.py','src/recoverability/vla_intervention.py','docs/native_slot_calibration_protocol.md',
           'scripts/audit_slot_inputs.py','scripts/analyze_slot_probe.py',
           'third_party/EventVLA/EventVLA/examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py',
           'third_party/EventVLA/RoboTwin-Mem/envs/cover_blocks_hard.py']
    hashes={}
    for name in names:
        content=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(content).hexdigest()
        dest=run/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
    state=dict(status='waiting_for_parent_collection',pid=os.getpid(),run_dir=str(run),gpu=args.gpu,
               started_utc=datetime.now(timezone.utc).isoformat(),source_hashes=hashes,
               execution_profile=execution_profile,
               boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),dependency=str(args.after.resolve()),
               parent_manifest=str(args.parents.resolve()),parent_sources={str(s):str(sources[s]) for s in seeds},
               seeds=seeds,conditions=CONDITIONS,protocol=PROTOCOL,episodes=[],formal_training_eligible=False)
    def save():
        atomic_json(run/'status.json',state);atomic_json(ROOT/'reports/latest_native_slot_probe.json',state)
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    server=child=None;resources=None;project_lock=None
    save();print(run,flush=True)
    try:
        deadline=time.monotonic()+14400
        while True:
            dependency=json.loads(args.after.read_text())
            if dependency['status']=='paused_by_user':raise RuntimeError('Predecessor paused by user; no automatic continuation')
            if dependency['status']!='complete' and dependency.get('boot_id')!=state['boot_id']:
                raise RuntimeError('Predecessor belongs to a different boot; do not wait on stale running status')
            if dependency['status']=='failed':raise RuntimeError('Parent collection failed; do not launch follow-on GPU work')
            if dependency['status']=='complete' and dependency.get('process_cleanup_completed'):break
            if time.monotonic()>deadline:raise TimeoutError('Parent collection did not finish within four hours')
            time.sleep(15)
        for name,sha in hashes.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Queued source changed: '+name)
        project_lock=(ROOT/'runs/native_controls_resume.lock').open('a+')
        # The predecessor saves cleanup before exiting its final context. Allow only that short exit race.
        lock_deadline=time.monotonic()+30
        while True:
            try:fcntl.flock(project_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>lock_deadline:raise RuntimeError('Another project experiment owns the GPU lock')
                time.sleep(1)
        initial=conservative_sample(resource_sample(args.gpu),args.gpu,execution_profile);gpu=next(g for g in initial['gpus'] if int(g['index'])==args.gpu)
        if initial['alarms'] or float(gpu['memory_MiB'])>2048 or float(gpu['utilization'])>10:
            raise RuntimeError('GPU unavailable after predecessor cleanup: '+json.dumps(initial))
        library=json.loads((ROOT/'reports/latest_shared_skill_library.json').read_text())['run_dir']+'/library.json'
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                   CUDA_HOME='/usr/local/cuda-12.8',TORCH_EXTENSIONS_DIR=str(ROOT/'cache/torch_extensions'),
                   XDG_CACHE_HOME=str(ROOT/'cache/xdg'),PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1')
        resources=(run/'resources.jsonl').open('a',buffering=1);last_sample=0.;alarms=0
        def check():
            nonlocal last_sample,alarms
            if time.monotonic()-last_sample>=execution_profile['monitor_interval_seconds']:
                sample=conservative_sample(resource_sample(args.gpu),args.gpu,execution_profile);resources.write(json.dumps(sample)+'\n');resources.flush();os.fsync(resources.fileno())
                alarms=alarms+1 if sample['alarms'] else 0;last_sample=time.monotonic()
                if alarms>=execution_profile['consecutive_alarm_limit']:raise RuntimeError('Resource guard: '+repr(sample['alarms']))
            if server is not None and server.poll() is not None:raise RuntimeError('Policy server exited')
        state.update(status='loading',port=port);save()
        with (run/'server.log').open('w') as log:
            server=subprocess.Popen([str(ROOT/'envs/policy/bin/python'),'-u',str(ROOT/'scripts/seeded_policy_server.py'),
                '--port',str(port),'--report',str(run/'server_ready.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['server_pid']=server.pid;save();deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            check()
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=1):break
            except OSError:time.sleep(1)
        else:raise TimeoutError('Server readiness timeout')
        for seed in seeds:
            paired_keep=None
            for condition in CONDITIONS:
                output=run/f'{seed}_{condition}'
                job=dict(seed=seed,gpu=args.gpu,port=port,condition='keep',slot_condition=condition,
                         source=str(sources[seed]),library=library,max_actions=300,fast_prefix=True,output=str(output))
                path=output.with_suffix('.job.json');atomic_json(path,job)
                if child is not None:raise RuntimeError('Simulation workers must remain serial')
                state['execution_stage']='inter_job_rest';save()
                rest_between_jobs(execution_profile['inter_job_rest_seconds'],check)
                state.pop('execution_stage',None)
                state.update(status='evaluating',current_job=str(path));save()
                with output.with_suffix('.log').open('w') as log:
                    child=subprocess.Popen([str(ROOT/'envs/sim/bin/python'),'-u',str(ROOT/'scripts/native_slot_probe_worker.py'),str(path)],
                        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state['worker_pid']=child.pid;save();deadline=time.monotonic()+2400
                    while child.poll() is None:
                        check()
                        if time.monotonic()>deadline:raise TimeoutError(str(output))
                        heartbeat=output/'result.json'
                        if heartbeat.exists() and time.time()-heartbeat.stat().st_mtime>240:raise TimeoutError('No worker progress')
                        time.sleep(2)
                    if child.returncode:raise RuntimeError('Worker failed: '+str(output))
                stop_group(child);child=None
                result=json.loads((output/'result.json').read_text())
                if result['status']!='complete':raise RuntimeError('Incomplete result')
                row={k:result[k] for k in ('first_lift_correct','first_reveal_correct','actions_executed','policy_physics_steps','action_hash','final_physics_hash','elapsed_seconds','forbidden_calls')}
                row.update(seed=seed,condition=condition,result_file=str(output/'result.json'),model_requests=len(result['request_audits']))
                if condition=='slots_paired':paired_keep=result
                if condition in ('native_legacy','slots_paired_repeat'):
                    reference=references[seed] if condition=='native_legacy' else paired_keep
                    inputs=lambda value:[(a['step'],a['paired_seed'],a['images']) for a in value['request_audits']]
                    row['reference_match']=dict(actions=result['action_hash']==reference['action_hash'],
                        physics=result['final_physics_hash']==reference['final_physics_hash'],inputs=inputs(result)==inputs(reference))
                state['episodes'].append(row);save()
                if 'reference_match' in row and not all(row['reference_match'].values()):raise RuntimeError('Native replay or paired RNG repeat did not reproduce')
        state.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat())
        state.pop('current_job',None);state.pop('worker_pid',None);save()
        audit=subprocess.run([str(ROOT/'envs/tools/bin/python'),str(ROOT/'scripts/analyze_slot_probe.py'),'--report',str(run/'status.json')],cwd=ROOT,env=env,capture_output=True,text=True)
        (run/'analysis.log').write_text(audit.stdout+audit.stderr)
        if audit.returncode:raise RuntimeError('Slot calibration analysis failed; inspect analysis.log')
        state['analysis_completed']=True;save()
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());save();raise
    finally:
        stop_group(child);stop_group(server)
        if resources is not None:resources.close()
        state['process_cleanup_completed']=True;save()


if __name__=='__main__':main()
