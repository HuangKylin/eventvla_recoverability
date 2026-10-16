"""Two serial controller gates; no policy server, training, or formal labels."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from resume_native_controls import atomic_json,resource_sample
from run_baseline import stop_group

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--seeds',nargs='+',type=int,default=[100000,100001]);args=p.parse_args()
    lock=(ROOT/'runs/native_controls_resume.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sample=resource_sample(args.gpu);gpu=next(g for g in sample['gpus'] if int(g['index'])==args.gpu)
    if sample['alarms'] or float(gpu['memory_MiB'])>2048 or float(gpu['utilization'])>10:raise RuntimeError('GPU unavailable')
    previous=json.loads((ROOT/'reports/latest_native_boundary_controls.json').read_text())
    sources={e['seed']:str(Path(e['result_file']).parent) for e in previous['episodes'] if e['condition']=='prepare'}
    library=json.loads((ROOT/'reports/latest_shared_skill_library.json').read_text())
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_recovery_gate');run.mkdir()
    names=['scripts/native_recovery_worker.py','scripts/run_native_recovery_gate.py','src/recoverability/native_recovery.py',
           'src/recoverability/fixed_skills.py','src/recoverability/sim_runtime.py','src/recoverability/snapshot.py',
           'src/recoverability/cue_reader.py','src/recoverability/camera_capture.py',
           'third_party/EventVLA/RoboTwin-Mem/envs/cover_blocks_hard.py']
    hashes={}
    for name in names:
        data=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(data).hexdigest()
        dest=run/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    state=dict(status='running',run_dir=str(run),pid=os.getpid(),gpu=args.gpu,seeds=args.seeds,episodes=[],
               source_hashes=hashes,started_utc=datetime.now(timezone.utc).isoformat(),formal_training_eligible=False)
    def save():atomic_json(run/'status.json',state);atomic_json(ROOT/'reports/latest_native_recovery_gate.json',state)
    def interrupt(signum,frame):raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),PYTHONUNBUFFERED='1')
    child=None;alarms=0;last=0
    save();print(run,flush=True)
    try:
        with (run/'resources.jsonl').open('a',buffering=1) as monitor:
            for seed in args.seeds:
                output=run/str(seed)
                job=dict(seed=seed,gpu=args.gpu,source=sources[seed],library=library['run_dir']+'/library.json',output=str(output))
                path=run/f'{seed}.job.json';atomic_json(path,job);state['current_job']=str(path);save()
                with (run/f'{seed}.log').open('w') as log:
                    child=subprocess.Popen([str(ROOT/'envs/sim/bin/python'),'-u',str(ROOT/'scripts/native_recovery_worker.py'),str(path)],
                          cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['worker_pid']=child.pid;save();deadline=time.monotonic()+1200
                while child.poll() is None:
                    if time.monotonic()>deadline:raise TimeoutError('Controller gate timeout')
                    if time.monotonic()-last>=10:
                        sample=resource_sample(args.gpu);monitor.write(json.dumps(sample)+'\n');last=time.monotonic()
                        alarms=alarms+1 if sample['alarms'] else 0
                        if alarms>=3:raise RuntimeError('Resource guard: '+repr(sample['alarms']))
                    time.sleep(2)
                if child.returncode:raise RuntimeError('Worker failed: '+str(output))
                stop_group(child);child=None
                result=json.loads((output/'result.json').read_text())
                state['episodes'].append(dict(seed=seed,result_file=str(output/'result.json'),gate_passed=result['gate_passed'],
                    motion_gate_passed=result['motion_gate_passed'],all_evidence_recovered=result['all_evidence_recovered'],
                    recovered_slots=result['recovered_slots'],unrecovered_slots=result['unrecovered_slots'],
                    recovery_physics_steps=result['recovery_physics_steps'],reads_correct=result['reads_correct']))
                save()
            state.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat());state.pop('current_job',None);save()
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());save();raise
    finally:
        stop_group(child);state['process_cleanup_completed']=True;state.pop('worker_pid',None);save()


if __name__=='__main__':main()
