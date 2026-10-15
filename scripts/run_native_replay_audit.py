"""Run one monitored, serial replay audit after the GPU controls have finished."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from resume_native_controls import atomic_json, resource_sample
from run_baseline import stop_group

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    lock = (ROOT / 'runs/native_controls_resume.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial = resource_sample(args.gpu)
    selected = next(g for g in initial['gpus'] if int(g['index']) == args.gpu)
    if initial['alarms'] or float(selected['memory_MiB']) > 2048 or float(selected['utilization']) > 10:
        raise RuntimeError('Audit requires an idle GPU after the control runner cleanup')
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_native_metric_audit')
    run.mkdir()
    state = dict(status='running', run_dir=str(run), pid=os.getpid(), source_branch=str(args.branch.resolve()),
                 gpu=args.gpu, started_utc=datetime.now(timezone.utc).isoformat(), source_hashes={},
                 formal_training_eligible=False)
    for name in ('scripts/audit_native_replay_worker.py', 'scripts/run_native_replay_audit.py',
                 'scripts/native_control_worker.py','scripts/resume_native_controls.py','src/recoverability/sim_runtime.py'):
        content = (ROOT / name).read_bytes()
        dest = run / 'source' / name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(content)
        state['source_hashes'][name] = hashlib.sha256(content).hexdigest()
    def save():
        atomic_json(run / 'status.json', state)
        atomic_json(ROOT / 'reports/latest_native_metric_audit.json', state)
    def interrupted(signum,frame):
        raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    job = dict(branch=str(args.branch.resolve()),output=str(run/'audit'),gpu=args.gpu)
    atomic_json(run/'job.json',job)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_HOME='/usr/local/cuda-12.8',
               TORCH_EXTENSIONS_DIR=str(ROOT/'cache/torch_extensions'),XDG_CACHE_HOME=str(ROOT/'cache/xdg'),
               PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1')
    child = None
    save()
    print(run,flush=True)
    try:
        with (run/'worker.log').open('w') as log:
            child = subprocess.Popen([str(ROOT/'envs/sim/bin/python'),'-u',str(ROOT/'scripts/audit_native_replay_worker.py'),str(run/'job.json')],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['worker_pid']=child.pid
        save()
        deadline=time.monotonic()+1800
        streak=0
        with (run/'resources.jsonl').open('a') as stream:
            while child.poll() is None:
                sample=resource_sample(args.gpu)
                stream.write(json.dumps(sample)+'\n');stream.flush();os.fsync(stream.fileno())
                streak=streak+1 if sample['alarms'] else 0
                if streak>=3: raise RuntimeError('Resource guard: '+str(sample['alarms']))
                if time.monotonic()>deadline: raise TimeoutError('Replay audit timeout')
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: pass
        if child.returncode: raise RuntimeError('Replay audit failed')
        result=json.loads((run/'audit/result.json').read_text())
        if result['status']!='complete': raise RuntimeError('Incomplete replay audit')
        state.update(status='complete',result=result,completed_utc=datetime.now(timezone.utc).isoformat())
        save()
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());save();raise
    finally:
        stop_group(child)
        state['process_cleanup_completed']=True
        save()


if __name__=='__main__':main()
