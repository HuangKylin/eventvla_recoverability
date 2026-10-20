"""Audit each newly completed native census result and exit with the batch.

This local job companion neither creates new experiments nor starts branches
or training. A failed actual-input audit stops only its own associated runner.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from resume_native_controls import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    run=a.run.resolve()
    if not run.is_relative_to((ROOT/'runs').resolve()):raise ValueError('Run outside project')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    count=-1
    progress=dict(status='watching',pid=os.getpid(),run_dir=str(run),audited_completed_episodes=0)
    def save(**updates):
        progress.update(updates,updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(run/'analysis_watcher.json',progress)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    save()
    while True:
        state=json.loads((run/'status.json').read_text())
        if state['boot_id']!=boot:raise RuntimeError('Old-boot census')
        completed=len(state['episodes'])
        if completed!=count or state['status'] in ('complete','failed'):
            with (run/'automatic_analysis.log').open('a') as log:
                result=subprocess.run([str(ROOT/'envs/sim/bin/python'),
                    str(ROOT/'scripts/analyze_evidence_census.py'),'--run',str(run),'--allow-partial'],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            if result.returncode:
                save(status='audit_failed',returncode=result.returncode)
                pid=state['pid']
                try:
                    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
                    if str(ROOT/'scripts/run_evidence_diagnostic.py') in cmd:
                        os.kill(pid,signal.SIGTERM)
                except (FileNotFoundError,ProcessLookupError):pass
                return
            count=completed
            save(audited_completed_episodes=count)
        if state['status'] in ('complete','failed'):
            save(status='complete' if state['status']=='complete' else 'batch_stopped')
            return
        if not Path(f"/proc/{state['pid']}").exists():
            save(status='runner_missing')
            return
        time.sleep(10)


if __name__=='__main__':main()
