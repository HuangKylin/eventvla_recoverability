"""Optional wall-clock cooling for this exact census only; no numerical changes.

Pauses owned worker/server process groups below the runner's unchanged 75 C
abort threshold. The runner/telemetry stay live. Always resumes owned groups on
exit, and exits when the target batch finishes. Never touches unrelated jobs.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args=parser.parse_args()
    run=args.run.resolve()
    if not run.is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('Run outside project')
    state=json.loads((run/'status.json').read_text())
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if boot!=state['boot_id']:
        raise ValueError('Cannot pace an old boot')
    paused={}
    log=(run/'cooling.jsonl').open('a', buffering=1)
    def record(**data):
        log.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),**data))+'\n')
        log.flush();os.fsync(log.fileno())
    def owned(pid, script):
        try:
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            return str(ROOT/script) in cmd and os.getpgid(pid)==pid
        except (FileNotFoundError,ProcessLookupError):
            return False
    def resume():
        for pid,script in list(paused.items()):
            if owned(pid,script):os.killpg(pid,signal.SIGCONT)
            record(action='resume',pid=pid)
            paused.pop(pid,None)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    record(action='start',pid=os.getpid(),pause_at_C=68,resume_at_C=60,max_pause_seconds=8,
           numerical_configuration_changed=False,runner_abort_threshold_C=75)
    began=None
    try:
        while True:
            state=json.loads((run/'status.json').read_text())
            if state['status'] in ('complete','failed') or not Path(f"/proc/{state['pid']}").exists():
                break
            gpu=subprocess.run(['nvidia-smi',f"--id={state['gpu']}",
                '--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],
                text=True,capture_output=True,check=True,timeout=6)
            temp=float(gpu.stdout.strip())
            if paused and (temp<=60 or time.monotonic()-began>=8):
                record(action='cooling_end',temperature_C=temp,paused_seconds=time.monotonic()-began)
                resume();time.sleep(5)
            elif not paused and temp>=68 and state['status']=='evaluating':
                for key,script in [('worker_pid','scripts/evidence_diagnostic_worker.py'),
                                   ('server_pid','scripts/seeded_policy_server.py')]:
                    pid=state.get(key)
                    if pid and owned(pid,script):
                        os.killpg(pid,signal.SIGSTOP);paused[pid]=script
                        record(action='pause',pid=pid,temperature_C=temp)
                began=time.monotonic()
            time.sleep(2)
    finally:
        resume();record(action='exit');log.close()


if __name__=='__main__':main()
