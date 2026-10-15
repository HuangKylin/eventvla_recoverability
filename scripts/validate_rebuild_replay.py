"""Fallback gate: rebuild each branch in a fresh process and replay an identical prefix."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def run_job(job_path, log_path, timeout=300):
    with log_path.open('w') as log:
        p = subprocess.Popen([str(ROOT/'envs/sim/bin/python'), '-u',
                              str(ROOT/'scripts/replay_worker.py'), str(job_path)],
                             cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
            if rc:
                raise RuntimeError(f'Replay worker exited {rc}; see {log_path}')
        finally:
            # Kill only this worker's process group, including its own planner children.
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def main():
    import numpy as np
    from recoverability.snapshot import max_error
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--restore-tolerance', type=float, default=1e-5)
    parser.add_argument('--rollout-tolerance', type=float, default=1e-3)
    args = parser.parse_args()
    args.source_run = args.source_run.resolve()
    if args.repeats < 2:
        raise ValueError('At least two independent rebuilds are required')
    source = json.loads((args.source_run/'status.json').read_text())
    run = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_rebuild_validation')
    run.mkdir(parents=True)
    status = dict(status='running', run_dir=str(run), source_run=str(args.source_run),
                  contract='fresh_process_prefix_replay', config={k:str(v) if isinstance(v,Path) else v
                  for k,v in vars(args).items()}, episodes=[])
    def save():
        content = json.dumps(status, indent=2)
        (run/'status.json').write_text(content)
        (ROOT/'reports/latest_rebuild_validation.json').write_text(content)
    save()
    start = time.monotonic()
    try:
        for episode in source['episodes']:
            seed = episode['seed']
            tape_path = args.source_run/f'tape_{seed}.npz'
            tape_length = len(np.load(tape_path, allow_pickle=False)['drives'])
            results, metadata = [], []
            for repeat in range(args.repeats):
                output = run/f'replay_{seed}_{repeat}.npz'
                job = dict(task='cover_blocks_hard', seed=seed, gpu=args.gpu,
                           tape=str(tape_path), prefix_steps=tape_length//2, output=str(output))
                job_path = output.with_suffix('.job.json')
                job_path.write_text(json.dumps(job, indent=2))
                print(json.dumps({'stage':'fresh_replay','seed':seed,'repeat':repeat}), flush=True)
                run_job(job_path, output.with_suffix('.log'))
                with np.load(output, allow_pickle=False) as loaded:
                    results.append({k:loaded[k] for k in loaded.files})
                metadata.append(json.loads(output.with_suffix('.json').read_text()))
            parent_error = max(max_error(results[0]['parent'], r['parent']) for r in results[1:])
            continuation_error = max(max_error(results[0]['trace'], r['trace']) for r in results[1:])
            image_error = max(max_error(results[0]['parent_head'].astype(float),
                                       r['parent_head'].astype(float)) for r in results[1:])
            rng_exact = len({r['rng_hash'] for r in metadata}) == 1
            result = dict(seed=seed, prefix_steps=tape_length//2, physics_steps=tape_length,
                          repeats=args.repeats, parent_error=parent_error,
                          max_continuation_error=continuation_error,
                          image_max_absolute_error=image_error, rng_exact=rng_exact,
                          accepted=parent_error<=args.restore_tolerance and
                          continuation_error<=args.rollout_tolerance and rng_exact)
            status['episodes'].append(result)
            save()
            print(json.dumps(result), flush=True)
        status.update(status='complete', all_accepted=bool(status['episodes']) and
                      all(r['accepted'] for r in status['episodes']), elapsed_seconds=time.monotonic()-start)
        save()
        if not status['all_accepted']:
            raise RuntimeError('Fresh-process prefix replay failed the continuation gate')
    except Exception:
        status.update(status='failed',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
        save()
        raise


if __name__ == '__main__':
    main()
