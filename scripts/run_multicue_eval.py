"""Paired-seed capacity experiments. Fail closed on control or prefix mismatches."""
import argparse
from datetime import datetime, timezone
import hashlib
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[200000,200001,200002])
    p.add_argument('--gpu', default='1')
    p.add_argument('--capacity-sweep', action='store_true')
    args = p.parse_args()
    args.library = args.library.resolve()
    library = json.loads(args.library.read_text())
    if library['status'] != 'complete' or library['calibration_seed'] in args.seeds:
        raise ValueError('Require a complete library and held-out evaluation seeds')
    names = ['multi_cue_task.py','fixed_skills.py','versioned_memory.py','cue_reader.py',
             'cue_probe_task.py','snapshot.py','sim_runtime.py']
    source_files = ['src/recoverability/'+name for name in names] + [
                    'scripts/multicue_worker.py','scripts/run_multicue_eval.py']
    hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files}
    run = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_multicue_eval')
    run.mkdir(parents=True)
    settings = [(m,4) for m in ('action_aware','fifo','oldest','uniform','archive','structured')]
    if args.capacity_sweep:
        settings += [(m,b) for b in (2,8) for m in ('fifo','action_aware')]
    status = dict(status='running',run_dir=str(run),seeds=args.seeds,library=str(args.library),
                  source_hashes=hashes,planned_episodes=len(settings)*len(args.seeds),episodes=[],
                  experiment_scope='fixed_workstation_versioned_colour_pilot',training_eligible=False)
    def save():
        content = json.dumps(status,indent=2)
        (run/'status.json').write_text(content)
        (ROOT/'reports/latest_multicue_eval.json').write_text(content)
    save()
    started = time.monotonic()
    try:
        for seed in args.seeds:
            reference = None
            for method,capacity in settings:
                output = run/f'seed_{seed}_{method}_b{capacity}'
                job = dict(seed=seed,method=method,capacity=capacity,gpu=args.gpu,
                           library=str(args.library),output=str(output),source_hashes=hashes)
                path = output.with_suffix('.job.json')
                path.write_text(json.dumps(job,indent=2))
                print(json.dumps({'stage':'episode','seed':seed,'method':method,'capacity':capacity}),flush=True)
                with output.with_suffix('.log').open('w') as log:
                    child = subprocess.Popen([str(ROOT/'envs/sim/bin/python'),'-u',
                              str(ROOT/'scripts/multicue_worker.py'),str(path)],cwd=ROOT,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        code = child.wait(timeout=300)
                        if code:
                            raise RuntimeError(f'Worker exited {code}: {output.with_suffix(".log")}')
                    finally:
                        try:
                            os.killpg(child.pid,signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                result = json.loads((output/'result.json').read_text())
                if result['status']!='complete' or result['perception_correct']!=8 or result['forbidden_calls']:
                    raise RuntimeError('Controller/observation gate failed; do not interpret this as memory failure')
                identity = {key:result[key] for key in ('parent_hash','prefix_image_hashes','prefix_trace_hash',
                                                       'rng_hash','library_hash')}
                if reference is None:
                    reference = identity
                if identity != reference:
                    raise RuntimeError(f'Methods did not share an identical prefix for seed {seed}')
                if method not in ('archive','structured'):
                    if any(row['memory']['image_count'] > capacity for row in result['observations']):
                        raise RuntimeError('Persistent image capacity exceeded')
                if any(row['history_images_read']>1 for row in result['queries']):
                    raise RuntimeError('Historical image read budget exceeded')
                row = dict(seed=seed,method=method,capacity=capacity,result_file=str(output/'result.json'),
                    correct_queries=result['correct_queries'],query_count=result['query_count'],
                    episode_success=result['episode_success'],recovery_calls=result['recovery_calls'],
                    recovery_physics_steps=result['recovery_physics_steps'],storage=result['pre_query_storage'],
                    same_prefix_validated=True,forbidden_calls=result['forbidden_calls'],
                    elapsed_seconds=result['elapsed_seconds'])
                status['episodes'].append(row)
                save()
                print(json.dumps(row),flush=True)
        aggregates=[]
        for method,capacity in settings:
            rows=[r for r in status['episodes'] if r['method']==method and r['capacity']==capacity]
            aggregates.append(dict(method=method,capacity=capacity,episodes=len(rows),
                successful_episodes=sum(r['episode_success'] for r in rows),
                correct_queries=sum(r['correct_queries'] for r in rows),
                queries=sum(r['query_count'] for r in rows),
                recovery_calls=sum(r['recovery_calls'] for r in rows),
                recovery_physics_steps=sum(r['recovery_physics_steps'] for r in rows),
                max_image_payload_bytes=max(r['storage']['image_payload_bytes'] for r in rows),
                max_metadata_payload_bytes=max(r['storage']['metadata_payload_bytes'] for r in rows)))
        status.update(status='complete',aggregates=aggregates,elapsed_seconds=time.monotonic()-started,
                      all_prefixes_matched=True)
        save()
        data=ROOT/'data/pilots'/run.name
        data.mkdir(parents=True)
        (data/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in status['episodes']))
        (data/'README.txt').write_text('Fixed-workstation diagnostic pilot, no learned scorer or EventVLA policy.\n'
                                       'Do not reuse these evaluation seeds as training data.\n')
    except Exception:
        status.update(status='failed',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        save()
        raise


if __name__ == '__main__':
    main()
