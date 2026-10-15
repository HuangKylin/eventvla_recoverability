"""One fresh-process capacity experiment, with no runtime expert planning."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('job', type=Path)
    args = p.parse_args()
    job = json.loads(args.job.read_text())
    for name, checksum in job.get('source_hashes', {}).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != checksum:
            raise RuntimeError('Evaluation source changed: ' + name)
    from recoverability.sim_runtime import configure, make_env, close_env, public_observation
    configure(ROOT, job.get('gpu','1'))
    import numpy as np
    from PIL import Image
    from recoverability.multi_cue_task import multi_cue_versions_v1
    from recoverability.fixed_skills import SharedRecoveryController, execute_skill, deny_oracle_actions
    from recoverability.versioned_memory import VersionedMemory
    from recoverability.snapshot import physics_vector, digest, rng_state
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=False)
    controller = SharedRecoveryController(job['library'])
    if job['seed'] == controller.manifest['calibration_seed']:
        raise ValueError('Evaluation seed must differ from calibration')
    memory = VersionedMemory(job['method'], job['capacity'])
    report = dict(status='running',job=job,library_hash=controller.library_hash,
                  runtime_expert_access=False,training_eligible=False,
                  shared_motor_parameter_bytes=sum(s.drives.nbytes+s.start_proprioception.nbytes
                                                   for s in controller.skills.values()),
                  observations=[],queries=[],control_failures=[],initialization_seconds=None)
    def save():
        (output/'result.json').write_text(json.dumps(report,indent=2))
    save()
    env = None
    started = time.monotonic()
    private_trace, image_hashes = [], []
    try:
        env = make_env(ROOT,'multi_cue_versions_v1',job['seed'],task_class=multi_cue_versions_v1)
        report['initialization_seconds'] = time.monotonic()-started
        with deny_oracle_actions(env) as audit:
            for version in range(2):
                if version:
                    # All methods receive the same public proposed action batch.
                    # This occurs before version-one images become available.
                    memory.proposed_advance([0,1,2,3])
                    for slot in range(4):
                        env.advance_indicator(slot)
                for slot in range(4):
                    print(json.dumps({'stage':'observe','slot':slot,'version':version}),flush=True)
                    result = execute_skill(env,controller,slot)
                    if result['read'].answer is None or result['closed_read'].answer is not None:
                        raise RuntimeError(f'Visibility/closure gate failed at {(slot,version)}: '
                                           f"{result['read']}; {result['closed_read']}")
                    # Evaluator checks perception after controller output, never supplies the answer.
                    correct = bool(env.score_answer(slot,version,result['read'].answer))
                    memory_audit = memory.observe(slot,version,result['image'])
                    report['observations'].append(dict(slot=slot,version=version,read_correct=correct,
                        physics_steps=result['physics_steps'],seconds=result['seconds'],
                        start_error_rad=result['start_error_rad'],memory=memory_audit))
                    private_trace.append(physics_vector(env))
                    image_hashes.append(digest(result['image']))
                    Image.fromarray(result['image']).save(output/f'event_{version}_{slot}.png')
                    # Full frames and answers are evaluator artifacts, inaccessible to memory/controller.
                    del result
                    save()
            report.update(parent_hash=digest(private_trace[-1]),prefix_image_hashes=image_hashes,
                          prefix_trace_hash=digest(np.asarray(private_trace)),rng_hash=digest(rng_state()),
                          pre_query_storage=memory.storage(),pre_query_keys=memory.keys())
            np.savez_compressed(output/'evaluator_prefix.npz',trace=np.asarray(private_trace))
            # Public, fixed interleaved query order; the memory policy did not see a future query.
            query_order=[(slot,version) for slot in (2,0,3,1) for version in (0,1)]
            for slot,version in query_order:
                begin = time.monotonic()
                answer, history_reads = memory.answer(slot,version)
                source = 'history' if answer is not None else 'missing'
                recovery_steps = 0
                recovery_seconds = 0.
                if answer is None and memory.recovery_possible_from_public_state(slot,version):
                    print(json.dumps({'stage':'recover','slot':slot,'version':version}),flush=True)
                    result = execute_skill(env,controller,slot)
                    answer = result['read'].answer
                    if result['closed_read'].answer is not None:
                        raise RuntimeError('Recovery did not close its cover')
                    source = 'recovery'
                    recovery_steps = result['physics_steps']
                    recovery_seconds = result['seconds']
                    Image.fromarray(result['image']).save(output/f'recovery_{version}_{slot}.png')
                    del result
                correct = bool(env.score_answer(slot,version,answer))
                report['queries'].append(dict(slot=slot,version=version,answer=answer,correct=correct,
                    source=source,history_images_read=history_reads,recovery_physics_steps=recovery_steps,
                    recovery_seconds=recovery_seconds,query_seconds=time.monotonic()-begin))
                # Queries are unique; the observed recovery frame is transient and released.
                # The same fixed trigger and update rule is used by every method.
                save()
            report['forbidden_calls'] = audit['forbidden_calls']
        report.update(status='complete',elapsed_seconds=time.monotonic()-started,
                      correct_queries=sum(q['correct'] for q in report['queries']),
                      query_count=len(report['queries']),
                      episode_success=all(q['correct'] for q in report['queries']),
                      perception_correct=sum(q['read_correct'] for q in report['observations']),
                      recovery_calls=sum(q['source']=='recovery' for q in report['queries']),
                      recovery_physics_steps=sum(q['recovery_physics_steps'] for q in report['queries']))
        import resource
        report['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report['storage_scope'] = 'Image and JSON metadata payloads; RSS includes simulator/runtime; motor library is shared.'
        save()
        print(json.dumps({k:report[k] for k in ('status','correct_queries','query_count','recovery_calls','forbidden_calls')}),flush=True)
    except Exception:
        report.update(status='failed',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        save()
        raise
    finally:
        if env is not None:
            close_env(env)


if __name__ == '__main__':
    main()
