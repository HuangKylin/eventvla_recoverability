"""Fresh-process T3 K/R branch. No oracle geometry enters either controller."""
import argparse
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
    job = json.loads(p.parse_args().job.read_text())
    for name, sha in job['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != sha:
            raise RuntimeError('Source changed: '+name)
    from recoverability.sim_runtime import configure, make_env, close_env, public_observation
    configure(ROOT, job.get('gpu','1'))
    import numpy as np
    from PIL import Image
    from recoverability.action_latch_task import action_latch_v1, physical_latch_steps
    from recoverability.fixed_skills import SharedRecoveryController, execute_skill, deny_oracle_actions
    from recoverability.snapshot import physics_vector, digest, rng_state
    out = Path(job['output'])
    out.mkdir(parents=True)
    report = dict(status='running',job=job,scope='physical_latch_fixed_controller',
                  formal_eventvla_training_eligible=False)
    def save():
        (out/'result.json').write_text(json.dumps(report,indent=2))
    env = None
    start = time.monotonic()
    try:
        controller = SharedRecoveryController(job['library'])
        env = make_env(ROOT,'action_latch_v1',job['seed'],task_class=action_latch_v1)
        order = np.random.default_rng(job['seed']+991).permutation(4).tolist()
        images, trace, hashes = {}, [], []
        # Evaluator has the full prefix; the decision controller gets only the
        # selected memory dictionary. Labels and artifact paths never enter it.
        with deny_oracle_actions(env) as audit:
            for slot in order:
                result = execute_skill(env,controller,slot)
                if result['read'].answer != env.block_color_names[slot] or result['closed_read'].answer is not None:
                    raise RuntimeError('Prefix visibility/perception failed')
                images[slot] = result['image'].copy()
                Image.fromarray(result['image']).save(out/f'evidence_{slot}.png')
                trace.append(physics_vector(env))
                hashes.append(digest(result['image']))
                del result
            current = public_observation(env.get_obs())
            report.update(order=order,parent_hash=digest(physics_vector(env)),
                prefix_trace_hash=digest(np.asarray(trace)),prefix_image_hashes=hashes,
                current_hash=digest(current),rng_hash=digest(rng_state()),
                library_hash=controller.library_hash)
            np.savez_compressed(out/'public_parent.npz',**current['images'],
                                proprioception=current['proprioception'])
            # Eviction precedes the physical action. K and R have identical
            # recovery permissions. Other evidence remains available in R.
            dropped = job.get('drop_slot')
            memory = {s:im for s,im in images.items() if s != dropped}
            images.clear()
            report['retained_slots'] = list(memory)
            skill = controller.skills[job['action_slot']]
            report['action_hash'] = digest(skill.drives)
            # Public proposed motor sequence, without action-slot or outcome
            # labels, is the learned scorer's action input.
            sampled = skill.drives[np.linspace(0,len(skill.drives)-1,16).astype(int)]
            np.savez_compressed(out/'public_action.npz',sampled_drives=sampled)
            env.arm_latches()
            with physical_latch_steps(env):
                action_result = execute_skill(env,controller,job['action_slot'])
            # Current RGB after the action is allowed; transient old imagery
            # from executing the open-loop motor program is not cached.
            del action_result
            env.latch_armed = False
            report.update(events=env.latch_events,post_action_hash=digest(physics_vector(env)),
                          post_action_rng_hash=digest(rng_state()))
            if [e['slot'] for e in env.latch_events] != [job['action_slot']]:
                raise RuntimeError('Proposed physical action did not trigger exactly its intended latch')
            queries = range(4) if dropped is None else [dropped]
            outcomes = []
            for slot in queries:
                answer = controller.read(slot,memory[slot]).answer if slot in memory else None
                steps = 0
                if answer is None:
                    result = execute_skill(env,controller,slot)
                    answer = result['read'].answer
                    steps = result['physics_steps']
                    Image.fromarray(result['image']).save(out/f'recovery_{slot}.png')
                    if result['closed_read'].answer is not None:
                        raise RuntimeError('Recovery failed to close cover')
                    del result
                outcomes.append(dict(slot=slot,answer=answer,
                    success=bool(answer==env.block_color_names[slot]),recovery_steps=steps,
                    recovery_allowed=True))
            report.update(outcomes=outcomes,forbidden_calls=audit['forbidden_calls'])
        report.update(status='complete',elapsed_seconds=time.monotonic()-start)
        save()
    except Exception:
        report.update(status='failed',error=traceback.format_exc())
        save()
        raise
    finally:
        if env is not None:
            close_env(env)


if __name__ == '__main__':
    main()
