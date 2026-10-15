"""Capacity-one real simulator rollout, with a scorer decision before action."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser();p.add_argument('job',type=Path)
    job=json.loads(p.parse_args().job.read_text())
    from recoverability.sim_runtime import configure,make_env,close_env
    configure(ROOT,job.get('gpu','1'))
    import numpy as np
    from recoverability.action_latch_task import action_latch_v1,physical_latch_steps
    from recoverability.fixed_skills import SharedRecoveryController,execute_skill,deny_oracle_actions
    from recoverability.snapshot import physics_vector,digest
    from recoverability.scorer import public_features,predict_scores
    out=Path(job['output']);out.mkdir(parents=True)
    report=dict(status='running',job=job,formal_EventVLA_result=False)
    def save(): (out/'result.json').write_text(json.dumps(report,indent=2))
    env=None;start=time.monotonic()
    try:
        env=make_env(ROOT,'action_latch_v1',job['seed'],task_class=action_latch_v1)
        controller=SharedRecoveryController(job['library'])
        order=np.random.default_rng(job['seed']+991).permutation(4).tolist()
        memory={};semantics={}
        method=job['method']
        with deny_oracle_actions(env) as audit:
            for slot in order:
                result=execute_skill(env,controller,slot)
                if result['read'].answer!=env.block_color_names[slot] or result['closed_read'].answer is not None:
                    raise RuntimeError('Prefix perception/closure failed')
                memory[slot]=result['image'].copy()
                # Only the explicit unbounded semantic baseline stores answers.
                if method=='structured': semantics[slot]=result['read'].answer
                del result
            report.update(parent_hash=digest(physics_vector(env)),order=order)
            skill=controller.skills[job['action_slot']]
            action=skill.drives[np.linspace(0,len(skill.drives)-1,16).astype(int)]
            report['action_hash']=digest(skill.drives)
            if method=='fifo': retained=[order[-1]]
            elif method=='oldest': retained=[order[0]]
            elif method=='action_rule': retained=[job['action_slot']]
            elif method=='archive': retained=list(memory)
            elif method=='structured': retained=[]
            elif method in ('learned_action','learned_no_action'):
                begin=time.monotonic()
                features=[public_features(memory[s],s,order.index(s),action) for s in range(4)]
                scores=predict_scores(job['checkpoint'],features)
                if not np.isfinite(scores).all(): raise RuntimeError('Non-finite learned scores')
                retained=[int(np.argmax(scores))]
                report.update(scores=scores.tolist(),score_seconds=time.monotonic()-begin)
                del features,scores
            else: raise ValueError('Unknown method')
            memory={s:im for s,im in memory.items() if s in retained}
            report.update(retained_slots=retained,image_payload_bytes=sum(im.nbytes for im in memory.values()),
                          semantic_payload_bytes=len(json.dumps(semantics).encode()) if semantics else 0,
                          capacity_scope='one-shot compression from four candidates to one persistent image')
            env.arm_latches()
            with physical_latch_steps(env): execute_skill(env,controller,job['action_slot'])
            env.latch_armed=False
            if [e['slot'] for e in env.latch_events]!=[job['action_slot']]:
                raise RuntimeError('Action failed to actuate its latch')
            queries=[]
            for slot in (2,0,3,1):
                if method=='structured': answer=semantics[slot]
                else: answer=controller.read(slot,memory[slot]).answer if slot in memory else None
                steps=0
                if answer is None:
                    result=execute_skill(env,controller,slot)
                    answer=result['read'].answer;steps=result['physics_steps']
                    if result['closed_read'].answer is not None: raise RuntimeError('Recovery closure failed')
                    del result
                queries.append(dict(slot=slot,answer=answer,success=bool(answer==env.block_color_names[slot]),
                                    recovery_steps=steps))
            report.update(queries=queries,correct=sum(q['success'] for q in queries),
                query_count=4,recovery_steps=sum(q['recovery_steps'] for q in queries),
                forbidden_calls=audit['forbidden_calls'],events=env.latch_events)
        report.update(status='complete',elapsed_seconds=time.monotonic()-start);save()
    except Exception:
        report.update(status='failed',error=traceback.format_exc());save();raise
    finally:
        if env is not None: close_env(env)


if __name__=='__main__': main()
