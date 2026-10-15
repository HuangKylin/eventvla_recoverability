"""Actual EventVLA closed-loop actions on the physical-latch task.

Expert-free shared motor prefix provides controlled history; after intervention
every action comes from the frozen checkpoint. Success is cover actuation, not
a symbolic answer or a published benchmark score.
"""
import argparse
import faulthandler
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    faulthandler.enable()
    faulthandler.dump_traceback_later(60,repeat=True)
    p=argparse.ArgumentParser();p.add_argument('job',type=Path)
    job=json.loads(p.parse_args().job.read_text())
    from recoverability.sim_runtime import configure,make_env,close_env,public_observation
    configure(ROOT,job.get('gpu','1'))
    sys.path.insert(0,str(ROOT/'third_party/EventVLA/EventVLA'))
    import numpy as np
    from PIL import Image
    from recoverability.action_latch_task import action_latch_v1,physical_latch_steps
    from recoverability.fixed_skills import SharedRecoveryController,deny_oracle_actions,measured_proprioception
    from recoverability.snapshot import apply_drive_vector,physics_vector,digest
    from recoverability.vla_intervention import purge_evidence,audit_payload
    path=ROOT/'third_party/EventVLA/EventVLA/examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py'
    spec=importlib.util.spec_from_file_location('upstream_client',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    audits=[]
    class AuditedClient(module.ModelClient):
        forbidden=()
        def _call_server_with_reconnect(self,name,*args):
            if name=='predict_action':
                payload=args[0]
                payload['paired_seed']=91000+int(payload['examples'][0]['timestep'])
                audit=audit_payload(payload,self.forbidden)
                audit['step']=int(payload['examples'][0]['timestep'])
                audits.append(audit)
            result=super()._call_server_with_reconnect(name,*args)
            self._raise_for_server_error(result)
            if name=='predict_action' and not result['data'].get('server_memory_reset'):
                raise RuntimeError('Server did not confirm cache reset')
            return result
    out=Path(job['output']);out.mkdir(parents=True)
    report=dict(status='running',job=job,scope='eventvla_closed_loop_latch_probe',
        formal_training_eligible=False,request_audits=audits,actions_executed=0)
    def save(): (out/'result.json').write_text(json.dumps(report,indent=2))
    env=model=None;started=time.monotonic()
    try:
        env=make_env(ROOT,'action_latch_v1',job['seed'],task_class=action_latch_v1)
        controller=SharedRecoveryController(job['library'])
        model=AuditedClient(str(ROOT/'checkpoints/runtime/EventVLA-RoboTwin-MeM/final_model/pytorch_model.pt'),
            unnorm_key='robotwin_mem',port=job['port'],first_chunk_random_replan=False)
        instruction=f"Open the {env.cover_name[job['action_slot']]} cover to inspect the block color, then place the cover back."
        model.reset(instruction)
        initial=public_observation(env.get_obs())
        views=('head_camera','left_camera','right_camera')
        model._record_temporal_images([model._resize_image(initial['images'][v]) for v in views],0)
        order=np.random.default_rng(job['seed']+991).permutation(4).tolist()
        evidence_steps={};trace=[]
        with deny_oracle_actions(env) as audit:
            for ordinal,slot in enumerate(order):
                skill,_=controller.select(slot,measured_proprioception(env.robot))
                captured=None
                for i,row in enumerate(skill.drives):
                    apply_drive_vector(env,row);env.scene.step()
                    if i+1==skill.observe_after_steps:
                        captured=public_observation(env.get_obs())
                if captured is None or controller.read(slot,captured['images']['head_camera']).answer!=env.block_color_names[slot]:
                    raise RuntimeError('Controlled prefix perception failed')
                step=10*(ordinal+1);evidence_steps[slot]=step
                frames=[captured['images'][v] for v in views]
                model._record_temporal_images([model._resize_image(im) for im in frames],step)
                model._commit_keyframe_image_memory(frames,step,1.)
                Image.fromarray(frames[0]).save(out/f'evidence_{slot}.png')
                trace.append(physics_vector(env))
                del captured,frames
            current=public_observation(env.get_obs())
            closed=[current['images'][v] for v in views]
            model._record_temporal_images([model._resize_image(im) for im in closed],41)
            report.update(parent_hash=digest(physics_vector(env)),prefix_trace_hash=digest(np.asarray(trace)),
                          current_hash=digest(current),order=order,instruction=instruction)
            dropped=[] if job['branch']=='K' else [evidence_steps[job['action_slot']]]
            # Positive-control: event-only deletion would still leave evidence
            # in temporal history; report its source-record count explicitly.
            report['temporal_records_before_purge']=[r['step'] for r in model.image_history if r['step'] in dropped]
            intervention=purge_evidence(model,dropped)
            model.forbidden=intervention['forbidden_image_hashes']
            report['intervention']=intervention
            del initial,current,closed
            env.arm_latches();actions=[];states=[]
            with physical_latch_steps(env):
                for i in range(job.get('max_actions',200)):
                    report.update(stage='get_observation',next_action=i)
                    save()
                    obs=public_observation(env.get_obs())
                    example=dict(lang=instruction,image=[obs['images'][v] for v in views],state=obs['proprioception'])
                    report['stage']='model_step';save()
                    action=model.step(example,step=41+i)
                    if np.asarray(action).shape!=(14,) or not np.isfinite(action).all():
                        raise RuntimeError('Invalid actual VLA action')
                    actions.append(np.asarray(action).copy())
                    report.update(stage='execute_action',last_action=np.asarray(action).tolist());save()
                    env.take_action(action)
                    states.append(measured_proprioception(env.robot))
                    report.update(actions_executed=len(actions),events=env.latch_events)
                    if i%25==0: save()
                    if env.latch_events: break
            env.latch_armed=False
            report.update(forbidden_calls=audit['forbidden_calls'],
                intended_cover_actuated=[e['slot'] for e in env.latch_events]==[job['action_slot']],
                action_hash=digest(np.asarray(actions)),joint_trace_hash=digest(np.asarray(states)),
                final_physics_hash=digest(physics_vector(env)))
            np.savez_compressed(out/'executed_actions.npz',actions=np.asarray(actions),measured_joints=np.asarray(states))
            final=public_observation(env.get_obs())
            Image.fromarray(final['images']['head_camera']).save(out/'final.png')
        report.update(status='complete',elapsed_seconds=time.monotonic()-started)
        save()
    except Exception:
        report.update(status='failed',error=traceback.format_exc());save();raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if model is not None and model.client is not None: model.client.close()
        if env is not None: close_env(env)


if __name__=='__main__': main()
