"""Collect fresh native parents without expert actions or success filtering."""
import argparse
import faulthandler
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);job=json.loads(p.parse_args().job.read_text())
    from recoverability.sim_runtime import configure,make_env,close_env,public_observation
    configure(ROOT,str(job['gpu']));sys.path.insert(0,str(ROOT/'third_party/EventVLA/EventVLA'))
    import numpy as np
    from PIL import Image
    from recoverability.camera_capture import serialize_camera_capture
    from recoverability.fixed_skills import deny_oracle_actions
    from recoverability.native_boundary import assess_parent
    from recoverability.snapshot import physics_vector,digest
    from recoverability.vla_intervention import audit_payload
    from native_boundary_worker import visible_blocks
    from resume_native_controls import atomic_json
    interface=ROOT/'third_party/EventVLA/EventVLA/examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py'
    spec=importlib.util.spec_from_file_location('parent_client',interface);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=Path(job['output']);out.mkdir();audits=[];actions=[];env=model=None;start=time.monotonic()
    report=dict(status='running',job=job,parent_saved=False,formal_training_eligible=False,request_audits=audits)
    def save():atomic_json(out/'result.json',report)
    class Client(module.ModelClient):
        def _call_server_with_reconnect(self,name,*args):
            if name=='predict_action':
                step=int(args[0]['examples'][0]['timestep']);args[0]['paired_seed']=91000+step
                audits.append(dict(audit_payload(args[0],()),step=step,paired_seed=91000+step))
            result=super()._call_server_with_reconnect(name,*args);self._raise_for_server_error(result)
            if name=='predict_action' and not result['data'].get('server_memory_reset'):raise RuntimeError('Server reset missing')
            return result
    faulthandler.enable();faulthandler.dump_traceback_later(90,repeat=True)
    try:
        env=make_env(ROOT,'cover_blocks_hard',job['seed']);serialize_camera_capture(env)
        initial=np.asarray([c.get_pose().p for c in env.covers])
        native=json.loads((ROOT/'third_party/EventVLA/RoboTwin-Mem/description/task_instruction/cover_blocks_hard.json').read_text())['seen'][0]
        model=Client(str(ROOT/'checkpoints/runtime/EventVLA-RoboTwin-MeM/final_model/pytorch_model.pt'),unnorm_key='robotwin_mem',port=job['port'])
        model.progress_output_dir=out/'progress';model.progress_output_dir.mkdir()
        model.keyframe_output_dir=out/'keyframes';model.keyframe_output_dir.mkdir();model.reset(native)
        views=('head_camera','left_camera','right_camera');settled=0;trace=[]
        def geometry():
            return assess_parent(positions=[c.get_pose().p for c in env.covers],initial_positions=initial,
                block_positions=[b.get_pose().p for b in env.blocks],
                grippers=[env.robot.get_left_gripper_val(),env.robot.get_right_gripper_val()],
                phase=env.phase,inspect_pointer=env.inspect_pointer,final_pointer=env.final_pointer,failed=bool(env.fail_flag),
                step=env.take_action_cnt,plan_start=model.plan_start_step,
                plan_length=0 if model.raw_actions is None else len(model.raw_actions),settled_steps=settled)
        with deny_oracle_actions(env) as guard:
            for i in range(min(int(job.get('max_actions',1700)),env.step_lim)):
                obs=public_observation(env.get_obs())
                action=model.step(dict(lang=native,image=[obs['images'][v] for v in views],state=obs['proprioception']),step=env.take_action_cnt)
                if np.asarray(action).shape!=(14,) or not np.isfinite(action).all():raise RuntimeError('Invalid action')
                actions.append(np.asarray(action).copy());env.take_action(action)
                g=geometry();stable=all(g['checks'][k] for k in ['covers_down','covers_over_blocks','grippers_open'])
                settled=settled+1 if stable else 0;g=geometry()
                report.update(stage='native_parent_collection',actions_executed=len(actions),boundary=g)
                if env.phase=='final':trace.append(g)
                if i%25==0:save()
                if g['eligible']:
                    current=public_observation(env.get_obs());visibility=visible_blocks(env)
                    event_steps=[int(e['step']) for e in model.keyframe_image_memory]
                    valid_events=len(event_steps)==4 and event_steps==sorted(set(event_steps)) and not set(event_steps)&{int(e['step']) for e in model.image_history}
                    report.update(boundary_visibility=visibility,boundary_event_steps=event_steps,boundary_events_valid=valid_events)
                    if visibility['all_hidden'] and valid_events:
                        parent=dict(seed=job['seed'],step=env.take_action_cnt,target_slot=int(env.color_order_slots[0]),
                            physics_hash=digest(physics_vector(env)),observation_hash=digest(current),
                            history_hash=digest(list(model.image_history)),event_hash=digest(model.keyframe_image_memory),
                            event_steps=event_steps,temporal_steps=[int(e['step']) for e in model.image_history],
                            native_instruction=native,initial_cover_positions=initial.tolist(),boundary=g,visibility=visibility,
                            provenance='Fresh predeclared seed; no post-branch-success selection')
                        state={k:v for k,v in model.__dict__.items() if k!='client'}
                        raw=pickle.dumps(state,protocol=5);restored=pickle.loads(raw)
                        if digest(restored['keyframe_image_memory'])!=parent['event_hash']:raise RuntimeError('Snapshot event roundtrip mismatch')
                        if digest(list(restored['image_history']))!=parent['history_hash']:raise RuntimeError('Snapshot history roundtrip mismatch')
                        (out/'client_state.pkl').write_bytes(raw)
                        np.savez_compressed(out/'prefix_actions.npz',actions=np.asarray(actions))
                        for v in views:Image.fromarray(current['images'][v]).save(out/f'parent_{v}.png')
                        atomic_json(out/'parent.json',parent);report.update(parent_saved=True,parent=parent);break
                if env.fail_flag or env.final_pointer>0:
                    report['rejection_reason']='Native failure or final submission before eligible closed boundary';break
            report['forbidden_calls']=guard['forbidden_calls']
        if not report['parent_saved']:report.setdefault('rejection_reason','No eligible parent before native action budget')
        atomic_json(out/'boundary_trace.json',trace)
        np.savez_compressed(out/'executed_actions.npz',actions=np.asarray(actions))
        report.update(status='complete',elapsed_seconds=time.monotonic()-start);save()
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());save();raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if model is not None and model.client is not None:model.client.close()
        if env is not None:close_env(env)


if __name__=='__main__':main()
