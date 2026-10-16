"""Audited native continuations: bundle, per-item pilot and content controls.

These development protocols do not produce formal scorer training labels.
"""
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
    faulthandler.enable();faulthandler.dump_traceback_later(90,repeat=True)
    from recoverability.sim_runtime import configure,make_env,close_env,public_observation
    configure(ROOT,str(job['gpu']));sys.path.insert(0,str(ROOT/'third_party/EventVLA/EventVLA'))
    import numpy as np
    from PIL import Image
    from recoverability.fixed_skills import SharedRecoveryController,deny_oracle_actions,measured_proprioception
    from recoverability.native_recovery import RecoveryProtocol,recovery_task_mode,sweep_recover
    from recoverability.camera_capture import serialize_camera_capture
    from recoverability.native_memory import intervene
    from recoverability.native_items import intervene_item,candidate_slot,event_signature,recover_candidate,replace_item_content
    from recoverability.native_boundary import AbsoluteLiftTracker
    from recoverability.snapshot import physics_vector,digest
    from recoverability.vla_intervention import purge_evidence,audit_payload,image_hash
    from native_boundary_worker import visible_blocks,observe_absolute
    from resume_native_controls import atomic_json
    path=ROOT/'third_party/EventVLA/EventVLA/examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py'
    spec=importlib.util.spec_from_file_location('recovery_client',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=Path(job['output']);out.mkdir();source=Path(job['source']);parent=json.loads((source/'parent.json').read_text())
    condition=job['condition']
    item_pilot=condition.startswith(('drop_item_','recover_item_','mask_item_'))
    candidate_index=int(condition.rsplit('_',1)[1]) if item_pilot else None
    if item_pilot:condition='recover_item' if condition.startswith('recover') else ('mask_item' if condition.startswith('mask') else 'drop_item')
    continuation_probe=condition in ('time_only_original','hold_original')
    if condition not in ('keep','recover','recover_discard','recover_original','time_only_original','hold_original','drop_item','recover_item','mask_item'):raise ValueError('Unknown condition')
    audits=[];forbidden=[];expected=None
    class Client(module.ModelClient):
        def _call_server_with_reconnect(self,name,*args):
            if name=='predict_action':
                example=args[0]['examples'][0];step=int(example['timestep']);args[0]['paired_seed']=91000+step
                record=dict(audit_payload(args[0],forbidden),step=step,paired_seed=91000+step,language=example.get('lang'),
                            event_steps=example.get('memory_keyframe_steps',[]),event_metas=example.get('memory_keyframe_image_metas',[]),
                            anchor_metas=example.get('image_metas',[]))
                if not audits:
                    if any(record[k]!=expected[k] for k in ['images','event_steps','event_metas','anchor_metas','language']):
                        raise RuntimeError('First actual payload differs from reconstructed recovery input')
                    record['first_input_verified']=True
                audits.append(record)
            result=super()._call_server_with_reconnect(name,*args);self._raise_for_server_error(result)
            if name=='predict_action' and not result['data'].get('server_memory_reset'):raise RuntimeError('Server reset missing')
            return result
    report=dict(status='running',job=job,request_audits=audits,formal_training_eligible=False,
                recovery_allowed=True,scope='all_event_bundle_recovery_integration')
    def save():atomic_json(out/'result.json',report)
    env=model=None;actions=[];started=time.monotonic();views=('head_camera','left_camera','right_camera')
    try:
        env=make_env(ROOT,'cover_blocks_hard',job['seed'])
        serialize_camera_capture(env);report['camera_capture']='serialized_readback'
        controller=SharedRecoveryController(job['library'])
        if job['seed']==controller.manifest['calibration_seed']:raise ValueError('Calibration/evaluation seed overlap')
        model=Client(str(ROOT/'checkpoints/runtime/EventVLA-RoboTwin-MeM/final_model/pytorch_model.pt'),unnorm_key='robotwin_mem',port=job['port'])
        model.progress_output_dir=out/'progress';model.progress_output_dir.mkdir()
        model.keyframe_output_dir=out/'keyframes';model.keyframe_output_dir.mkdir();model.reset(parent['native_instruction'])
        def obs():return public_observation(env.get_obs())
        with deny_oracle_actions(env) as guard:
            with np.load(source/'prefix_actions.npz',allow_pickle=False) as f:prefix=f['actions']
            for i,action in enumerate(prefix):
                if not job.get('fast_prefix',False):env.get_obs()
                env.take_action(action)
                if i%100==0:report.update(stage='prefix_replay',replay_actions=i+1);save()
            current=obs();state=pickle.loads((source/'client_state.pkl').read_bytes())
            excluded={'client','port','host','progress_output_dir','keyframe_output_dir','current_episode_keyframe_dir','progress_records'}
            model.__dict__.update({k:v for k,v in state.items() if k not in excluded})
            checks=dict(physics=digest(physics_vector(env))==parent['physics_hash'],observation=digest(current)==parent['observation_hash'],
                        history=digest(list(model.image_history))==parent['history_hash'],events=digest(model.keyframe_image_memory)==parent['event_hash'])
            if not all(checks.values()):raise RuntimeError('Native parent mismatch')
            if item_pilot:
                if candidate_index not in range(len(model.keyframe_image_memory)):raise ValueError('Candidate index outside event memory')
                candidate=model.keyframe_image_memory[candidate_index]
                slot=candidate_slot(candidate,controller.read)
                candidate_step=int(candidate['step'])
                # Persist only pre-intervention public inputs, separately from outcome/seed metadata.
                np.savez_compressed(out/'candidate_public.npz',candidate_head=candidate['images'][0],
                    current_head=current['images']['head_camera'],current_left=current['images']['left_camera'],
                    current_right=current['images']['right_camera'],proprioception=current['proprioception'])
                atomic_json(out/'candidate_public.json',dict(language=parent['native_instruction'],
                    candidate_step=candidate_step,decision_step=parent['step'],public_slot=slot,
                    event_steps=[int(e['step']) for e in model.keyframe_image_memory],
                    action_condition='native_next_action_not_paired',feature_schema='public_predecision_v1'))
                if condition=='mask_item':
                    change=replace_item_content(model,candidate_step,model._resize_image(current['images']['head_camera']))
                else:change=intervene_item(model,candidate_step,drop=True)
                report.update(scope='per_item_development_pilot',candidate_index=candidate_index,
                    candidate_step=candidate_step,public_recovery_slot=slot,endpoint='first_correct_target_reveal',
                    training_exclusion='Two tuned development parents; no action pairs or full-task labels',
                    recovery_allowed=condition=='recover_item')
                if condition=='mask_item':report.update(scope='constant_count_content_probe',training_exclusion='Synthetic content intervention; never a training label')
            else:
                change=intervene(model,'keep' if condition in ('keep','recover_original') or continuation_probe else 'drop_all')
            forbidden.extend(change['forbidden_image_hashes'])
            report.update(parent_match=checks,intervention=change,library_hash=controller.library_hash)
            protocol=RecoveryProtocol();flags=(env.phase,env.inspect_pointer,env.final_pointer,bool(env.fail_flag));observations=[];capture_counts={}
            with recovery_task_mode(env,protocol):
                if condition not in ('keep','drop_item','mask_item'):
                    protocol.begin(already_failed=bool(env.fail_flag))
                    virtual_ticks=0
                    def clock():return parent['step']+((virtual_ticks if condition=='time_only_original' else protocol.physics_steps)+79)//80
                    def record_temporal(value):
                        env.take_action_cnt=clock()
                        model._record_temporal_images([model._resize_image(value['images'][v]) for v in views],clock())
                    closed_phase=False;closed_visibility=[]
                    def after_tick():
                        if protocol.physics_steps%80==0:
                            record_temporal(obs())
                            if closed_phase:
                                # External evaluator audit only; never used to select actions.
                                closed_visibility.append(dict(step=clock(),**visible_blocks(env)))
                    def observe(kind,slot):
                        value=obs();record_temporal(value)
                        key=(kind,slot);capture_counts[key]=capture_counts.get(key,0)+1
                        for v in views:Image.fromarray(value['images'][v]).save(out/f'{kind}_{slot}_attempt{capture_counts[key]}_{v}.png')
                        for v in views:Image.fromarray(value['images'][v]).save(out/f'{kind}_{slot}_{v}.png')
                        if kind=='visible' and controller.read(slot,value['images']['head_camera']).answer is not None:
                            if condition!='recover_original':
                                model._commit_keyframe_image_memory([value['images'][v] for v in views],clock(),1.)
                            observations.append(dict(slot=slot,step=clock(),physics_step=protocol.physics_steps,
                                image_hash=image_hash(model._resize_image(value['images']['head_camera']))))
                        return value
                    def after_slot(record):
                        report.setdefault('slots',[]).append(record);report.update(stage='recover',recovery_physics_steps=protocol.physics_steps);save()
                    report['stage']='recover';save()
                    if continuation_probe:
                        target_ticks=int(job['matched_recovery_ticks'])
                        if not 2560<=target_ticks<=protocol.max_physics_steps:raise ValueError('Invalid matched duration')
                        recovery=dict(kind=condition,recovered_slots=[],unrecovered_slots=[],matched_ticks=target_ticks)
                        if condition=='hold_original':
                            closed_phase=True
                            for _ in range(target_ticks):env.scene.step();after_tick()
                        else:
                            # Counterfactual input diagnostic only: no simulation step and no new evidence.
                            # Reproduce the elapsed-clock/queue update schedule with the frozen parent RGB.
                            for virtual_ticks in range(80,target_ticks+1,80):record_temporal(current)
                            virtual_ticks=target_ticks;record_temporal(current)
                            closed_visibility.append(dict(step=clock(),**visible_blocks(env)))
                            if digest(physics_vector(env))!=parent['physics_hash']:raise RuntimeError('Time-only probe changed physics')
                        report.update(scope='continuation_adaptation_diagnostic',matched_recovery_ticks=target_ticks,
                                      virtual_physics_ticks=target_ticks if condition=='time_only_original' else 0,
                                      training_exclusion='Synthetic time/hold diagnostic; never a K/R label')
                    elif condition=='recover_item':
                        recovery=recover_candidate(env,controller,slot,observe=observe,after_tick=after_tick,after_slot=after_slot)
                    else:
                        recovery=sweep_recover(env,controller,observe=observe,after_tick=after_tick,after_slot=after_slot)
                    # Flush the entire actual queue in BOTH controls, including non-anchor frames.
                    # Hold the same joint targets; charge every physical tick.
                    hold_ticks=0 if continuation_probe else max(32,int(model.temporal_history_capacity)+1)*80
                    closed_phase=True
                    for _ in range(hold_ticks):env.scene.step();after_tick()
                    current=obs();record_temporal(current)
                    closed_history=all(controller.read(slot,row['images'][0]).answer is None
                        for row in model.image_history for slot in range(4))
                    if not closed_history:raise RuntimeError('Recent temporal queue still contains visible recovery evidence')
                    if not closed_visibility or not all(row['all_hidden'] for row in closed_visibility):
                        raise RuntimeError('A closed-history camera exposed evidence during the fixed hold')
                    if condition=='recover_discard':
                        removed=purge_evidence(model,[e['step'] for e in model.keyframe_image_memory]);forbidden.extend(removed['forbidden_image_hashes'])
                    else:purge_evidence(model,[])
                    if condition=='recover_original':forbidden.extend(o['image_hash'] for o in observations)
                    report.update(recovery=recovery,recovered_observations=observations,
                                  recovery_physics_steps=protocol.physics_steps,closed_settle_ticks=hold_ticks,
                                  temporal_history_capacity=model.temporal_history_capacity,closed_history_verified=closed_history,
                                  closed_history_visibility=closed_visibility,
                                  model_clock='parent + ceil(matched_ticks/80)' if continuation_probe else 'parent + ceil(recovery_physics_ticks/80)',
                                  suppressed_task_updates=protocol.suppressed_updates)
                else:report['recovery_physics_steps']=0
                hidden=all(controller.read(s,current['images']['head_camera']).answer is None for s in range(4))
                grippers=bool(np.all(measured_proprioception(env.robot)[[6,13]]>.8))
                visibility=visible_blocks(env)
                if not visibility['all_hidden']:raise RuntimeError('Recovered final observation exposes colours')
                if (env.phase,env.inspect_pointer,env.final_pointer,bool(env.fail_flag))!=flags:raise RuntimeError('Recovery changed native task state')
                protocol.commit(colours_hidden=hidden,grippers_open=grippers)
                env.update_state_transition()
                if (env.phase,env.inspect_pointer,env.final_pointer,bool(env.fail_flag))!=flags:raise RuntimeError('Invalid commit parent')
                model._record_temporal_images([model._resize_image(current['images'][v]) for v in views],env.take_action_cnt)
                anchor,meta=model._build_temporal_anchor_images(env.take_action_cnt)
                if item_pilot:
                    others=[e for e in model.keyframe_image_memory if int(e['step'])!=candidate_step and int(e['step']) in change['remaining_event_steps']]
                    if event_signature(others)!=change['noncandidate_signature']:raise RuntimeError('Recovery changed noncandidate memory')
                    report['noncandidate_preserved_at_commit']=True
                    report['candidate_recovered']=bool(observations) if condition=='recover_item' else False
                example=dict(image=anchor,image_metas=meta);model._inject_keyframe_memory_fields(example,env.take_action_cnt)
                expected=dict(audit_payload(dict(examples=[example]),forbidden),event_steps=example['memory_keyframe_steps'],
                    event_metas=example['memory_keyframe_image_metas'],anchor_metas=meta,language=parent['native_instruction'])
                report.update(stage='commit',commit_step=env.take_action_cnt,commit_physics_hash=digest(physics_vector(env)),
                    commit_observation_hash=digest(current),commit_visibility=visibility,expected_first_input=expected,
                    forbidden_image_hashes=sorted(set(forbidden)),commit_history_hash=digest(list(model.image_history)),
                    commit_proprioception=current['proprioception'].tolist(),
                    commit_history_steps=[int(row['step']) for row in model.image_history])
                save()
                tracker=AbsoluteLiftTracker(np.asarray(parent['initial_cover_positions'])[:,2]);reveals=[]
                with observe_absolute(env,tracker):
                    for i in range(job['max_actions']):
                        action=model.step(dict(lang=parent['native_instruction'],image=[current['images'][v] for v in views],
                                               state=current['proprioception']),step=env.take_action_cnt)
                        if np.asarray(action).shape!=(14,) or not np.isfinite(action).all():raise RuntimeError('Invalid action')
                        actions.append(np.asarray(action).copy());env.take_action(action)
                        opened=[s for s in range(4) if env._is_slot_open_for_progress(s)]
                        if opened and not reveals:reveals.append(dict(action=i+1,slots=opened))
                        report.update(stage='continuation',actions_executed=len(actions),lift_events=tracker.events,revealed_events=reveals)
                        if i%25==0:save()
                        if reveals or env.fail_flag or (tracker.events and not tracker.first_target(parent['target_slot'])):break
                        current=obs()
                report.update(first_lift_correct=tracker.first_target(parent['target_slot']),
                    first_reveal_correct=bool(reveals and reveals[0]['slots']==[parent['target_slot']]),
                    target_slot=parent['target_slot'],policy_physics_steps=tracker.tick,native_failed=bool(env.fail_flag),
                    native_first_target_opened=env.final_pointer>0,final_protocol_mode=protocol.mode)
            report['forbidden_calls']=guard['forbidden_calls']
        np.savez_compressed(out/'executed_actions.npz',actions=np.asarray(actions))
        for v,image in obs()['images'].items():Image.fromarray(image).save(out/f'final_{v}.png')
        report.update(status='complete',action_hash=digest(np.asarray(actions)),final_physics_hash=digest(physics_vector(env)),
                      elapsed_seconds=time.monotonic()-started);save()
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());save();raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if model is not None and model.client is not None:model.client.close()
        if env is not None:close_env(env)


if __name__=='__main__':main()
