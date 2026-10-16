"""Validate shared physical reinspection from exact native branch parents."""
import argparse
import faulthandler
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);job=json.loads(p.parse_args().job.read_text())
    faulthandler.enable();faulthandler.dump_traceback_later(90,repeat=True)
    from recoverability.sim_runtime import configure,make_env,close_env,public_observation
    configure(ROOT,str(job['gpu']))
    import numpy as np
    from PIL import Image
    from recoverability.snapshot import physics_vector,digest
    from recoverability.fixed_skills import SharedRecoveryController,deny_oracle_actions,measured_proprioception
    from recoverability.native_recovery import RecoveryProtocol,recovery_task_mode,sweep_recover
    from recoverability.camera_capture import serialize_camera_capture
    from native_boundary_worker import visible_blocks
    from resume_native_controls import atomic_json
    out=Path(job['output']);out.mkdir();source=Path(job['source'])
    parent=json.loads((source/'parent.json').read_text())
    report=dict(status='running',job=job,formal_training_eligible=False,scope='native_parent_shared_recovery_gate')
    def save():atomic_json(out/'result.json',report)
    env=None;start=time.monotonic()
    try:
        env=make_env(ROOT,'cover_blocks_hard',job['seed'])
        serialize_camera_capture(env);report['camera_capture']='serialized_readback'
        controller=SharedRecoveryController(job['library'])
        if job['seed']==controller.manifest['calibration_seed']:raise ValueError('Calibration/evaluation seed overlap')
        report.update(library_hash=controller.library_hash,calibration_seed=controller.manifest['calibration_seed'])
        with deny_oracle_actions(env) as audit:
            with np.load(source/'prefix_actions.npz',allow_pickle=False) as f:prefix=f['actions']
            for i,a in enumerate(prefix):
                env.get_obs();env.take_action(a)
                if i%100==0:report.update(stage='prefix_replay',replay_actions=i+1);save()
            current=public_observation(env.get_obs())
            checks=dict(physics=digest(physics_vector(env))==parent['physics_hash'],observation=digest(current)==parent['observation_hash'])
            if not all(checks.values()):raise RuntimeError('Parent mismatch: '+repr(checks))
            report.update(parent_match=checks,parent_step=parent['step'],parent_proprioception=measured_proprioception(env.robot).tolist(),
                          parent_visibility=visible_blocks(env))
            original_flags=(env.phase,env.inspect_pointer,env.final_pointer,bool(env.fail_flag))
            protocol=RecoveryProtocol();protocol.begin(already_failed=bool(env.fail_flag))
            observations=[]
            def observe(kind,slot):
                value=public_observation(env.get_obs())
                attempt=1+sum(o['kind']==kind and o['slot']==slot for o in observations)
                for view,image in value['images'].items():Image.fromarray(image).save(out/f'{kind}_{slot}_attempt{attempt}_{view}.png')
                for view,image in value['images'].items():Image.fromarray(image).save(out/f'{kind}_{slot}_{view}.png')
                observations.append(dict(kind=kind,slot=slot,attempt=attempt,recovery_physics_step=protocol.physics_steps,observation_hash=digest(value)))
                return value
            def after_slot(record):
                report.setdefault('slots',[]).append(record)
                report.update(stage='recover',recovery_physics_steps=protocol.physics_steps);save()
            with recovery_task_mode(env,protocol):
                report['stage']='recover';save()
                recovery=sweep_recover(env,controller,observe=observe,after_slot=after_slot)
                flags=(env.phase,env.inspect_pointer,env.final_pointer,bool(env.fail_flag))
                if flags!=original_flags:raise RuntimeError('Recovery altered native submission state')
                protocol.commit(colours_hidden=recovery['colours_hidden'],grippers_open=recovery['grippers_open'])
                # The normal task checker is active again; no error flag is cleared.
                env.update_state_transition()
                if (env.phase,env.inspect_pointer,env.final_pointer,bool(env.fail_flag))!=original_flags:
                    raise RuntimeError('Recovered scene cannot enter commit mode')
            # Ground truth enters only this external scorer, after motion has finished.
            truth=dict(zip(env.color_order_slots,['red','green','blue','yellow']))
            reads_correct=[None if row['read']['answer'] is None else row['read']['answer']==truth[row['slot']]
                           for row in recovery['records']]
            initial=np.asarray(parent['initial_cover_positions']);positions=np.asarray([c.get_pose().p for c in env.covers])
            blocks=np.asarray([b.get_pose().p for b in env.blocks])
            lifts=positions[:,2]-initial[:,2];offsets=np.linalg.norm(positions[:,:2]-blocks[:,:2],axis=1)
            visibility=visible_blocks(env)
            motion_gate=bool(all(x is not False for x in reads_correct) and np.max(np.abs(lifts))<.01 and np.max(offsets)<.03
                      and visibility['all_hidden'] and recovery['colours_hidden'] and recovery['grippers_open'])
            gate=bool(motion_gate and all(reads_correct))
            report.update(status='complete',recovery=recovery,observations=observations,reads_correct=reads_correct,
                gate_passed=gate,motion_gate_passed=motion_gate,all_evidence_recovered=all(reads_correct),
                recovered_slots=recovery['recovered_slots'],unrecovered_slots=recovery['unrecovered_slots'],
                native_flags_preserved=True,final_visibility=visibility,
                final_cover_lifts_m=lifts.tolist(),final_cover_offsets_m=offsets.tolist(),
                recovery_physics_steps=protocol.physics_steps,suppressed_task_updates=protocol.suppressed_updates,
                final_protocol_mode=protocol.mode,forbidden_calls=audit['forbidden_calls'],
                final_physics_hash=digest(physics_vector(env)),elapsed_seconds=time.monotonic()-start)
            save()
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());save();raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if env is not None:close_env(env)


if __name__=='__main__':main()
