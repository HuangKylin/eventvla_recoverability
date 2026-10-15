"""Fresh-process K/D/R calibration with an RGB-only answer reader and fixed drives."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('job',type=Path)
    args=parser.parse_args()
    job=json.loads(args.job.read_text())
    from recoverability.sim_runtime import configure,make_env,close_env,public_observation
    configure(ROOT,job.get('gpu','1'))
    import numpy as np
    from PIL import Image
    from recoverability.cue_probe_task import revisit_or_erase_cue_probe_v0
    from recoverability.cue_reader import ProbeMemory,read_colour
    from recoverability.snapshot import replay,physics_vector,digest,rng_state
    env=None
    program=job['program']
    for name,expected in program['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Compiled program source changed: '+name)
    output=Path(job['output'])
    output.mkdir(parents=True,exist_ok=False)
    try:
        env=make_env(ROOT,program['task'],program['seed'],task_class=revisit_or_erase_cue_probe_v0)
        prefix=np.load(program['stages']['prefix']['tape'],allow_pickle=False)['drives']
        recovery=np.load(program['stages']['recovery']['tape'],allow_pickle=False)['drives']
        for name,tape in (('prefix',prefix),('recovery',recovery)):
            if digest(tape)!=program['stages'][name]['tape_hash']:
                raise RuntimeError('Compiled drive tape checksum mismatch: '+name)
        observed=int(program['stages']['prefix']['observe_after_steps'])
        prefix_a=replay(env,prefix[:observed])
        evidence=public_observation(env.get_obs())['images']['head_camera']
        prefix_b=replay(env,prefix[observed:])
        parent_public=public_observation(env.get_obs())
        parent=physics_vector(env)
        state_fields=dict(phase=env.phase,probe_command=env.probe_command,
                          probe_submitted=env.probe_submitted,probe_inspections=env.probe_inspections,
                          take_action_cnt=env.take_action_cnt)
        parent_hash=digest((parent,state_fields))
        rng_hash=digest(rng_state())
        evidence_hash=digest(evidence)
        prefix_trace=np.concatenate((prefix_a,prefix_b))
        memory=ProbeMemory(evidence)
        if job['branch'] in {'D','R'}:
            memory.drop()
        elif job['branch']!='K':
            raise ValueError('Unknown branch')
        # Evaluator-only evidence is saved for audit, then released from controller scope.
        Image.fromarray(evidence).save(output/'evidence.png')
        del evidence
        # The proposed apparatus action executes AFTER the memory intervention.
        started=time.monotonic()
        env.apply_apparatus_command(job['command'])
        current=public_observation(env.get_obs())['images']['head_camera']
        current_read=read_colour(current)
        answer=current_read.answer or memory.answer()
        recovery_used=answer is None and job['branch']!='D'
        trace=np.empty((0,len(parent)))
        recovery_read=None
        recovery_steps=0
        if recovery_used:
            at=int(program['stages']['recovery']['observe_after_steps'])
            first=replay(env,recovery[:at])
            rgb=public_observation(env.get_obs())['images']['head_camera']
            recovery_read=read_colour(rgb)
            answer=recovery_read.answer
            Image.fromarray(rgb).save(output/'recovery.png')
            second=replay(env,recovery[at:])
            trace=np.concatenate((first,second))
            recovery_steps=len(recovery)
        success=env.submit_answer(answer)
        elapsed=time.monotonic()-started
        np.savez_compressed(output/'evaluator_trace.npz',parent=parent,prefix_trace=prefix_trace,
                            recovery_trace=trace,parent_head=parent_public['images']['head_camera'])
        outcome=dict(status='complete',seed=program['seed'],task=program['task'],
                     family_id=f"cue_probe_v0_seed_{program['seed']}",
                     parent_state_hash=parent_hash,rng_state_hash=rng_hash,
                     evidence_id=evidence_hash,action_context_hash=digest(job['command']),
                     controller_hash=job['controller_hash'],branch=job['branch'],command=job['command'],
                     success=bool(success),answer=answer,steps=recovery_steps,
                     prefix_physics_steps=len(prefix),continuation_wall_seconds=elapsed,
                     recovery_allowed=job['branch']!='D',recovery_used=recovery_used,
                     current_read=current_read.__dict__,
                     recovery_read=None if recovery_read is None else recovery_read.__dict__,
                     persistent_evidence_bytes=memory.nbytes,
                     privileged_drive_compilation=True,
                     controller_kind='fixed_drive_RGB_probe',
                     observation_contract='RGB plus public command; fixed calibrated ROI')
        (output/'outcome.json').write_text(json.dumps(outcome,indent=2))
        print(json.dumps(outcome),flush=True)
    finally:
        if env is not None:
            close_env(env)


if __name__=='__main__':
    main()
