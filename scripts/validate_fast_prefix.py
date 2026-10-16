"""Check whether unused prefix RGB reads can be omitted for fixed action replay.

This does not run the policy without observations. The original policy history
is restored separately from its trusted snapshot by the real experiment worker.
"""
import argparse,faulthandler,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--gpu',default='1');args=p.parse_args()
    from recoverability.sim_runtime import configure,make_env,close_env,public_observation
    configure(ROOT,args.gpu)
    import numpy as np
    from recoverability.fixed_skills import deny_oracle_actions
    from recoverability.camera_capture import serialize_camera_capture
    from recoverability.snapshot import digest,physics_vector
    from resume_native_controls import atomic_json
    state=json.loads((ROOT/'reports/latest_native_boundary_controls.json').read_text())
    source=Path(next(e['result_file'] for e in state['episodes'] if e['condition']=='prepare' and e['seed']==args.seed)).parent
    parent=json.loads((source/'parent.json').read_text());report=dict(seed=args.seed,status='running',source=str(source))
    dest=ROOT/'reports'/f'fast_prefix_{args.seed}.json';env=None;start=time.monotonic()
    faulthandler.enable();faulthandler.dump_traceback_later(90,repeat=True)
    try:
        env=make_env(ROOT,'cover_blocks_hard',args.seed);serialize_camera_capture(env)
        if env.eval_video_path is not None or env.crazy_random_light:raise RuntimeError('Fast prefix contract excludes video writers or mutable lighting')
        with deny_oracle_actions(env) as guard:
            with np.load(source/'prefix_actions.npz',allow_pickle=False) as f:actions=f['actions']
            for i,action in enumerate(actions):
                env.take_action(action)
                if i%100==0:report['actions']=i+1;atomic_json(dest,report)
            current=public_observation(env.get_obs())
            checks=dict(physics=digest(physics_vector(env))==parent['physics_hash'],observation=digest(current)==parent['observation_hash'],
                        phase=env.phase=='final' and env.inspect_pointer==4 and env.final_pointer==0 and not env.fail_flag)
            report.update(status='complete',checks=checks,passed=all(checks.values()),actions=len(actions),
                          forbidden_calls=guard['forbidden_calls'],elapsed_seconds=time.monotonic()-start,prefix_rgb_captures=0)
            atomic_json(dest,report)
            if not report['passed']:raise RuntimeError('Fast replay does not match native parent')
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());atomic_json(dest,report);raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if env is not None:close_env(env)


if __name__=='__main__':main()
