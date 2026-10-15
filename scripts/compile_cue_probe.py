"""Compile privileged inspection drive tapes; no learned-controller claims."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[100000,100001])
    p.add_argument('--gpu', default='1')
    args=p.parse_args()
    from recoverability.sim_runtime import configure, make_env, close_env, public_observation
    configure(ROOT,args.gpu)
    import numpy as np
    from PIL import Image
    from recoverability.cue_probe_task import revisit_or_erase_cue_probe_v0
    from recoverability.snapshot import DriveRecorder, digest
    from recoverability.cue_reader import read_colour
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_cue_programs')
    run.mkdir(parents=True)
    records=[]
    source_files=['src/recoverability/cue_probe_task.py','src/recoverability/snapshot.py',
                  'src/recoverability/sim_runtime.py','scripts/compile_cue_probe.py']
    source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files}
    def save_status(state, **extra):
        payload=dict(status=state,run_dir=str(run),programs=records,source_hashes=source_hashes,
                     scope='apparatus and branch-collector calibration, not VLA evaluation',**extra)
        content=json.dumps(payload,indent=2)
        (run/'programs.json').write_text(content)
        (ROOT/'reports/latest_cue_programs.json').write_text(content)
    save_status('running')
    for seed in args.seeds:
        env=None
        try:
            env=make_env(ROOT,'revisit_or_erase_cue_probe_v0',seed,
                         task_class=revisit_or_erase_cue_probe_v0)
            stages={}
            for name in ('prefix','recovery'):
                indices=[]
                original_delay=env.delay
                with DriveRecorder(env) as recorder:
                    def observe_delay(*a,**kw):
                        indices.append(len(recorder.rows))
                        public=public_observation(env.get_obs())
                        image=public['images']['head_camera']
                        Image.fromarray(image).save(run/f'{seed}_{name}_cue.png')
                        return original_delay(*a,**kw)
                    env.delay=observe_delay
                    try:
                        env.inspect_cover(0)
                    finally:
                        env.delay=original_delay
                if not env.plan_success or len(indices)!=1 or not recorder.rows:
                    raise RuntimeError('Privileged inspection trajectory compilation failed')
                tape=np.asarray(recorder.rows)
                path=run/f'{seed}_{name}.npz'
                np.savez_compressed(path,drives=tape)
                image=np.asarray(Image.open(run/f'{seed}_{name}_cue.png').convert('RGB'))
                read=read_colour(image)
                stages[name]=dict(tape=str(path),tape_hash=digest(tape),steps=len(tape),
                                  observe_after_steps=indices[0],rgb_read=read.__dict__)
            record=dict(seed=seed,task='revisit_or_erase_cue_probe_v0',stages=stages,
                        privileged_compilation=True,truth_for_audit=env.block_color_names[0],
                        source_hashes=source_hashes)
            records.append(record)
            save_status('running')
            print(json.dumps(record),flush=True)
        except Exception:
            save_status('failed',error=traceback.format_exc(),failed_seed=seed)
            raise
        finally:
            if env is not None:
                close_env(env)
    save_status('complete')


if __name__=='__main__':
    main()
