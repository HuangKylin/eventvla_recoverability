"""Empirically validate restored same-scene contact continuations, without replanning."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=[100000, 100001])
    p.add_argument('--gpu', default='1')
    p.add_argument('--restore-tolerance', type=float, default=1e-5)
    p.add_argument('--rollout-tolerance', type=float, default=1e-3)
    args = p.parse_args()
    from recoverability.sim_runtime import configure, make_env, close_env, public_observation
    configure(ROOT, args.gpu)
    import numpy as np
    from recoverability.snapshot import capture, restore, replay, DriveRecorder, digest, max_error, rng_state
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_snapshot_validation')
    run.mkdir(parents=True)
    status = dict(status='running', pid=os.getpid(), run_dir=str(run), config=vars(args),
                  contract='same_scene_fixed_drive_replay', episodes=[])
    def save():
        payload = json.dumps(status, indent=2)
        (run/'status.json').write_text(payload)
        (ROOT/'reports/latest_snapshot_validation.json').write_text(payload)
    save()
    env = None
    start = time.monotonic()
    try:
        for seed in args.seeds:
            print(json.dumps({'stage':'setup', 'seed':seed}), flush=True)
            env = make_env(ROOT, 'cover_blocks_hard', seed)
            obs = public_observation(env.get_obs())
            parent = capture(env, {'cursor':0, 'memory':[obs['images']['head_camera']]},
                             mode='fixed_drive_replay')
            print(json.dumps({'stage':'compile_privileged_contact_trajectory', 'seed':seed}), flush=True)
            with DriveRecorder(env) as recorder:
                env.inspect_cover(0)
            if not env.plan_success or len(recorder.rows) == 0:
                raise RuntimeError('Expert trajectory compilation failed')
            tape = np.asarray(recorder.rows)
            np.savez_compressed(run/f'tape_{seed}.npz', drives=tape)
            repeats, errors, rng_ok, controller_ok, rgb = [], [], [], [], []
            for repeat in range(3):
                controller, error = restore(env, parent)
                errors.append(error)
                rng_ok.append(digest(rng_state()) == digest(parent.random_state))
                controller_ok.append(digest(controller) == digest(parent.controller_state))
                print(json.dumps({'stage':'replay', 'seed':seed, 'repeat':repeat,
                                  'restore_error':error, 'physics_steps':len(tape)}), flush=True)
                repeats.append(replay(env, tape))
                rgb.append(public_observation(env.get_obs())['images']['head_camera'])
                controller['memory'].clear()  # Must not mutate the stored parent.
            rollout_error = max(max_error(repeats[0], item) for item in repeats[1:])
            # Path tracing has its own RNG; measure noise instead of claiming bitwise image restore.
            rgb_again = public_observation(env.get_obs())['images']['head_camera']
            render_mae = lambda a,b: float(np.abs(a.astype(float)-b.astype(float)).mean())
            result = dict(seed=seed, physics_steps=len(tape), trace_samples=len(repeats[0]),
                          parent_hash=parent.parent_hash, tape_hash=digest(tape),
                          restore_errors=errors, max_continuation_error=rollout_error,
                          rng_exact=all(rng_ok), controller_exact=all(controller_ok),
                          image_repeat_mae=render_mae(rgb[-1],rgb_again),
                          image_branch_mae=render_mae(rgb[0],rgb[-1]),
                          accepted=max(errors)<=args.restore_tolerance and
                          rollout_error<=args.rollout_tolerance and all(rng_ok) and all(controller_ok))
            np.savez_compressed(run/f'traces_{seed}.npz', traces=np.asarray(repeats))
            status['episodes'].append(result)
            save()
            print(json.dumps(result), flush=True)
            close_env(env)
            env = None
        status.update(status='complete', all_accepted=all(x['accepted'] for x in status['episodes']),
                      elapsed_seconds=time.monotonic()-start)
        save()
        if not status['all_accepted']:
            raise RuntimeError('Snapshot continuation gate failed; these states cannot produce labels')
    except Exception:
        status.update(status='failed', error=traceback.format_exc(), elapsed_seconds=time.monotonic()-start)
        save()
        raise
    finally:
        if env is not None:
            close_env(env)


if __name__ == '__main__':
    main()
