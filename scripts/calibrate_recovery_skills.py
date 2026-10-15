"""One offline expert calibration; its seed must never be used for evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=424242)
    p.add_argument('--gpu', default='1')
    args = p.parse_args()
    from recoverability.sim_runtime import configure, make_env, close_env, public_observation
    configure(ROOT, args.gpu)
    import numpy as np
    from PIL import Image
    from recoverability.multi_cue_task import multi_cue_versions_v1
    from recoverability.fixed_skills import measured_proprioception, SLOT_ROIS
    from recoverability.snapshot import DriveRecorder
    from recoverability.cue_reader import read_colour
    run = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_shared_skill_library')
    run.mkdir(parents=True)
    report = dict(status='running', run_dir=str(run), calibration_seed=args.seed,
                  controller_scope='fixed_workstation', offline_expert_calibration=True,
                  runtime_object_pose_access=False, skills=[])
    def save():
        text = json.dumps(report, indent=2)
        (run/'library.json').write_text(text)
        (ROOT/'reports/latest_shared_skill_library.json').write_text(text)
    save()
    env = None
    try:
        env = make_env(ROOT, 'multi_cue_versions_v1', args.seed, task_class=multi_cue_versions_v1)
        for slot in range(4):
            print(json.dumps({'stage':'offline_calibration','slot':slot}), flush=True)
            start_state = measured_proprioception(env.robot)
            observe_indices, images = [], []
            original_delay = env.delay
            with DriveRecorder(env) as recorder:
                def delay(*a, **kw):
                    observe_indices.append(len(recorder.rows))
                    images.append(public_observation(env.get_obs())['images']['head_camera'])
                    return original_delay(*a, **kw)
                env.delay = delay
                try:
                    env.inspect_cover(slot)
                finally:
                    env.delay = original_delay
            if not env.plan_success or len(observe_indices) != 1:
                raise RuntimeError('Expert calibration failed')
            read = read_colour(images[0], roi=SLOT_ROIS[slot])
            closed = public_observation(env.get_obs())['images']['head_camera']
            closed_read = read_colour(closed, roi=SLOT_ROIS[slot])
            if not env.score_answer(slot, 0, read.answer) or closed_read.answer is not None:
                raise RuntimeError(f'Calibration observation gate failed for slot {slot}: {read}, {closed_read}')
            file = run/f'slot_{slot}.npz'
            np.savez_compressed(file, drives=np.asarray(recorder.rows), start_proprioception=start_state)
            Image.fromarray(images[0]).save(run/f'slot_{slot}_visible.png')
            Image.fromarray(closed).save(run/f'slot_{slot}_closed.png')
            report['skills'].append(dict(slot=slot, file=file.name,
                sha256=hashlib.sha256(file.read_bytes()).hexdigest(), physics_steps=len(recorder.rows),
                observe_after_steps=observe_indices[0], colour_read=read.__dict__, closed_read=closed_read.__dict__))
            save()
        report['status'] = 'complete'
        save()
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        save()
        raise
    finally:
        if env is not None:
            close_env(env)


if __name__ == '__main__':
    main()
