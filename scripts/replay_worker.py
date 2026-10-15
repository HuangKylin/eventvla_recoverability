"""One fresh-process replay. Internal evaluator job files contain privileged state paths."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    from recoverability.sim_runtime import configure, make_env, close_env, public_observation
    configure(ROOT, job.get('gpu', '1'))
    import numpy as np
    from recoverability.snapshot import replay, physics_vector, rng_state, digest
    env = None
    try:
        env = make_env(ROOT, job['task'], job['seed'])
        tape = np.load(job['tape'], allow_pickle=False)['drives']
        split = int(job['prefix_steps'])
        if not 0 < split < len(tape):
            raise ValueError('Prefix must end strictly inside the contact trajectory')
        replay(env, tape[:split])
        parent = physics_vector(env)
        public = public_observation(env.get_obs())
        rng_hash = digest(rng_state())
        trace = replay(env, tape[split:])
        output = Path(job['output'])
        np.savez_compressed(output, parent=parent, trace=trace,
                            parent_head=public['images']['head_camera'],
                            parent_proprioception=public['proprioception'])
        output.with_suffix('.json').write_text(json.dumps(dict(
            status='complete', seed=job['seed'], task=job['task'], prefix_steps=split,
            continuation_steps=len(tape)-split, rng_hash=rng_hash,
            controller_contract='fixed_drive_replay', tape_hash=digest(tape)), indent=2))
    finally:
        if env is not None:
            close_env(env)


if __name__ == '__main__':
    main()
