"""Replay already recorded native actions to audit the lift metric, no policy."""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    job = json.loads(parser.parse_args().job.read_text())
    from recoverability.sim_runtime import configure, make_env, close_env, public_observation
    configure(ROOT, str(job['gpu']))
    import numpy as np
    from recoverability.snapshot import physics_vector, digest
    from recoverability.fixed_skills import deny_oracle_actions
    from native_control_worker import observe_lifts
    from resume_native_controls import atomic_json
    branch = Path(job['branch'])
    original = json.loads((branch / 'result.json').read_text())
    source = Path(original['job']['source'])
    parent = original['parent']
    out = Path(job['output'])
    out.mkdir()
    report = dict(status='running', job=job, original_condition=original['job']['condition'],
                  original_metric=original['intended_cover_actuated'], formal_training_eligible=False)
    def save():
        atomic_json(out / 'result.json', report)
    env = None
    try:
        env = make_env(ROOT, 'cover_blocks_hard', original['job']['seed'])
        initial_z = [float(cover.get_pose().p[2]) for cover in env.covers]
        with deny_oracle_actions(env) as oracle:
            with np.load(source / 'prefix_actions.npz', allow_pickle=False) as data:
                for i, action in enumerate(data['actions']):
                    env.get_obs()
                    env.take_action(action)
                    if i % 100 == 0:
                        report.update(stage='prefix', actions=i+1)
                        save()
            parent_checks = dict(physics=digest(physics_vector(env)) == parent['physics_hash'],
                                 observation=digest(public_observation(env.get_obs())) == parent['observation_hash'])
            if not all(parent_checks.values()):
                raise RuntimeError('Audit replay did not match original parent')
            if env.final_pointer != 0 or original['target_slot'] != int(env.color_order_slots[0]):
                raise RuntimeError('Native first-open audit requires the original first color target')
            parent_z = [float(cover.get_pose().p[2]) for cover in env.covers]
            report.update(parent_checks=parent_checks, initial_cover_z=initial_z, parent_cover_z=parent_z,
                          parent_cover_lift_m=[p-z for p,z in zip(parent_z,initial_z)],
                          target_slot=original['target_slot'], parent_task_phase=env.phase,
                          parent_final_pointer=env.final_pointer, original_parent_step=parent['step'])
            transitions = []
            covers_trace = []
            events = []
            phase = None
            with np.load(branch / 'executed_actions.npz', allow_pickle=False) as data:
                actions = data['actions']
            if digest(actions) != original['action_hash']:
                raise RuntimeError('Recorded branch actions do not match original action hash')
            with observe_lifts(env, events):
                for i, action in enumerate(actions):
                    env.get_obs()
                    env.take_action(action)
                    current_phase = (env.phase, env.inspect_pointer, env.final_pointer, bool(env.fail_flag))
                    if current_phase != phase:
                        transitions.append(dict(branch_action=i+1, task_step=env.take_action_cnt, state=current_phase))
                        phase = current_phase
                    covers_trace.append(dict(action=i+1, z=[float(cover.get_pose().p[2]) for cover in env.covers]))
                    if i % 25 == 0:
                        report.update(stage='branch_replay', actions=i+1, transitions=transitions)
                        save()
            final_match = digest(physics_vector(env)) == original['final_physics_hash']
            event_match = events == original['events']
            if not final_match or not event_match:
                raise RuntimeError('Recorded action replay did not match final physics or original metric events')
            native_first_open = any(t['state'][2] >= 1 for t in transitions)
            report.update(status='complete', parent_checks=parent_checks, final_physics_match=final_match,
                          original_metric_events_match=event_match, relative_lift_events=events,
                          task_transitions=transitions, native_first_target_opened=native_first_open,
                          native_first_target_false_negative=not original['intended_cover_actuated'] and native_first_open,
                          final_task_state=current_phase, oracle_calls=oracle['forbidden_calls'],
                          actions=len(actions))
            atomic_json(out / 'cover_heights.json', covers_trace)
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
