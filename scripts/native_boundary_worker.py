"""Finish an existing native chunk, then run matched safe-boundary controls.

Task geometry and segmentation are evaluator-only. Policy receives RGB and
proprioception through the existing whitelist. No expert actions are called.
"""
import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


@contextmanager
def observe_absolute(env, tracker):
    scene = env.scene
    class Observer:
        def __getattr__(self, name): return getattr(scene, name)
        def step(self):
            scene.step()
            tracker.update([float(c.get_pose().p[2]) for c in env.covers])
    env.scene = Observer()
    try: yield
    finally: env.scene = scene


def visible_blocks(env):
    """Raw actor IDs avoid palette truncation; never feed these to the policy."""
    import numpy as np
    try:
        ids = [int(block.actor.per_scene_id) for block in env.blocks]
        cameras = dict(zip(env.cameras.static_camera_name, env.cameras.static_camera_list))
        cameras.update(left_camera=env.cameras.left_camera, right_camera=env.cameras.right_camera)
        counts = {}
        for name in ('head_camera', 'left_camera', 'right_camera'):
            segmentation = cameras[name].get_picture('Segmentation')[..., 1]
            counts[name] = [int(np.count_nonzero(segmentation == actor)) for actor in ids]
        return dict(status='measured', actor_ids=ids, pixels=counts,
                    all_hidden=all(n == 0 for row in counts.values() for n in row))
    except Exception as exc:
        return dict(status='unavailable', error=repr(exc), all_hidden=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    job = json.loads(parser.parse_args().job.read_text())
    from recoverability.sim_runtime import configure, make_env, close_env, public_observation
    configure(ROOT, str(job['gpu']))
    sys.path.insert(0, str(ROOT / 'third_party/EventVLA/EventVLA'))
    import numpy as np
    from PIL import Image
    from recoverability.snapshot import physics_vector, digest
    from recoverability.fixed_skills import deny_oracle_actions
    from recoverability.native_boundary import assess_parent, AbsoluteLiftTracker
    from recoverability.vla_intervention import audit_payload
    from resume_native_controls import atomic_json
    interface = ROOT / 'third_party/EventVLA/EventVLA/examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py'
    spec = importlib.util.spec_from_file_location('boundary_client', interface)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = Path(job['output']); out.mkdir()
    audits = []; memory_check = None; forbidden = ()
    class Client(module.ModelClient):
        def _call_server_with_reconnect(self, name, *args):
            if name == 'predict_action':
                step = int(args[0]['examples'][0]['timestep'])
                args[0]['paired_seed'] = 91000 + step
                example = args[0]['examples'][0]
                audit = dict(audit_payload(args[0], forbidden), step=step,
                             language=example.get('lang'), paired_seed=91000 + step,
                             event_steps=example.get('memory_keyframe_steps', []),
                             event_metas=example.get('memory_keyframe_image_metas', []))
                if not audits and memory_check is not None:
                    anchors = [r for r in audit['images'] if r['channel'] == 'image']
                    events = [r['sha256'] for r in audit['images'] if r['channel'] == 'memory_keyframe_images']
                    if (anchors != memory_check['anchors'] or events != memory_check['events']
                            or audit['event_metas'] != memory_check['metas']
                            or audit['event_steps'] != memory_check['steps']
                            or step != memory_check['step'] or audit['language'] != memory_check['language']):
                        raise RuntimeError('First live input differs from preregistered intervention')
                    audit['intervention_verified'] = True
                audits.append(audit)
            result = super()._call_server_with_reconnect(name, *args)
            self._raise_for_server_error(result)
            if name == 'predict_action' and not result['data'].get('server_memory_reset'):
                raise RuntimeError('Unconfirmed server cache reset')
            return result
    report = dict(status='running', job=job, formal_training_eligible=False, request_audits=audits)
    def save(): atomic_json(out / 'result.json', report)
    env = model = None
    actions = []; start = time.monotonic()
    try:
        env = make_env(ROOT, 'cover_blocks_hard', job['seed'])
        initial = np.array([c.get_pose().p for c in env.covers])
        source = Path(job['source']); parent = json.loads((source / 'parent.json').read_text())
        native = parent['native_instruction']
        model = Client(str(ROOT / 'checkpoints/runtime/EventVLA-RoboTwin-MeM/final_model/pytorch_model.pt'),
                       unnorm_key='robotwin_mem', port=job['port'])
        model.progress_output_dir = out / 'progress'; model.progress_output_dir.mkdir()
        model.keyframe_output_dir = out / 'keyframes'; model.keyframe_output_dir.mkdir()
        model.reset(native)
        views = ('head_camera', 'left_camera', 'right_camera')
        def obs(): return public_observation(env.get_obs())
        def write_views(current, stem):
            for name in views: Image.fromarray(current['images'][name]).save(out / f'{stem}_{name}.png')
        def predict(current, language):
            action = model.step(dict(lang=language, image=[current['images'][v] for v in views],
                                     state=current['proprioception']), step=env.take_action_cnt)
            if np.asarray(action).shape != (14,) or not np.isfinite(action).all():
                raise ValueError('Invalid action')
            actions.append(np.asarray(action).copy()); env.take_action(action)
        def geometry(settled):
            return assess_parent(positions=[c.get_pose().p for c in env.covers], initial_positions=initial,
                block_positions=[b.get_pose().p for b in env.blocks],
                grippers=[env.robot.get_left_gripper_val(), env.robot.get_right_gripper_val()],
                phase=env.phase, inspect_pointer=env.inspect_pointer, final_pointer=env.final_pointer,
                failed=bool(env.fail_flag), step=env.take_action_cnt, plan_start=model.plan_start_step,
                plan_length=0 if model.raw_actions is None else len(model.raw_actions), settled_steps=settled)
        with deny_oracle_actions(env) as oracle:
            with np.load(source / 'prefix_actions.npz', allow_pickle=False) as data:
                prefix = data['actions'].copy()
            for i, action in enumerate(prefix):
                env.get_obs(); env.take_action(action)
                if i % 100 == 0:
                    report.update(stage='prefix_replay', replay_actions=i+1); save()
            current = obs()
            checks = dict(physics=digest(physics_vector(env)) == parent['physics_hash'],
                          observation=digest(current) == parent['observation_hash'])
            if not all(checks.values()): raise RuntimeError('Parent replay mismatch: ' + repr(checks))
            # Trusted snapshots created by this project; do not load external pickles.
            state = pickle.loads((source / 'client_state.pkl').read_bytes())
            excluded = {'client', 'port', 'host', 'progress_output_dir', 'keyframe_output_dir',
                        'current_episode_keyframe_dir', 'progress_records'}
            model.__dict__.update({k: v for k, v in state.items() if k not in excluded})
            history_hash = digest(list(model.image_history)); events_hash = digest(model.keyframe_image_memory)
            checks['history'] = history_hash == parent['history_hash']
            checks['events'] = events_hash == parent['event_hash']
            if not checks['events'] and job['condition'] == 'prepare':
                from native_snapshot_audit import verify_event_snapshot
                audit = verify_event_snapshot(source / 'client_state.pkl', original_hash=parent['event_hash'], restored_hash=events_hash)
                report['legacy_snapshot_audit'] = audit
                checks['events'] = audit['verified']
            if not all(checks.values()): raise RuntimeError('Client snapshot mismatch: ' + repr(checks))
            report.update(parent_match=checks, source_parent=parent)
            if job['condition'] == 'prepare':
                trace = []; settled = 0; accepted = False
                for i in range(job.get('max_extension', 220)):
                    predict(current, native)
                    g = geometry(settled)
                    stable = all(g['checks'][k] for k in ['covers_down', 'covers_over_blocks', 'grippers_open'])
                    settled = settled + 1 if stable else 0
                    g = geometry(settled); trace.append(g)
                    report.update(stage='finish_native_chunk', extension_actions=len(actions), boundary=g)
                    if i % 10 == 0: save()
                    if g['eligible']:
                        current = obs(); visibility = visible_blocks(env)
                        newparent = dict(step=env.take_action_cnt, target_slot=int(env.color_order_slots[0]),
                            physics_hash=digest(physics_vector(env)), observation_hash=digest(current),
                            history_hash=digest(list(model.image_history)), event_hash=digest(model.keyframe_image_memory),
                            event_steps=[e['step'] for e in model.keyframe_image_memory],
                            temporal_steps=[e['step'] for e in model.image_history], native_instruction=native,
                            seed=job['seed'], initial_cover_positions=initial.tolist(), boundary=g, visibility=visibility,
                            source_parent_step=parent['step'], policy_model_requests=len(audits))
                        # Protocol 5 preserves ndarray writeability in the new snapshots.
                        (out / 'client_state.pkl').write_bytes(pickle.dumps({k: v for k, v in model.__dict__.items() if k != 'client'}, protocol=5))
                        np.savez_compressed(out / 'prefix_actions.npz', actions=np.concatenate([prefix, np.asarray(actions)]))
                        atomic_json(out / 'parent.json', newparent); write_views(current, 'parent')
                        report.update(parent=newparent, parent_saved=True); accepted = True; break
                    if env.fail_flag or env.final_pointer > 0: break
                    current = obs()
                atomic_json(out / 'boundary_trace.json', trace)
                report['parent_saved'] = accepted
            else:
                g = geometry(3)
                if not g['eligible']: raise RuntimeError('Unsafe prepared branch parent')
                visibility = visible_blocks(env)
                target = parent['target_slot']
                language = native
                if job['condition'] == 'position_other':
                    target = (target + 1) % 4
                    language = f'Finally open the {env.cover_name[target]} cover.'
                # Language-only intervention; no deletion or cache purge at this natural boundary.
                model.task_description = language
                report.update(target_slot=target, instruction=language, boundary=g, visibility=visibility,
                    history_before_intervention=history_hash, events_before_intervention=events_hash,
                    native_target_slot=parent['target_slot'])
                if job['condition'] in ('keep', 'drop_all', 'mismatch'):
                    from recoverability.native_memory import intervene
                    if not visibility['all_hidden']:
                        raise RuntimeError('Current camera reveals a block')
                    donor_entries = None
                    donor_source = Path(job['donor_source'])
                    donor_parent = json.loads((donor_source / 'parent.json').read_text())
                    if donor_parent['seed'] == parent['seed'] or donor_parent['target_slot'] == target:
                        raise ValueError('Donor must have a different red target')
                    if job['condition'] == 'mismatch':
                        donor_state = pickle.loads((donor_source / 'client_state.pkl').read_bytes())
                        donor_entries = donor_state['keyframe_image_memory']
                        if digest(donor_entries) != donor_parent['event_hash']:
                            raise RuntimeError('Donor snapshot mismatch')
                        if max(e['step'] for e in donor_entries) >= parent['step']:
                            raise ValueError('Donor event lies after recipient branch time')
                    intervention = intervene(model, job['condition'], donor_entries)
                    if digest(list(model.image_history)) != history_hash:
                        raise RuntimeError('Memory intervention changed temporal history')
                    baseline = json.loads((source / 'input_audit/report.json').read_text())
                    memory_check = dict(anchors=[r for r in baseline['audit']['images'] if r['channel'] == 'image'],
                        events=intervention['expected_event_image_hashes'],
                        metas=[] if job['condition'] == 'drop_all' else baseline['event_metas'],
                        steps=[] if job['condition'] == 'drop_all' else parent['event_steps'],
                        step=parent['step'], language=native)
                    forbidden = intervention['forbidden_image_hashes']
                    report.update(intervention=intervention, donor_seed=donor_parent['seed'],
                        donor_target_slot=donor_parent['target_slot'], temporal_history_unchanged=True,
                        stage='memory_intervention_applied')
                    save()
                tracker = AbsoluteLiftTracker(initial[:, 2]); transitions = []; revealed = []
                previous = (env.phase, env.inspect_pointer, env.final_pointer, bool(env.fail_flag))
                with observe_absolute(env, tracker):
                    for i in range(job['max_actions']):
                        predict(current, language)
                        now = (env.phase, env.inspect_pointer, env.final_pointer, bool(env.fail_flag))
                        if now != previous:
                            transitions.append(dict(action=i+1, state=now)); previous = now
                        opened = [s for s in range(4) if env._is_slot_open_for_progress(s)]
                        if opened and not revealed: revealed.append(dict(action=i+1, slots=opened))
                        report.update(stage='continuation', actions_executed=len(actions), lift_events=tracker.events,
                                      task_transitions=transitions, revealed_events=revealed)
                        if i % 25 == 0: save()
                        if revealed or env.fail_flag or (tracker.events and not tracker.first_target(target)): break
                        current = obs()
                report.update(first_lift_correct=tracker.first_target(target),
                    first_reveal_correct=bool(revealed and revealed[0]['slots'] == [target]),
                    native_first_target_opened=env.final_pointer > 0, native_failed=bool(env.fail_flag),
                    action_hash=digest(np.asarray(actions)), lift_events=tracker.events,
                    revealed_events=revealed, task_transitions=transitions)
                if memory_check is not None:
                    report['first_lift_matches_donor'] = tracker.first_target(report['donor_target_slot'])
            report['forbidden_calls'] = oracle['forbidden_calls']
        np.savez_compressed(out / 'executed_actions.npz', actions=np.asarray(actions))
        write_views(obs(), 'final')
        report.update(status='complete', actions_executed=len(actions), elapsed_seconds=time.monotonic()-start,
                      final_physics_hash=digest(physics_vector(env)), action_hash=digest(np.asarray(actions)))
        save()
    except BaseException:
        report.update(status='failed', error=traceback.format_exc()); save(); raise
    finally:
        if model is not None and model.client is not None: model.client.close()
        if env is not None: close_env(env)


if __name__ == '__main__': main()
