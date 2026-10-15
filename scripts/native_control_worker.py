"""Native-history positive control and matched-parent instruction interventions.

No expert prefix, synthetic timestamps, manual keyframes, or latch mutation.
Environment truth is used only for evaluation and a public positional request.
Pickles are trusted, locally generated client snapshots, never external inputs.
"""
import argparse
from contextlib import contextmanager
import faulthandler
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
def observe_lifts(env, events):
    """Evaluator-only observer; it does not alter task physics or appearance."""
    import numpy as np
    scene = env.scene
    origins = [float(c.get_pose().p[2]) for c in env.covers]
    tick = 0
    class ObservedScene:
        def __getattr__(self, name):
            return getattr(scene, name)
        def step(self):
            nonlocal tick
            scene.step()
            tick += 1
            for slot, cover in enumerate(env.covers):
                lift = float(cover.get_pose().p[2]) - origins[slot]
                if lift > .04 and slot not in {e['slot'] for e in events}:
                    events.append(dict(slot=slot, tick=tick, lift_m=lift))
    env.scene = ObservedScene()
    try:
        yield
    finally:
        env.scene = scene


def main():
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
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
    from recoverability.vla_intervention import purge_evidence, audit_payload
    interface = ROOT / 'third_party/EventVLA/EventVLA/examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py'
    spec = importlib.util.spec_from_file_location('native_client', interface)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = Path(job['output'])
    out.mkdir(parents=True)
    audits = []
    class Client(module.ModelClient):
        def _call_server_with_reconnect(self, name, *args):
            if name == 'predict_action':
                args[0]['paired_seed'] = 91000 + int(args[0]['examples'][0]['timestep'])
                record = audit_payload(args[0], ())
                record['step'] = int(args[0]['examples'][0]['timestep'])
                audits.append(record)
            result = super()._call_server_with_reconnect(name, *args)
            self._raise_for_server_error(result)
            if name == 'predict_action' and not result['data'].get('server_memory_reset'):
                raise RuntimeError('Server cache reset not confirmed')
            return result
    report = dict(status='running', job=job, formal_training_eligible=False,
                  request_audits=audits, actions_executed=0)
    def save():
        (out / 'result.json').write_text(json.dumps(report, indent=2))
    env = model = None
    start = time.monotonic()
    try:
        env = make_env(ROOT, 'cover_blocks_hard', job['seed'])
        native = json.loads((ROOT / 'third_party/EventVLA/RoboTwin-Mem/description/task_instruction/cover_blocks_hard.json').read_text())['seen'][0]
        model = Client(str(ROOT / 'checkpoints/runtime/EventVLA-RoboTwin-MeM/final_model/pytorch_model.pt'),
                       unnorm_key='robotwin_mem', port=job['port'])
        model.progress_output_dir = out / 'progress'
        model.keyframe_output_dir = out / 'keyframes'
        model.progress_output_dir.mkdir()
        model.keyframe_output_dir.mkdir()
        model.reset(native)
        views = ('head_camera', 'left_camera', 'right_camera')
        actions = []
        phases = []
        instruction = native
        source = Path(job['source']) if job.get('source') else None
        with deny_oracle_actions(env) as audit:
            if source:
                parent = json.loads((source / 'parent.json').read_text())
                with np.load(source / 'prefix_actions.npz', allow_pickle=False) as data:
                    for index, action in enumerate(data['actions']):
                        env.get_obs()
                        env.take_action(action)
                        if index % 100 == 0:
                            report.update(stage='replay', replay_actions=index + 1)
                            save()
                current = public_observation(env.get_obs())
                hashes = dict(physics_hash=digest(physics_vector(env)), observation_hash=digest(current))
                report['parent_match'] = {key: value == parent[key] for key, value in hashes.items()}
                if not all(report['parent_match'].values()):
                    raise RuntimeError('Fresh-process replay did not match native parent')
                # Snapshot created by this experiment; no simulator scene is unpickled.
                state = pickle.loads((source / 'client_state.pkl').read_bytes())
                excluded = {'client', 'port', 'host', 'progress_output_dir', 'keyframe_output_dir',
                            'current_episode_keyframe_dir', 'progress_records'}
                model.__dict__.update({k: v for k, v in state.items() if k not in excluded})
                report['history_before_intervention'] = digest(list(model.image_history))
                report['events_before_intervention'] = digest(model.keyframe_image_memory)
                report['cache_intervention'] = purge_evidence(model, [])
                target = parent['target_slot']
                if job['condition'] == 'position':
                    instruction = f'Open the {env.cover_name[target]} cover to inspect the block color, then place the cover back.'
                elif job['condition'] == 'color':
                    instruction = 'Finally open the cover hiding the red block.'
                # Deliberately preserve the SAME history when changing language:
                # ModelClient.step otherwise resets all memory on a new string.
                model.task_description = instruction
                report.update(parent=parent, target_slot=target, instruction=instruction)
                events = []
                with observe_lifts(env, events):
                    for i in range(job['max_actions']):
                        obs = public_observation(env.get_obs())
                        action = model.step(dict(lang=instruction, image=[obs['images'][v] for v in views],
                                                 state=obs['proprioception']), step=env.take_action_cnt)
                        if np.asarray(action).shape != (14,) or not np.isfinite(action).all():
                            raise RuntimeError('Invalid model action')
                        actions.append(np.asarray(action).copy())
                        env.take_action(action)
                        report.update(stage='continuation', actions_executed=len(actions), events=events)
                        if i % 25 == 0: save()
                        if events or env.eval_success: break
                report.update(intended_cover_actuated=bool(events and events[0]['slot'] == target),
                              events=events, action_hash=digest(np.asarray(actions)))
            else:
                parent_saved = False
                for i in range(env.step_lim):
                    obs = public_observation(env.get_obs())
                    action = model.step(dict(lang=native, image=[obs['images'][v] for v in views],
                                             state=obs['proprioception']), step=env.take_action_cnt)
                    if np.asarray(action).shape != (14,) or not np.isfinite(action).all():
                        raise RuntimeError('Invalid model action')
                    actions.append(np.asarray(action).copy())
                    env.take_action(action)
                    phase = (env.phase, env.inspect_pointer, env.final_pointer, bool(env.fail_flag))
                    if not phases or tuple(phases[-1]['state']) != phase:
                        phases.append(dict(step=env.take_action_cnt, state=phase))
                    if not parent_saved and env.phase == 'final' and not env.fail_flag and env.final_pointer == 0:
                        current = public_observation(env.get_obs())
                        parent = dict(step=env.take_action_cnt, target_slot=int(env.color_order_slots[0]),
                                      physics_hash=digest(physics_vector(env)), observation_hash=digest(current),
                                      history_hash=digest(list(model.image_history)),
                                      event_hash=digest(model.keyframe_image_memory),
                                      event_steps=[e['step'] for e in model.keyframe_image_memory],
                                      native_instruction=native, seed=job['seed'])
                        (out / 'parent.json').write_text(json.dumps(parent, indent=2))
                        np.savez_compressed(out / 'prefix_actions.npz', actions=np.asarray(actions))
                        (out / 'client_state.pkl').write_bytes(pickle.dumps({k: v for k, v in model.__dict__.items() if k != 'client'}))
                        Image.fromarray(current['images']['head_camera']).save(out / 'parent.png')
                        parent_saved = True
                        report['parent'] = parent
                    report.update(stage='native_rollout', actions_executed=len(actions), phases=phases,
                                  native_success=bool(env.eval_success), parent_saved=parent_saved)
                    if i % 25 == 0: save()
                    if env.eval_success or env.fail_flag: break
                report.update(native_success=bool(env.eval_success), phases=phases, parent_saved=parent_saved)
            report['forbidden_calls'] = audit['forbidden_calls']
        np.savez_compressed(out / 'executed_actions.npz', actions=np.asarray(actions))
        Image.fromarray(public_observation(env.get_obs())['images']['head_camera']).save(out / 'final.png')
        report.update(status='complete', elapsed_seconds=time.monotonic() - start,
                      final_physics_hash=digest(physics_vector(env)), action_hash=digest(np.asarray(actions)))
        save()
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        save()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if model is not None and model.client is not None: model.client.close()
        if env is not None: close_env(env)


if __name__ == '__main__':
    main()
