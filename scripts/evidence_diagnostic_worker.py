"""Native rollouts and explicit same-parent evidence replacements, no experts.

Evaluator state is written separately and never passed to ModelClient. A saved
parent is before a naturally exhausted action chunk, with no pending write.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import faulthandler
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from recoverability.evidence_diagnostic import candidate_ready, inference_seed, validate_replacements
from resume_native_controls import atomic_json

VIEWS = ('head_camera', 'left_camera', 'right_camera')


@contextmanager
def forbid_experts(env):
    originals = {}
    audit = {'forbidden_calls': 0}
    for owner, names in [(env, ('play_once', 'grasp_actor', 'place_actor', 'get_grasp_pose',
                               'get_place_pose', 'get_keyframe_oracle_info')),
                         (env.robot, tuple(n for n in dir(env.robot) if n.startswith(('left_plan_', 'right_plan_'))))]:
        for name in names:
            if hasattr(owner, name):
                originals[(owner, name)] = getattr(owner, name)
    def forbidden(*args, **kwargs):
        audit['forbidden_calls'] += 1
        raise RuntimeError('Expert entry point is forbidden')
    for owner, name in originals:
        setattr(owner, name, forbidden)
    try:
        yield audit
    finally:
        for (owner, name), value in originals.items():
            setattr(owner, name, value)


def evaluator_state(env):
    fields = ('phase', 'fail_flag', 'press_cnt', 'press_flag', 'progress_index',
              'actual_first_visit_side_ids', 'first_center_return_achieved',
              'second_center_return_achieved', 'actual_first_block_id',
              'actual_other_block_id', 'layout_restored_achieved', 'other_center_achieved')
    result = {k:deepcopy(getattr(env, k)) for k in fields if hasattr(env, k)}
    result['both_on_center'] = all(bool(env._block_on_center(i)) for i in range(2))
    result['block_positions'] = [b.get_pose().p.tolist() for b in env.blocks]
    button = env.button.actor if hasattr(env.button, 'actor') else env.button
    result['button_qpos'] = button.get_qpos().tolist()
    result['button_qvel'] = button.get_qvel().tolist()
    # Normalize numpy scalars without importing them into controller inputs.
    return json.loads(json.dumps(result, default=lambda x:x.item()))


def main():
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    job = json.loads(parser.parse_args().job.read_text())
    from recoverability.sim_runtime import configure, make_env, close_env, public_observation
    configure(ROOT, str(job['gpu']))
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(2)
    from recoverability.snapshot import physics_vector, digest, rng_state, restore_rng
    from recoverability.vla_intervention import audit_payload, image_hash
    interface = ROOT/'third_party/EventVLA/EventVLA/examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py'
    sys.path.insert(0, str(ROOT/'third_party/EventVLA/EventVLA'))
    spec = importlib.util.spec_from_file_location('diagnostic_client', interface)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = Path(job['output'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'observations').mkdir()
    (out/'requests').mkdir()
    report = dict(status='running', job=job, actions_executed=0, request_audits=[],
                  candidate_saved=False, formal_training_eligible=False)
    actions, observed_steps = [], []
    env = model = None
    started = time.monotonic()
    def save():
        atomic_json(out/'result.json', report)
        if actions:
            temp = out/'actions.tmp.npz'
            np.savez_compressed(temp, actions=np.asarray(actions))
            temp.replace(out/'executed_actions.npz')
    class Client(module.ModelClient):
        request_index = 0
        def _call_server_with_reconnect(self, name, *args):
            if name == 'predict_action':
                args[0]['paired_seed'] = inference_seed(job['seed'], self.request_index)
                record = audit_payload(args[0], ())
                record.update(request_index=self.request_index,
                              step=int(args[0]['examples'][0]['timestep']),
                              paired_seed=args[0]['paired_seed'])
                # Trusted local payload archive permits exact subsequent audits.
                (out/'requests'/f'{self.request_index:04d}.pkl').write_bytes(pickle.dumps(args[0], protocol=5))
                report['request_audits'].append(record)
                save()
                response = super()._call_server_with_reconnect(name, *args)
                self._raise_for_server_error(response)
                if not response['data'].get('server_memory_reset'):
                    raise RuntimeError('Unconfirmed server memory reset')
                self.request_index += 1
                return response
            return super()._call_server_with_reconnect(name, *args)
    try:
        env = make_env(ROOT, job['task'], job['seed'])
        native = json.loads((ROOT/f"third_party/EventVLA/RoboTwin-Mem/description/task_instruction/{job['task']}.json").read_text())['seen'][0]
        if '{' in native or '}' in native:
            raise ValueError('Unfilled native instruction')
        model = Client(str(ROOT/'checkpoints/runtime/EventVLA-RoboTwin-MeM/final_model/pytorch_model.pt'),
                       unnorm_key='robotwin_mem', port=job['port'])
        model.progress_output_dir = out/'progress'
        model.keyframe_output_dir = out/'keyframes'
        model.progress_output_dir.mkdir()
        model.keyframe_output_dir.mkdir()
        model.reset(native)
        report.update(instruction=native, native_step_limit=int(env.step_lim), stage='initialized')
        save()
        with forbid_experts(env) as expert_audit:
            source = Path(job['source']) if job.get('source') else None
            if source:
                parent = json.loads((source/'parent.json').read_text())
                if parent['task'] != job['task'] or parent['seed'] != job['seed']:
                    raise ValueError('Parent task/seed mismatch')
                prefix = np.load(source/'prefix_actions.npz', allow_pickle=False)['actions']
                for i, action in enumerate(prefix):
                    env.get_obs()
                    env.take_action(action)
                    if i % 25 == 0:
                        report.update(stage='prefix_replay', replay_actions=i+1)
                        save()
                current = public_observation(env.get_obs())
                matches = dict(physics=digest(physics_vector(env)) == parent['physics_hash'],
                               observation=digest(current) == parent['observation_hash'],
                               evaluator_state=evaluator_state(env) == parent['evaluator_state'])
                report['parent_match'] = matches
                if not all(matches.values()):
                    raise RuntimeError('Parent replay mismatch')
                state = pickle.loads((source/'client_state.pkl').read_bytes())
                excluded = {'client', 'port', 'host', 'progress_output_dir', 'keyframe_output_dir',
                            'current_episode_keyframe_dir', 'progress_records'}
                model.__dict__.update({k:v for k,v in state.items() if k not in excluded})
                replacements = job.get('replacements', [])
                validate_replacements(replacements, len(model.keyframe_image_memory), parent['step'],
                                      set(parent['observed_steps']))
                report['before_events_hash'] = digest(model.keyframe_image_memory)
                for replacement in replacements:
                    entry = model.keyframe_image_memory[replacement['slot']]
                    raw = np.asarray(Image.open(source/'observations'/f"{replacement['source_step']:06d}"/'head_camera.png'))
                    # Keep slot, timestamp, confidence and all anchor inputs fixed.
                    # This is a content-only diagnostic, not a deployed retriever.
                    entry['images'] = [model._resize_image(raw)]
                report['after_events_hash'] = digest(model.keyframe_image_memory)
                restore_rng(pickle.loads((source/'rng.pkl').read_bytes()))
                report.update(stage='paired_continuation', parent_step=parent['step'],
                              replacement_semantics='content_only_fixed_slot_metadata',
                              inherited_request_index=model.request_index)
            prior_trace = None
            trace = (out/'evaluator_trace.jsonl').open('w', buffering=1)
            try:
                while env.take_action_cnt < env.step_lim and not env.eval_success and not getattr(env, 'fail_flag', False):
                    t = int(env.take_action_cnt)
                    obs = public_observation(env.get_obs())
                    observed_steps.append(t)
                    folder = out/'observations'/f'{t:06d}'
                    folder.mkdir()
                    for view in VIEWS:
                        Image.fromarray(obs['images'][view]).save(folder/f'{view}.png', compress_level=1)
                    np.save(folder/'proprioception.npy', obs['proprioception'], allow_pickle=False)
                    state = evaluator_state(env)
                    trace.write(json.dumps(dict(step=t, state=state))+'\n')
                    # A natural end-of-chunk boundary, never forcibly replan the native rollout.
                    chunk_done = (model.raw_actions is not None and model.plan_start_step is not None and
                                  t-model.plan_start_step >= len(model.raw_actions))
                    if (source is None and not report['candidate_saved'] and
                        candidate_ready(job['task'], state, chunk_done, model.pending_commit_done)):
                        parent = dict(task=job['task'], seed=job['seed'], step=t,
                            physics_hash=digest(physics_vector(env)), observation_hash=digest(obs),
                            evaluator_state=state, event_steps=[int(e['step']) for e in model.keyframe_image_memory],
                            event_count=len(model.keyframe_image_memory), observed_steps=observed_steps[:-1],
                            request_index=model.request_index, native_instruction=native)
                        atomic_json(out/'parent.json', parent)
                        np.savez_compressed(out/'prefix_actions.npz', actions=np.asarray(actions))
                        (out/'client_state.pkl').write_bytes(pickle.dumps({k:v for k,v in model.__dict__.items() if k != 'client'}, protocol=5))
                        (out/'rng.pkl').write_bytes(pickle.dumps(rng_state(), protocol=5))
                        report.update(candidate_saved=True, candidate_step=t)
                        save()
                    action = model.step(dict(lang=native, image=[obs['images'][v] for v in VIEWS],
                                             state=obs['proprioception']), step=t)
                    if np.asarray(action).shape != (14,) or not np.isfinite(action).all():
                        raise RuntimeError('Invalid action')
                    actions.append(np.asarray(action).copy())
                    env.take_action(action)
                    report.update(stage='paired_continuation' if source else 'native_rollout',
                                  actions_executed=len(actions), env_step=int(env.take_action_cnt),
                                  latest_evaluator_state=evaluator_state(env))
                    if len(actions) % 10 == 0:
                        save()
            finally:
                trace.close()
            final = evaluator_state(env)
            report.update(native_success=bool(env.eval_success), native_failed=bool(getattr(env,'fail_flag',False)),
                          final_evaluator_state=final, forbidden_calls=expert_audit['forbidden_calls'],
                          action_hash=digest(np.asarray(actions)), final_physics_hash=digest(physics_vector(env)),
                          native_step_limit_reached=env.take_action_cnt >= env.step_lim)
            if job['task'] == 'put_back_block_hard':
                report['full_move_return_sequence_observed'] = bool(final['first_center_return_achieved'] and final['second_center_return_achieved'])
                report['native_success_without_both_returns'] = bool(env.eval_success and not report['full_move_return_sequence_observed'])
            Image.fromarray(public_observation(env.get_obs())['images']['head_camera']).save(out/'final.png')
        report.update(status='complete', elapsed_seconds=time.monotonic()-started)
        save()
    except BaseException:
        report.update(status='failed', error=traceback.format_exc(), elapsed_seconds=time.monotonic()-started)
        save()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if model is not None and model.client is not None:
            model.client.close()
        if env is not None:
            close_env(env)


if __name__ == '__main__':
    main()
