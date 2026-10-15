"""Reject invalid paired evidence even when the simulator reports success."""
import importlib.util
import json
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/summarize_native_controls.py'
spec = importlib.util.spec_from_file_location('native_summary', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_case(tmp_path, *, history='history', observation=True, repeat_hash='same'):
    episodes = []
    for condition, action_hash in [('native', 'same'), ('native_repeat', repeat_hash)]:
        result = dict(job=dict(seed=100000, condition=condition), status='complete',
            actions_executed=50, request_audits=[{}], forbidden_calls=0,
            parent=dict(history_hash='history', event_hash='events'),
            parent_match=dict(physics_hash=True, observation_hash=observation),
            history_before_intervention=history, events_before_intervention='events',
            intended_cover_actuated=True, target_slot=2, events=[dict(slot=2)],
            action_hash=action_hash, final_physics_hash='physics')
        file = tmp_path / (condition + '.json')
        file.write_text(json.dumps(result))
        episodes.append(dict(result_file=str(file)))
    status = tmp_path / 'status.json'
    status.write_text(json.dumps(dict(status='complete', episodes=episodes)))
    return status


def test_changed_history_cannot_be_accepted_as_a_language_only_pair(tmp_path):
    with pytest.raises(ValueError, match='Parent identity mismatch'):
        module.summarize(write_case(tmp_path, history='different'))


def test_same_physics_but_different_pixels_rejects_pair(tmp_path):
    with pytest.raises(ValueError, match='Parent identity mismatch'):
        module.summarize(write_case(tmp_path, observation=False))


def test_same_success_does_not_prove_repeated_action_trajectory(tmp_path):
    result = module.summarize(write_case(tmp_path, repeat_hash='different'))
    assert result['counts']['native_repeat']['successes'] == 1
    assert not result['repeated_trajectories'][0]['identical']
    assert result['formal_training_eligible'] is False


def test_interrupted_result_cannot_be_counted_even_if_success_field_exists(tmp_path):
    status = write_case(tmp_path)
    file = tmp_path / 'native.json'
    result = json.loads(file.read_text())
    result['status'] = 'running'
    file.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='Unfinished result'):
        module.summarize(status)


def test_reused_result_cannot_inflate_sample_count(tmp_path):
    status = write_case(tmp_path)
    state = json.loads(status.read_text())
    state['episodes'].append(state['episodes'][0])
    status.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='Duplicate condition'):
        module.summarize(status)


def test_completed_runner_with_missing_conditions_is_not_completed_suite(tmp_path):
    report = module.summarize(write_case(tmp_path))
    assert report['status'] == 'complete'
    assert report['requested_suite_complete'] is False
    assert dict(seed=100001, condition='position') in report['missing_conditions']
