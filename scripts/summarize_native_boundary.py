"""Validate paired controls without treating unsuccessful policies as missing data."""
import argparse
import json
from pathlib import Path


def summarize(status_path):
    status_path = Path(status_path)
    state = json.loads(status_path.read_text())
    if state['status'] != 'complete': raise ValueError('Only complete runs can be summarized')
    records = {}
    for row in state['episodes']:
        key = (row['seed'], row['condition'])
        if key in records: raise ValueError('Duplicate episode')
        path = Path(row['result_file'])
        if not path.exists(): path = status_path.parent / f'{key[0]}_{key[1]}' / 'result.json'
        result = json.loads(path.read_text())
        if result['status'] != 'complete' or result['forbidden_calls'] != 0:
            raise ValueError('Incomplete result or forbidden expert call')
        if not all(result['parent_match'].values()): raise ValueError('Parent mismatch')
        records[key] = result
    rows = []; paired = []; skipped = []
    for seed in state['seeds']:
        prepare = records[(seed, 'prepare')]
        if not prepare['parent_saved']:
            skipped.append(dict(seed=seed, reason='No eligible natural boundary'))
            continue
        parent = prepare['parent']
        group = [records[(seed, c)] for c in ['native','native_repeat','position_other']]
        native, repeat, other = group
        checks = dict(natural_boundary=parent['boundary']['eligible'],
            identical_actions=native['action_hash'] == repeat['action_hash'],
            identical_final_physics=native['final_physics_hash'] == repeat['final_physics_hash'],
            target_changed=other['target_slot'] != native['target_slot'],
            same_first_payload=all(v['request_audits'][0]['images'] == native['request_audits'][0]['images'] for v in group),
            first_request_at_parent=all(v['request_audits'][0]['step'] == parent['step'] for v in group),
            same_history=all(v['history_before_intervention'] == parent['history_hash'] for v in group),
            same_events=all(v['events_before_intervention'] == parent['event_hash'] for v in group),
            instructions_applied=all(v['request_audits'][0]['language'] == v['instruction'] for v in group))
        if not all(checks.values()): raise ValueError('Pair validation failed: '+repr(checks))
        paired.append(dict(seed=seed, checks=checks, parent_step=parent['step'],
                           source_parent_step=parent['source_parent_step'], visibility=parent['visibility']))
        for condition, result in zip(['native','native_repeat','position_other'], group):
            rows.append(dict(seed=seed, condition=condition, parent_step=parent['step'],
                target_slot=result['target_slot'], native_target_slot=result['native_target_slot'],
                first_lift_correct=result['first_lift_correct'], first_reveal_correct=result['first_reveal_correct'],
                native_first_target_opened=result['native_first_target_opened'], native_failed=result['native_failed'],
                actions=result['actions_executed'], model_requests=len(result['request_audits']),
                lift_events=result['lift_events'], revealed_events=result['revealed_events']))
    return dict(status='validated', source_status=str(status_path), pairs=paired, outcomes=rows, skipped=skipped,
                formal_training_eligible=False, deletion_or_recovery_performed=False,
                note='Unique-seed outcomes exclude repeat runs; position controls are not equivalent memory tasks.')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('status');p.add_argument('--output',type=Path)
    a=p.parse_args(); result=summarize(a.status)
    if a.output:a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
