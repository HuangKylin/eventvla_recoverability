"""Validate completed native-history controls without assigning formal K/R labels."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_snapshot_audit import verify_event_snapshot

ROOT = Path(__file__).resolve().parents[1]


def summarize(status):
    state = json.loads(Path(status).read_text())
    results = [json.loads(Path(e['result_file']).read_text()) for e in state['episodes']]
    rows = []
    seen = set()
    for result in results:
        job = result['job']
        key = (job['seed'], job['condition'])
        if result['status'] != 'complete':
            raise ValueError(f'Unfinished result cannot be counted: {key}')
        if key in seen:
            raise ValueError(f'Duplicate condition cannot be counted twice: {key}')
        seen.add(key)
        row = dict(seed=job['seed'], condition=job['condition'], status=result['status'],
                   actions=result['actions_executed'], requests=len(result['request_audits']),
                   forbidden_calls=result['forbidden_calls'])
        if job['condition'] == 'rollout':
            row.update(success=result['native_success'], parent_saved=result['parent_saved'],
                       phases=result['phases'], parent=result.get('parent'))
        else:
            parent = result['parent']
            checks = dict(result['parent_match'])
            events_match = result['events_before_intervention'] == parent['event_hash']
            if not events_match and job.get('source'):
                source = Path(job['source']).resolve()
                if not source.is_relative_to((ROOT / 'runs').resolve()):
                    raise ValueError('Refusing to unpickle a snapshot outside this project run directory')
                audit = verify_event_snapshot(source / 'client_state.pkl', original_hash=parent['event_hash'],
                                              restored_hash=result['events_before_intervention'])
                row['event_serialization_audit'] = audit
                events_match = audit['verified']
            checks.update(history=result['history_before_intervention'] == parent['history_hash'],
                          events=events_match)
            if not all(checks.values()):
                raise ValueError(f'Parent identity mismatch: {job}: {checks}')
            row.update(success=result['intended_cover_actuated'], parent_checks=checks,
                       target_slot=result['target_slot'], events=result['events'],
                       action_hash=result['action_hash'], final_physics_hash=result['final_physics_hash'])
        rows.append(row)
    repeated = []
    for row in rows:
        if row['condition'] != 'native_repeat': continue
        first = next(r for r in rows if r['seed'] == row['seed'] and r['condition'] == 'native')
        repeated.append(dict(seed=row['seed'], identical=all(first[k] == row[k] for k in
                        ('action_hash', 'final_physics_hash', 'events', 'actions'))))
    counts = {}
    for row in rows:
        group = counts.setdefault(row['condition'], dict(successes=0, episodes=0))
        group['successes'] += int(row['success'])
        group['episodes'] += 1
    expected = {(seed, condition) for seed in state.get('seeds', [100000, 100001])
                for condition in ['rollout'] + state.get('conditions', ['native', 'position', 'color', 'native_repeat'])}
    missing = sorted(expected - seen)
    return dict(status=state['status'], source_status=str(status), formal_training_eligible=False,
                counts_are_raw_diagnostic_metric=True, closed_parent_boundary_verified=False,
                requested_suite_complete=state['status'] == 'complete' and not missing,
                missing_conditions=[dict(seed=seed, condition=condition) for seed, condition in missing],
                rows=rows, counts=counts, repeated_trajectories=repeated,
                all_oracle_tripwires_clean=all(r['forbidden_calls'] == 0 for r in rows),
                paired_parent_checks_passed=bool(any('parent_checks' in r for r in rows)) and
                    all(all(r.get('parent_checks', {}).values()) for r in rows),
                caveats=['Small diagnostic set; not a benchmark score or formal K/R dataset.',
                         'Native rollout success and first-cover actuation measure different outcomes.',
                         'Server resets memory and RNG on each inference in every condition.',
                         'Language interventions preserve native visual history explicitly.',
                         'Lift is measured from the parent height; already raised parent covers can produce false negatives. Closed-parent evaluation is not yet validated.',
                         'No new duplicate colors, latch mutation, or manual prefix in this control.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('status', type=Path)
    args = parser.parse_args()
    report = summarize(args.status)
    target = ROOT / 'reports/native_control_validation.json'
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['counts'], indent=2))
