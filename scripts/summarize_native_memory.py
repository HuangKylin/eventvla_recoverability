"""Independently audit paired inputs and summarize first-choice diagnostics."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('keep', 'drop_all', 'mismatch')


def artifact(path):
    p = Path(path)
    if p.exists(): return p
    normalized = str(path).replace('\\', '/')
    if '/runs/' in normalized:
        p = ROOT/'runs'/normalized.split('/runs/', 1)[1]
    if not p.exists(): raise FileNotFoundError(path)
    return p


def read(path): return json.loads(artifact(path).read_text())
def channel(audit, name): return [r for r in audit['images'] if r['channel'] == name]


def require(value, message):
    if not value: raise ValueError(message)


def summarize(state):
    require(state['status'] == 'complete', 'Run is incomplete')
    require(state['process_cleanup_completed'], 'Process cleanup not confirmed')
    resume_audit = None
    if state.get('resumed_from'):
        prior = read(artifact(state['resumed_from'])/'status.json')
        require(prior['status'] == 'failed' and prior.get('process_cleanup_completed'), 'Previous run was not cleanly stopped')
        shared = set(prior['source_hashes']) & set(state['source_hashes'])
        require(all(prior['source_hashes'][n] == state['source_hashes'][n] for n in shared),
                'Experiment source changed across retry')
        resume_audit = dict(previous_run=state['resumed_from'], unchanged_shared_sources=len(shared),
                            inherited_episode_count=state['inherited_episode_count'], infrastructure_interruptions=1)
    expected = {(s, c) for s in state['seeds'] for c in CONDITIONS}
    episodes = {(e['seed'], e['condition']): read(e['result_file']) for e in state['episodes']}
    result_files = {(e['seed'], e['condition']): e['result_file'] for e in state['episodes']}
    require(set(episodes) == expected and len(state['episodes']) == len(expected), 'Missing/duplicate conditions')
    if resume_audit is not None and prior.get('current_job'):
        attempt = artifact(prior['run_dir'])/Path(prior['current_job']).name.removesuffix('.job.json')/'result.json'
        partial = read(attempt); key = (partial['job']['seed'], partial['job']['condition'])
        saved = partial['request_audits']
        require(episodes[key]['request_audits'][:len(saved)] == saved, 'Saved retry inputs differ from interrupted attempt')
        resume_audit['saved_request_prefix_matches_retry'] = True
        resume_audit['saved_request_count'] = len(saved)
    rows = []; keep_reproductions = []; request_count = 0
    for seed in state['seeds']:
        source = artifact(state['parent_sources'][str(seed)])
        parent = read(source/'parent.json'); baseline = read(source/'input_audit/report.json')
        donor_seed = state['donor_seeds'][str(seed)]
        donor_source = artifact(state['parent_sources'][str(donor_seed)])
        donor_parent = read(donor_source/'parent.json'); donor = read(donor_source/'input_audit/report.json')
        previous = read(source.parent/f'{seed}_native/result.json')
        for condition in CONDITIONS:
            result = episodes[seed, condition]; audits = result['request_audits']
            require(result['status'] == 'complete' and not result['formal_training_eligible'], 'Invalid result type')
            require(all(result['parent_match'].values()), 'Parent mismatch')
            require(result['visibility']['all_hidden'] and result['boundary']['eligible'], 'Unsafe parent')
            require(result['forbidden_calls'] == 0, 'Oracle action called')
            require(result['source_parent'] == parent and result['target_slot'] == parent['target_slot'], 'Changed parent or target')
            require(result['instruction'] == parent['native_instruction'], 'Changed instruction')
            intervention = result['intervention']
            require(intervention['action_cache_cleared'] and intervention['pending_commit_cleared'], 'Cache not cleared')
            require(result['temporal_history_unchanged'], 'Changed temporal history')
            require(result['history_before_intervention'] == parent['history_hash'], 'History mismatch')
            require(result['events_before_intervention'] == parent['event_hash'], 'Original events mismatch')
            require(audits and audits[0]['step'] == parent['step'], 'First request not at parent boundary')
            first = audits[0]
            require(first.get('intervention_verified'), 'Missing live intervention audit')
            require(channel(first, 'image') == channel(baseline['audit'], 'image'), 'First temporal anchors changed')
            want_events = [] if condition == 'drop_all' else channel((donor if condition == 'mismatch' else baseline)['audit'], 'memory_keyframe_images')
            require(channel(first, 'memory_keyframe_images') == want_events, 'First event pixels differ from independent source')
            require(first['event_steps'] == ([] if condition == 'drop_all' else parent['event_steps']), 'First event timestamps changed')
            require(first['event_metas'] == ([] if condition == 'drop_all' else baseline['event_metas']), 'First metadata changed')
            expected_forbidden = {r['sha256'] for r in channel(baseline['audit'], 'memory_keyframe_images')} - {r['sha256'] for r in want_events}
            require(set(intervention['forbidden_image_hashes']) == expected_forbidden, 'Incorrect forbidden set')
            for audit in audits:
                require(audit['language'] == parent['native_instruction'], 'Language changed in later request')
                require(audit['paired_seed'] == 91000+audit['step'], 'Pair RNG seed differs')
                require(audit['forbidden_hits'] == 0 and not any(r['sha256'] in expected_forbidden for r in audit['images']), 'Old pixels leaked')
            request_count += len(audits)
            lifts = result['lift_events']; reveals = result['revealed_events']; target = parent['target_slot']
            lift_slots = lifts[0]['slots'] if lifts else []
            reveal_slots = reveals[0]['slots'] if reveals else []
            require(result['first_lift_correct'] == (lift_slots == [target]), 'Lift label inconsistent')
            require(result['first_reveal_correct'] == (reveal_slots == [target]), 'Reveal label inconsistent')
            require(result['first_lift_matches_donor'] == (lift_slots == [donor_parent['target_slot']]), 'Donor label inconsistent')
            row = dict(seed=seed, condition=condition, result_file=result_files[seed, condition],
                target_slot=target, donor_target_slot=donor_parent['target_slot'],
                first_lift_slots=lift_slots, first_reveal_slots=reveal_slots,
                all_lift_events=lifts,
                first_lift_correct=result['first_lift_correct'], first_reveal_correct=result['first_reveal_correct'],
                first_lift_matches_donor=result['first_lift_matches_donor'],
                actions_executed=result['actions_executed'], model_requests=len(audits),
                first_input_bytes=first['image_payload_bytes'], first_event_count=first['event_images'],
                native_failed=result['native_failed'], action_hash=result['action_hash'],
                reached_action_limit=result['actions_executed'] == result['job']['max_actions'],
                stopped_after_wrong_lift=bool(lifts and lift_slots != [target] and not reveals))
            rows.append(row)
            if condition == 'keep':
                keep_reproductions.append(dict(seed=seed,
                    action_hash_matches_previous=result['action_hash'] == previous['action_hash'],
                    final_physics_matches_previous=result['final_physics_hash'] == previous['final_physics_hash']))
    counts = {c: dict(n=sum(r['condition'] == c for r in rows),
        first_lift_correct=sum(r['first_lift_correct'] for r in rows if r['condition'] == c),
        first_reveal_correct=sum(r['first_reveal_correct'] for r in rows if r['condition'] == c),
        first_lift_matches_donor=sum(r['first_lift_matches_donor'] for r in rows if r['condition'] == c)) for c in CONDITIONS}
    return dict(status='verified', run_dir=state['run_dir'], rows=rows, counts=counts,
        audited_model_requests=request_count, keep_reproductions=keep_reproductions,
        resume_audit=resume_audit,
        formal_training_eligible=False,
        limitation='Two selected native parents; first-choice diagnostics only, whole-history donor swap; no recovery or K/R labels.')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('status', type=Path)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    state = read(args.status); report = summarize(state)
    resource_runs = [state['run_dir']]
    if state.get('resumed_from'): resource_runs.append(state['resumed_from'])
    samples = [json.loads(line) for run in resource_runs
               for line in (artifact(run)/'resources.jsonl').read_text().splitlines() if line.strip()]
    gpu_samples = [g for s in samples for g in s['gpus'] if int(g['index']) == state['gpu']]
    report['resources'] = dict(run_dirs=resource_runs, samples=len(samples), alarm_samples=sum(bool(s['alarms']) for s in samples),
        min_available_GiB=min(s['memory']['MemAvailable_KiB']/1024**2 for s in samples),
        max_cpu_C=max(t['celsius'] for s in samples for t in s['temperatures']),
        max_gpu_C=max(float(g['temperature_C']) for g in gpu_samples),
        max_gpu_memory_MiB=max(float(g['memory_MiB']) for g in gpu_samples))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status', 'counts', 'audited_model_requests', 'keep_reproductions', 'resources']}, indent=2))


if __name__ == '__main__': main()
