"""Summarize the fixed diagnostic without treating content probes as training labels."""
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
from resume_native_controls import atomic_json
from run_evidence_diagnostic import sha


def main():
    state=json.loads((ROOT/'reports/latest_evidence_failure_branches.json').read_text())
    if state['status']!='complete' or not state['process_cleanup_completed']:
        raise ValueError('Require finished and cleaned batch')
    run=Path(state['run_dir']);source=Path(state['source'])
    parent=json.loads((source/'parent.json').read_text())
    native=json.loads((source.parent/'status.json').read_text())
    census=json.loads((source.parent/'analysis.json').read_text())
    manifest=json.loads((run/'selection_manifest.json').read_text())
    if sha(run/'selection_manifest.json')!=state['manifest_sha256']:
        raise ValueError('Manifest changed')
    if [e['name'] for e in state['episodes']] != [b['name'] for b in manifest['branches']]:
        raise ValueError('Incomplete or changed branch set')
    hashes=defaultdict(lambda:defaultdict(list))
    for row in native['episodes']:
        path=Path(row['result_file'])
        if sha(path)!=row['result_sha256']:raise ValueError('Native result changed')
        result=json.loads(path.read_text())
        hashes[row['task']][result['action_hash']].append(row['seed'])
    original=np.load(source/'executed_actions.npz',allow_pickle=False)['actions'][parent['step']:]
    branches=[]
    for row in state['episodes']:
        path=Path(row.get('branch_dir',str(run/row['name'])))
        if sha(path/'result.json')!=row['result_sha256']:raise ValueError('Branch result changed')
        result=json.loads((path/'result.json').read_text())
        actions=np.load(path/'executed_actions.npz',allow_pickle=False)['actions']
        n=min(len(actions),len(original))
        branches.append(dict(name=row['name'],success=row['native_success'],failed=row['native_failed'],
            final_step=row['final_step'],continuation_actions=len(actions),
            first_native_suffix_actions_equal=np.array_equal(actions[:n],original[:n]),
            first_native_suffix_action_max_abs_difference=float(np.max(np.abs(actions[:n]-original[:n]))),
            original_suffix_actions_compared=n,final_press_cnt=row['final_state']['press_cnt'],
            other_center_achieved=row['final_state']['other_center_achieved'],
            parent_match=result['parent_match'],actual_input_audit=row['audit'],branch_dir=str(path),
            inherited_from=row.get('inherited_from')))
    repairs=sum(row['success'] for row in branches[1:])
    recency_repair=next(row['success'] for row in branches if row['name']=='public_recency')
    if recency_repair:
        decision='stop_current_task_route_simple_recency_repairs_only_failure'
        next_action='Archive this diagnostic; simple recency already repairs the only observed failure under this content-only protocol. No scorer training or automatic expansion.'
    elif repairs:
        decision='limited_control_repair_requires_authentic_metadata_and_stronger_baselines'
        next_action='Review the repair without starting training or automatic expansion'
    else:
        decision='stop_current_bounded_task_route_no_repair_observed'
        next_action='Archive this diagnostic; do not expand these tasks or start scorer training. A new claim needs a new explicit scope.'
    samples=[json.loads(x) for x in (run/'resources.jsonl').read_text().splitlines()]
    trace=[json.loads(x) for x in (source/'evaluator_trace.jsonl').read_text().splitlines()]
    timing=[{'step':x['step'],'phase':x['state']['phase'],'press_cnt':x['state']['press_cnt'],
             'press_flag':x['state']['press_flag'],'button_qpos':x['state']['button_qpos']}
            for x in trace if x['step'] in (520,521,529,530,531)]
    final=json.loads((source/'result.json').read_text())['final_evaluator_state']
    timing.append({'step':532,'phase':final['phase'],'press_cnt':final['press_cnt'],
                   'press_flag':final['press_flag'],'button_qpos':final['button_qpos'],'fail_flag':final['fail_flag']})
    report=dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),
        native_run=str(source.parent),branch_run=str(run),native_episodes=12,
        native_successes=sum(e['native_success'] for e in native['episodes']),
        native_failures=sum(not e['native_success'] for e in native['episodes']),
        native_actual_policy_requests=census['audited_policy_requests'],
        native_actual_input_images=census['images_verified_against_observed_rgb'],
        native_action_hash_groups=hashes,unique_action_sequences=sum(len(v) for v in hashes.values()),
        independence_note='Unique hashes are not independent task families. put_back repeats one identical action sequence six times.',
        unique_failed_parent_count=1,branch_count=len(branches),branches=branches,
        event_slot_count=parent['event_count'],configured_memory_capacity=5,
        tested_parent_has_capacity_pressure=False,native_button_timeline=timing,
        bounded_content_repairs=repairs,formal_KR_labels=0,formal_training_started=False,
        phase=2,original_research_goal_achieved=False,
        decision=decision,
        scope='One failed parent, two predetermined content alternatives; not an exhaustive evidence oracle or strongest-baseline comparison',
        timestamp_semantics=manifest['intervention_semantics'],
        failure_attribution='Native repeated button count at chunk boundary established; intent, motor continuity and button detector contributions not separately identified',
        next_action=next_action,
        resource_alarm_samples=sum(bool(x['alarms']) for x in samples),
        peak_selected_gpu_temperature_C=max(float(g['temperature_C']) for x in samples for g in x['gpus'] if int(g['index'])==state['gpu']),
        process_cleanup_completed=state['process_cleanup_completed'],
        remaining_batch_processes={key:Path(f'/proc/{state[key]}').exists() for key in ['pid','server_pid','worker_pid','pacer_pid']})
    atomic_json(run/'decision.json',report)
    atomic_json(ROOT/'reports/latest_evidence_failure_decision.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
