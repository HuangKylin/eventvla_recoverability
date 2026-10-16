"""Audit the two predeclared continuation controls against original-memory runs."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def local(path):
    return ROOT/str(path).replace('\\','/').split(ROOT.name+'/',1)[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    status=read(ROOT/'reports/latest_native_recovery_continuation_probe.json')
    original=read(ROOT/'reports/latest_native_recovery_original_probe.json')
    baseline=read(ROOT/'reports/latest_native_recovery_vla.json')
    assert status['status']=='complete' and status['process_cleanup_completed']
    checks={};rows=[]
    for seed in status['seeds']:
        ref=read(local(next(e['result_file'] for e in original['episodes'] if e['seed']==seed)))
        keep=read(local(next(e['result_file'] for e in baseline['episodes'] if e['seed']==seed and e['condition']=='keep')))
        parent=read(local(status['parent_sources'][str(seed)])/'parent.json')
        values={e['condition']:read(local(e['result_file'])) for e in status['episodes'] if e['seed']==seed}
        for condition,result in values.items():
            prefix=f'{seed}_{condition}_';a=result['expected_first_input'];b=keep['expected_first_input']
            events=lambda payload:[x for x in payload['images'] if x['channel']=='memory_keyframe_images']
            checks[prefix+'complete']=result['status']=='complete'
            checks[prefix+'parent']=all(result['parent_match'].values())
            checks[prefix+'original_events']=events(a)==events(b)
            checks[prefix+'original_metadata']=all(a[k]==b[k] for k in ('event_steps','event_metas','language'))
            checks[prefix+'clock']=result['commit_step']==ref['commit_step']
            checks[prefix+'duration']=result['matched_recovery_ticks']==ref['recovery_physics_steps']
            checks[prefix+'history_length']=len(result['commit_history_steps'])==result['temporal_history_capacity']
            checks[prefix+'first_input']=result['request_audits'][0]['first_input_verified']
            checks[prefix+'no_leak']=all(audit['forbidden_hits']==0 for audit in result['request_audits'])
            checks[prefix+'no_oracle']=result['forbidden_calls']==0
            checks[prefix+'no_visible_history']=result['closed_history_verified'] and all(v['all_hidden'] for v in result['closed_history_visibility'])
            checks[prefix+'rng']=result['request_audits'][0]['paired_seed']==ref['request_audits'][0]['paired_seed']
            checks[prefix+'not_training']=result['formal_training_eligible'] is False
            if condition=='time_only_original':
                checks[prefix+'physics_unchanged']=result['commit_physics_hash']==parent['physics_hash']
                checks[prefix+'observation_unchanged']=result['commit_observation_hash']==parent['observation_hash']
                checks[prefix+'zero_actual_cost']=result['recovery_physics_steps']==0
            else:
                checks[prefix+'actual_cost']=result['recovery_physics_steps']==ref['recovery_physics_steps']
            rows.append(dict(seed=seed,condition=condition,**{k:result[k] for k in [
                'first_reveal_correct','first_lift_correct','actions_executed','recovery_physics_steps',
                'policy_physics_steps','commit_step','elapsed_seconds']}))
        checks[f'{seed}_pair_history_steps']=values['time_only_original']['commit_history_steps']==values['hold_original']['commit_history_steps']
    result=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,summary=rows,
                source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Two development controls; first-reveal endpoint, not full task success or training data')
    path=ROOT/'reports/native_recovery_continuation_validation.json'
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
