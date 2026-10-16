"""Validate completed recovery integration controls; never emit training labels."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    path=Path(path)
    if not path.exists():path=ROOT/path.as_posix().split(ROOT.name+'/',1)[1]
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    state=read(ROOT/'reports/latest_native_recovery_vla.json')
    if state['status']!='complete':raise ValueError('Need all six completed branches')
    rows={(e['seed'],e['condition']):read(e['result_file']) for e in state['episodes']}
    old=read(ROOT/'reports/latest_native_memory_controls.json')
    reference={e['seed']:read(e['result_file']) for e in old['episodes'] if e['condition']=='keep'}
    checks={};summary=[]
    for seed in state['seeds']:
        for condition in ['keep','recover','recover_discard']:
            result=rows[seed,condition];key=f'{seed}_{condition}'
            requests=result['request_audits'];first=requests[0]
            checks[key+'_parent']=all(result['parent_match'].values())
            checks[key+'_input']=bool(first.get('first_input_verified')) and all(a['forbidden_hits']==0 for a in requests)
            checks[key+'_oracle']=result['forbidden_calls']==0
            expected_count=4 if condition=='keep' else (len(result['recovered_observations']) if condition=='recover' else 0)
            checks[key+'_first_event_count']=first['event_images']==expected_count
            if condition!='keep':
                checks[key+'_closed_history']=result['closed_history_verified']
                checks[key+'_all_history_cameras_hidden']=bool(result['closed_history_visibility']) and all(x['all_hidden'] for x in result['closed_history_visibility'])
            summary.append(dict(seed=seed,condition=condition,first_lift_correct=result['first_lift_correct'],
                first_reveal_correct=result['first_reveal_correct'],actions=result['actions_executed'],
                recovery_physics_steps=result['recovery_physics_steps'],policy_physics_steps=result['policy_physics_steps'],
                recovered_slots=result.get('recovery',{}).get('recovered_slots'),
                unrecovered_slots=result.get('recovery',{}).get('unrecovered_slots'),
                model_requests=len(requests),image_payload_bytes=sum(a['image_payload_bytes'] for a in requests)))
        keep=rows[seed,'keep'];baseline=reference[seed]
        checks[f'{seed}_keep_action_reproduction']=keep['action_hash']==baseline['action_hash']
        checks[f'{seed}_keep_physics_reproduction']=keep['final_physics_hash']==baseline['final_physics_hash']
        recovered,discarded=rows[seed,'recover'],rows[seed,'recover_discard']
        for field in ['commit_step','commit_physics_hash','commit_observation_hash','commit_history_hash','recovery_physics_steps']:
            checks[f'{seed}_recovery_pair_{field}']=recovered[field]==discarded[field]
        def anchors(row):return [x for x in row['request_audits'][0]['images'] if x['channel']=='image']
        checks[f'{seed}_recovery_pair_anchors']=anchors(recovered)==anchors(discarded)
        checks[f'{seed}_recovery_pair_rng']=recovered['request_audits'][0]['paired_seed']==discarded['request_audits'][0]['paired_seed']
    report=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,summary=summary,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        run_dir=state['run_dir'],formal_training_eligible=False,
        scope='Two selected native parents; all-event bundle integration, not per-item K/R or general success rate.')
    dest=ROOT/'reports/native_recovery_validation.json';dest.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not all(checks.values()):raise RuntimeError('Recovery integration comparison failed validation')


if __name__=='__main__':main()
