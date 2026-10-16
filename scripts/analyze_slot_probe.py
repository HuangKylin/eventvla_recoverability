"""Separate technical pairing checks from empirical slot-calibration outcomes."""
import argparse
import json
from pathlib import Path
import sys
from analyze_continuation_probe import ROOT,local,read
sys.path.insert(0,str(ROOT/'src'))
from recoverability.slot_protocol import CONDITIONS,PROTOCOL,paired_request_seed,specification


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,default=ROOT/'reports/latest_native_slot_probe.json')
    args=parser.parse_args();state=read(args.report)
    if state['status']!='complete':raise ValueError('Finish all predeclared calibration branches first')
    checks={};rows=[];fill_comparisons=[];control_health=[]
    for seed in state['seeds']:
        episodes={e['condition']:e for e in state['episodes'] if e['seed']==seed}
        if set(episodes)!=set(CONDITIONS):raise ValueError('Incomplete predeclared conditions')
        values={name:read(local(e['result_file'])) for name,e in episodes.items()}
        paired=values['slots_paired'];native=values['native_legacy']
        for condition,result in values.items():
            prefix=f'{seed}_{condition}_';spec=specification(condition);requests=result['request_audits']
            checks[prefix+'complete']=result['status']=='complete' and bool(requests)
            checks[prefix+'parent']=all(result['parent_match'].values())
            checks[prefix+'no_recovery']=result['recovery_physics_steps']==0 and not result['recovery_allowed']
            checks[prefix+'no_oracle']=result['forbidden_calls']==0
            checks[prefix+'first_actual_input']=requests[0]['first_input_verified']
            checks[prefix+'no_old_image_bypass']=all(a['forbidden_hits']==0 for a in requests)
            checks[prefix+'same_physical_parent']=all(result[k]==native[k] for k in ('commit_step','commit_physics_hash','commit_observation_hash','commit_history_hash'))
            if spec['capacity'] is not None:checks[prefix+'fixed_count']=all(a['event_images']==4 for a in requests)
            checks[prefix+'rng']=all(a['request_index']==i and a['paired_seed']==(
                91000+a['step'] if spec['rng']=='timestep_legacy' else paired_request_seed(result['randomness_key'],i)) for i,a in enumerate(requests))
            if 'reference_match' in episodes[condition]:checks[prefix+'reproduces_reference']=all(episodes[condition]['reference_match'].values())
            before=native['expected_first_input'];after=result['expected_first_input']
            channel=lambda payload,name:[x for x in payload['images'] if x['channel']==name]
            checks[prefix+'same_anchors']=channel(before,'image')==channel(after,'image')
            checks[prefix+'same_metadata']=all(before[k]==after[k] for k in ('event_steps','event_metas','anchor_metas','language'))
            changed=[i for i,(a,b) in enumerate(zip(channel(before,'memory_keyframe_images'),channel(after,'memory_keyframe_images'))) if a!=b]
            checks[prefix+'only_candidate_changed']=changed==([] if spec['candidate_index'] is None else [spec['candidate_index']])
            rows.append(dict(seed=seed,condition=condition,first_reveal_correct=result['first_reveal_correct'],
                actions=result['actions_executed'],policy_physics_steps=result['policy_physics_steps'],
                candidate_index=spec['candidate_index'],rng=spec['rng']))
        control_health.append(dict(seed=seed,native=native['first_reveal_correct'],
            fixed_capacity_legacy=values['slots_legacy']['first_reveal_correct'],
            fixed_capacity_paired=paired['first_reveal_correct']))
        for index in range(4):
            gray=values[f'gray_item_{index}'];black=values[f'black_item_{index}']
            fill_comparisons.append(dict(seed=seed,candidate_index=index,
                gray_correct=gray['first_reveal_correct'],black_correct=black['first_reveal_correct'],
                success_agrees=gray['first_reveal_correct']==black['first_reveal_correct'],
                gray_physics_steps=gray['policy_physics_steps'],black_physics_steps=black['policy_physics_steps']))
    report=dict(status='passed' if all(checks.values()) else 'failed',protocol=PROTOCOL,checks=checks,
        summary=rows,keep_control_health=control_health,padding_sensitivity=fill_comparisons,
        all_keep_controls_succeed=all(all(v for k,v in row.items() if k!='seed') for row in control_health),
        all_padding_successes_agree=all(row['success_agrees'] for row in fill_comparisons),
        formal_training_eligible=False,full_task_evaluated=False,
        interpretation='Technical checks validate execution, not scientific effect. These two parents are calibration only. Gray is primary and black is sensitivity analysis; never select padding per item or using success. R, action pairs, scene-group splits and cost sweeps remain unvalidated.')
    path=ROOT/'reports/native_slot_probe_validation.json';path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],checks=len(checks),branches=len(rows),
        all_keep_controls_succeed=report['all_keep_controls_succeed'],all_padding_successes_agree=report['all_padding_successes_agree'])))
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
