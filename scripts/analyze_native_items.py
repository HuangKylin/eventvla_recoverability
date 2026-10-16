"""Audit per-item pilots and export grouped label components, never training claims."""
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_continuation_probe import ROOT, local, read


def images(payload, channel):
    return [x for x in payload['images'] if x['channel']==channel]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    state=read(ROOT/'reports/latest_native_recovery_item_pilot.json')
    complete=state['status']=='complete' and state.get('process_cleanup_completed',False)
    if not complete and not args.allow_partial:
        raise RuntimeError('Pilot must finish and clean up before label export')
    checks={};labels=[];summary=[];missing=[]
    for seed in state['seeds']:
        es=[e for e in state['episodes'] if e['seed']==seed]
        rows={e['condition']:read(local(e['result_file'])) for e in es}
        paths={e['condition']:local(e['result_file']) for e in es}
        if 'keep' not in rows:
            missing.extend(dict(seed=seed,candidate_index=i,reason='No completed keep') for i in state['candidate_indices']);continue
        keep=rows['keep'];reference=next(e for e in es if e['condition']=='keep')
        checks[f'{seed}_keep_reproduced']=all(reference['reference_match'].values())
        for index in state['candidate_indices']:
            if any(k not in rows for k in [f'drop_item_{index}',f'recover_item_{index}']):
                missing.append(dict(seed=seed,candidate_index=index,reason='K/D/R triple not complete'));continue
            d=rows[f'drop_item_{index}'];r=rows[f'recover_item_{index}'];prefix=f'{seed}_{index}_'
            checks[prefix+'same_candidate']=d['candidate_step']==r['candidate_step'] and d['public_recovery_slot']==r['public_recovery_slot']
            checks[prefix+'same_untouched_memories']=d['intervention']['noncandidate_signature']==r['intervention']['noncandidate_signature']
            checks[prefix+'D_same_commit']=all(d[k]==keep[k] for k in ['commit_step','commit_physics_hash','commit_observation_hash','commit_history_hash'])
            checks[prefix+'D_same_anchor_images']=images(d['expected_first_input'],'image')==images(keep['expected_first_input'],'image')
            old=images(keep['expected_first_input'],'memory_keyframe_images')
            checks[prefix+'D_only_one_removed']=images(d['expected_first_input'],'memory_keyframe_images')==old[:index]+old[index+1:]
            checks[prefix+'R_other_events_preserved']=images(r['expected_first_input'],'memory_keyframe_images')[:3]==old[:index]+old[index+1:]
            checks[prefix+'R_count']=r['expected_first_input']['event_images']==3+int(r['candidate_recovered'])
            checks[prefix+'R_route']=len(r['recovery']['records'])==1 and r['recovery']['records'][0]['slot']==r['public_recovery_slot']
            checks[prefix+'R_hidden']=r['closed_history_verified'] and all(v['all_hidden'] for v in r['closed_history_visibility'])
            checks[prefix+'costs']=d['recovery_physics_steps']==0 and r['recovery_physics_steps']>0
            for kind,value in [('D',d),('R',r)]:
                checks[prefix+kind+'_parent']=all(value['parent_match'].values())
                checks[prefix+kind+'_complete']=value['status']=='complete'
                checks[prefix+kind+'_noncandidate']=value['noncandidate_preserved_at_commit']
                checks[prefix+kind+'_first_actual_input']=value['request_audits'][0]['first_input_verified']
                checks[prefix+kind+'_leak']=all(a['forbidden_hits']==0 for a in value['request_audits'])
                checks[prefix+kind+'_oracle']=value['forbidden_calls']==0
                checks[prefix+kind+'_excluded']=value['formal_training_eligible'] is False
            pd=paths[f'drop_item_{index}'].parent;pr=paths[f'recover_item_{index}'].parent
            fd=read(pd/'candidate_public.json');fr=read(pr/'candidate_public.json')
            checks[prefix+'same_public_metadata']=fd==fr
            checks[prefix+'metadata_whitelist']=set(fd)=={'language','candidate_step','decision_step','public_slot','event_steps','action_condition','feature_schema'}
            with np.load(pd/'candidate_public.npz',allow_pickle=False) as a,np.load(pr/'candidate_public.npz',allow_pickle=False) as b:
                checks[prefix+'same_public_arrays']=a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
                checks[prefix+'array_whitelist']=set(a.files)=={'candidate_head','current_head','current_left','current_right','proprioception'}
            success={k:bool(v['first_reveal_correct']) for k,v in [('K',keep),('D',d),('R',r)]}
            costs={k:v['recovery_physics_steps']+v['policy_physics_steps'] for k,v in [('K',keep),('D',d),('R',r)]}
            row=dict(parent_group=f'development_parent_{seed}',candidate_index=index,candidate_step=d['candidate_step'],
                public_slot=d['public_recovery_slot'],split='development_only',formal_training_eligible=False,
                endpoint='first_correct_target_reveal',action_pairs_present=False,success=success,physics_ticks=costs,
                failure_delta_R_minus_K=int(not success['R'])-int(not success['K']),
                failure_delta_D_minus_K=int(not success['D'])-int(not success['K']),
                recovery_gain_R_minus_D=int(success['R'])-int(success['D']),
                physics_delta_R_minus_K=costs['R']-costs['K'],
                recovered=r['candidate_recovered'],
                input_arrays=(pd/'candidate_public.npz').relative_to(ROOT).as_posix(),
                input_metadata=(pd/'candidate_public.json').relative_to(ROOT).as_posix(),
                branches={k:paths[c].relative_to(ROOT).as_posix() for k,c in [('K','keep'),('D',f'drop_item_{index}'),('R',f'recover_item_{index}')]})
            labels.append(row);summary.append({k:row[k] for k in ['parent_group','candidate_index','public_slot','success','recovered','physics_delta_R_minus_K']})
    passed=bool(labels) and all(checks.values())
    result=dict(status=('passed' if complete else 'passed_partial') if passed else 'failed',checks=checks,summary=summary,
        independent_parent_count=len(state['seeds']),candidate_pair_count=len(labels),actual_branches=len(state['episodes']),
        run_complete=complete,last_known_run_status=state['status'],missing_pairs=missing,
        source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),formal_training_eligible=False)
    suffix='' if complete else '_partial'
    (ROOT/f'reports/native_item_pilot{suffix}_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    if not passed:
        print(json.dumps({k:v for k,v in checks.items() if not v},indent=2));raise SystemExit(1)
    dest=ROOT/'data/native_item_development_pilot';dest.mkdir(parents=True,exist_ok=True)
    (dest/f'labels{suffix}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels),encoding='utf-8')
    (dest/f'manifest{suffix}.json').write_text(json.dumps(dict(status='validated_development_only' if complete else 'validated_partial_development_only',run_dir=state['run_dir'],
        run_complete=complete,missing_pairs=missing,
        independent_parents=len(state['seeds']),candidate_pairs=len(labels),actual_branches=len(state['episodes']),
        formal_training_eligible=False,split_rule='All sibling candidates and action conditions of a parent stay together; these tuned parents are excluded from formal validation/test.',
        endpoint='First target reveal only; not full native task success',
        randomness='Current client uses 91000+timestep, so R and K may use different random streams after elapsed recovery. Common-random-number control remains required for formal labels.'),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],checks_passed=len(checks),summary=summary),indent=2))


if __name__=='__main__':main()
