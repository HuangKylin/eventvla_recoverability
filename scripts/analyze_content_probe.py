"""Verify fixed-count image interventions before interpreting their outcomes."""
import hashlib
import json
from pathlib import Path
from analyze_continuation_probe import ROOT,local,read
from analyze_native_items import images


def main():
    state=read(ROOT/'reports/latest_native_recovery_content_probe.json')
    assert state['status']=='complete' and state.get('process_cleanup_completed')
    checks={};summary=[]
    for seed in state['seeds']:
        es=[e for e in state['episodes'] if e['seed']==seed]
        k=read(local(next(e['result_file'] for e in es if e['condition']=='keep')))
        for e in es:
            if e['condition']=='keep':continue
            d=read(local(e['result_file']));index=d['candidate_index'];prefix=f'{seed}_{index}_'
            a=d['expected_first_input'];b=k['expected_first_input']
            checks[prefix+'same_state']=all(d[n]==k[n] for n in ['commit_step','commit_physics_hash','commit_observation_hash','commit_history_hash'])
            checks[prefix+'same_anchors']=images(a,'image')==images(b,'image')
            checks[prefix+'same_metadata']=all(a[n]==b[n] for n in ['event_steps','event_metas','anchor_metas','language'])
            checks[prefix+'same_count']=a['event_images']==b['event_images']==4
            old=images(b,'memory_keyframe_images');new=images(a,'memory_keyframe_images')
            checks[prefix+'one_image_changed']=[i for i,(x,y) in enumerate(zip(old,new)) if x!=y]==[index]
            checks[prefix+'uses_observed_current_head']=new[index]['sha256']==images(a,'image')[-3]['sha256']
            checks[prefix+'no_cost']=d['recovery_physics_steps']==0
            checks[prefix+'same_initial_rng']=d['request_audits'][0]['paired_seed']==k['request_audits'][0]['paired_seed']
            checks[prefix+'parent']=all(d['parent_match'].values())
            checks[prefix+'input_audit']=d['request_audits'][0]['first_input_verified'] and all(v['forbidden_hits']==0 for v in d['request_audits'])
            checks[prefix+'no_oracle']=d['forbidden_calls']==0
            checks[prefix+'not_training']=d['formal_training_eligible'] is False and d['scope']=='constant_count_content_probe'
            summary.append(dict(seed=seed,candidate_index=index,slot=d['public_recovery_slot'],
                first_reveal_correct=d['first_reveal_correct'],first_lift_correct=d['first_lift_correct'],
                actions=d['actions_executed'],elapsed_seconds=d['elapsed_seconds']))
    result=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,summary=summary,
        source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),formal_training_eligible=False,
        scope='Post-pilot exploratory content/count diagnostic; two selected candidates of one development parent')
    (ROOT/'reports/native_content_probe_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
