"""Validate the exploratory original-memory control against the main recovery."""
import hashlib,json
from pathlib import Path
from analyze_native_recovery import ROOT,read


def main():
    state=read(ROOT/'reports/latest_native_recovery_original_probe.json')
    if state['status']!='complete':raise ValueError('Probe not complete')
    original=read(ROOT/'reports/latest_native_recovery_vla.json')
    refs={(e['seed'],e['condition']):read(e['result_file']) for e in original['episodes']}
    checks={};summary=[]
    for entry in state['episodes']:
        seed=entry['seed'];probe=read(entry['result_file']);recovery=refs[seed,'recover'];keep=refs[seed,'keep']
        for field in ['commit_step','commit_physics_hash','commit_observation_hash','commit_history_hash','recovery_physics_steps']:
            checks[f'{seed}_pair_{field}']=probe[field]==recovery[field]
        def images(row,channel):return [r for r in row['request_audits'][0]['images'] if r['channel']==channel]
        checks[f'{seed}_pair_anchors']=images(probe,'image')==images(recovery,'image')
        checks[f'{seed}_original_events']=images(probe,'memory_keyframe_images')==images(keep,'memory_keyframe_images')
        for field in ['event_steps','event_metas']:
            checks[f'{seed}_original_{field}']=probe['request_audits'][0][field]==keep['request_audits'][0][field]
        checks[f'{seed}_input_audit']=probe['request_audits'][0]['first_input_verified'] and all(x['forbidden_hits']==0 for x in probe['request_audits'])
        checks[f'{seed}_oracle']=probe['forbidden_calls']==0
        checks[f'{seed}_rng']=probe['request_audits'][0]['paired_seed']==recovery['request_audits'][0]['paired_seed']
        checks[f'{seed}_hidden_history']=bool(probe['closed_history_visibility']) and all(x['all_hidden'] for x in probe['closed_history_visibility'])
        summary.append({k:probe[k] for k in ['first_lift_correct','first_reveal_correct','lift_events','revealed_events','actions_executed','recovery_physics_steps','policy_physics_steps']})
        summary[-1].update(seed=seed,condition=entry['condition'],model_requests=len(probe['request_audits']))
    report=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,summary=summary,
        source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),run_dir=state['run_dir'],
        formal_training_eligible=False,scope='Exploratory single-scene diagnosis after main-run failure; not a new memory-method benchmark')
    (ROOT/'reports/native_recovery_original_probe_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not all(checks.values()):raise RuntimeError('Original-memory probe pairing failed')


if __name__=='__main__':main()
