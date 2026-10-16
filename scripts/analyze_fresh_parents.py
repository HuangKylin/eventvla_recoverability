"""Audit fresh-parent attempts, retaining rejected and unsuccessful outcomes."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path, PurePosixPath
import sys
from analyze_continuation_probe import ROOT,local,read
sys.path.insert(0,str(ROOT/'src'))
from recoverability.fixed_skills import SharedRecoveryController
from recoverability.vla_intervention import image_hash


class TrustedSnapshotReader(pickle.Unpickler):
    """Read our own Linux snapshots on Windows; never accept untrusted pickles."""
    def find_class(self,module,name):
        if module=='pathlib' and name=='PosixPath':return PurePosixPath
        return super().find_class(module,name)


def scene_group(source,audit):
    with (source/'client_state.pkl').open('rb') as f:state=TrustedSnapshotReader(f).load()
    entries=state['keyframe_image_memory']
    hashes=[image_hash(im) for entry in entries for im in entry['images']]
    actual=[x['sha256'] for x in audit['audit']['images'] if x['channel']=='memory_keyframe_images']
    if hashes!=actual:raise ValueError('Grouping images differ from audited model input')
    colours={}
    for entry in entries:
        if len(entry['images'])!=1:return None
        readings=[(slot,SharedRecoveryController.read(slot,entry['images'][0]).answer) for slot in range(4)]
        visible=[(slot,answer) for slot,answer in readings if answer is not None]
        if len(visible)!=1 or visible[0][0] in colours:return None
        colours[visible[0][0]]=visible[0][1]
    if len(colours)!=4 or len(set(colours.values()))!=4:return None
    return '-'.join(colours[slot] for slot in range(4))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reports',type=Path,nargs='+',default=[ROOT/'reports/latest_native_parent_collection.json'])
    args=parser.parse_args();checks={};rows=[];seen=set()
    for manifest in args.reports:
        state=read(manifest)
        if state['status']!='complete' or not state.get('process_cleanup_completed'):
            raise ValueError('Finish the collection before auditing it: '+str(manifest))
        for seed in state['seeds']:
            if seed in seen:raise ValueError('Repeated parent seed is not an independent new sample')
            seen.add(seed);es=[e for e in state['episodes'] if e['seed']==seed]
            prepare=next(e for e in es if e['condition']=='prepare')
            source=local(prepare['result_file']).parent;result=read(source/'result.json')
            prefix=str(seed)+'_'
            checks[prefix+'attempt_recorded']=result['status']=='complete'
            checks[prefix+'no_expert']=result['forbidden_calls']==0
            checks[prefix+'predeclared']=seed in state['predeclared_seed_pool']
            row=dict(seed=seed,run_dir=state['run_dir'],parent_saved=result['parent_saved'],
                     prefix_actions=result['actions_executed'],collection_seconds=result['elapsed_seconds'],
                     first_reveal_correct=None,full_task_evaluated=False,formal_training_eligible=False)
            if not result['parent_saved']:
                row['rejection_reason']=result['rejection_reason']
                checks[prefix+'rejection_preserved']=bool(row['rejection_reason']) and not any(e['condition']=='native' for e in es)
            else:
                parent=read(source/'parent.json');replay=read(source/'replay_validation.json')
                native=next(e for e in es if e['condition']=='native');keep=read(local(native['result_file']))
                audit=read(source/'input_audit/report.json')
                checks[prefix+'closed_chunk']=parent['boundary']['eligible'] and parent['visibility']['all_hidden']
                checks[prefix+'four_events']=len(parent['event_steps'])==4 and len(set(parent['event_steps']))==4
                checks[prefix+'replay']=replay['passed'] and all(replay['checks'].values())
                checks[prefix+'actual_parent']=all(keep['parent_match'].values())
                checks[prefix+'first_request']=keep['request_audits'][0]['first_input_verified'] and audit['actual_request_images_match']
                checks[prefix+'offline_deletions']=audit['status']=='verified' and all(x['forbidden_hits']==0 and x['anchor_images_unchanged'] for x in audit['deletion_checks'])
                checks[prefix+'keep_no_expert']=keep['forbidden_calls']==0
                checks[prefix+'keep_complete']=keep['status']=='complete'
                checks[prefix+'metadata_pair']=parent['seed']==seed and parent['step']==keep['commit_step']
                row.update(parent_step=parent['step'],first_reveal_correct=keep['first_reveal_correct'],
                    keep_actions=keep['actions_executed'],keep_seconds=keep['elapsed_seconds'],
                    replay_seconds=replay['elapsed_seconds'],parent_source=source.relative_to(ROOT).as_posix())
                identity={k:parent[k] for k in ['physics_hash','observation_hash','history_hash','event_hash','native_instruction']}
                row['parent_fingerprint']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
                row['colour_layout_group']=scene_group(source,audit)
            rows.append(row)
    groups={}
    for row in rows:
        if row['parent_saved']:groups.setdefault(row['parent_fingerprint'],[]).append(row['seed'])
    layouts={}
    for row in rows:
        if row.get('colour_layout_group'):layouts.setdefault(row['colour_layout_group'],[]).append(row['seed'])
    report=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,parents=rows,
        attempted_seeds=len(rows),eligible_parents=sum(x['parent_saved'] for x in rows),
        correct_first_reveals=sum(x['first_reveal_correct'] is True for x in rows),
        unique_exact_parent_states=len(groups),duplicate_parent_seed_groups=[seeds for seeds in groups.values() if len(seeds)>1],
        unique_colour_layouts=len(layouts),colour_layout_groups=layouts,
        unresolved_colour_layout_seeds=[r['seed'] for r in rows if r['parent_saved'] and not r.get('colour_layout_group')],
        grouping_note='Different seeds are not automatically independent scenes. Exact states and colour permutations are reported separately. RGB-derived layout groups are audit-only; group shared layouts together before formal train/validation/test. Fixed geometry has at most 24 colour permutations; geometry/cost/action diversification is still required.',
        full_task_evaluated=False,formal_training_eligible=False,
        source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'reports/native_fresh_parent_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
