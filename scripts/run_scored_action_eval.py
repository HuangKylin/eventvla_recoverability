"""Execute trained scorer and strong baselines on the held-out pilot family."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
from run_baseline import stop_group
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--training-report',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True);a=p.parse_args()
    trained=json.loads(a.training_report.read_text())
    if trained['status']!='complete': raise RuntimeError('Training must finish')
    collection=json.loads(Path(trained['collection']).read_text())
    test_families=trained['families_by_split']['test']
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_scored_action_eval')
    run.mkdir(parents=True)
    names=['scripts/scored_action_worker.py','scripts/run_scored_action_eval.py','src/recoverability/scorer.py',
           'src/recoverability/action_latch_task.py','src/recoverability/fixed_skills.py']
    state=dict(status='running',run_dir=str(run),episodes=[],training_report=str(a.training_report),
        source_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        scope='held_out_color_order_family_fixed_controller_one_shot_capacity_pilot')
    def save():
        text=json.dumps(state,indent=2);(run/'status.json').write_text(text)
        (ROOT/'reports/latest_scored_action_eval.json').write_text(text)
    child=None;save()
    try:
        for family in test_families:
            seed=int(family.rsplit('_',1)[1]);parent=None
            for action in collection['actions']:
                for method in ('learned_action','learned_no_action','fifo','oldest','action_rule','archive','structured'):
                    out=run/f's{seed}_a{action}_{method}'
                    job=dict(seed=seed,action_slot=action,method=method,output=str(out),library=str(a.library.resolve()))
                    if method in trained['models']: job['checkpoint']=trained['models'][method]['checkpoint']
                    jobfile=out.with_suffix('.job.json');jobfile.write_text(json.dumps(job,indent=2))
                    with out.with_suffix('.log').open('w') as log:
                        child=subprocess.Popen([str(ROOT/'envs/sim/bin/python'),'-u',str(ROOT/'scripts/scored_action_worker.py'),str(jobfile)],
                            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        try:
                            if child.wait(timeout=300): raise RuntimeError('Rollout failed: '+str(out))
                        finally: stop_group(child);child=None
                    result=json.loads((out/'result.json').read_text())
                    if result['status']!='complete' or result['forbidden_calls']: raise RuntimeError('Runtime gate failed')
                    if parent is None: parent=result['parent_hash']
                    if parent!=result['parent_hash']: raise RuntimeError('Method/action prefix mismatch')
                    state['episodes'].append(dict(seed=seed,action_slot=action,method=method,result_file=str(out/'result.json'),
                        correct=result['correct'],queries=4,retained_slots=result['retained_slots'],
                        recovery_steps=result['recovery_steps'],image_payload_bytes=result['image_payload_bytes'],
                        semantic_payload_bytes=result['semantic_payload_bytes']))
                    save()
        state.update(status='complete',all_parent_states_matched=True)
        save()
    except Exception:
        state.update(status='failed',error=traceback.format_exc());save();raise
    finally: stop_group(child)


if __name__=='__main__': main()
