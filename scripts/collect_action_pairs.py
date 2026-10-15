"""Collect strictly matched fresh-process K/R labels for the physical T3 probe."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[310000,310001,310002,310003])
    p.add_argument('--actions',type=int,nargs='+',default=[0,3])
    p.add_argument('--gpu',default='1')
    a=p.parse_args()
    from recoverability.branch_protocol import BranchOutcome,CostConfig,deletion_cost
    names=['scripts/action_pair_worker.py','scripts/collect_action_pairs.py']+[
        'src/recoverability/'+n+'.py' for n in
        ('action_latch_task','fixed_skills','cue_probe_task','cue_reader','snapshot','sim_runtime','branch_protocol')]
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_action_pairs')
    run.mkdir(parents=True)
    state=dict(status='running',run_dir=str(run),seeds=a.seeds,actions=a.actions,
        source_hashes=hashes,episodes=[],labels=[],planned_episodes=len(a.seeds)*len(a.actions)*6,
        scope='fixed_geometry_physical_latch_mechanism_pilot',formal_eventvla_training_eligible=False)
    def save():
        text=json.dumps(state,indent=2)
        (run/'status.json').write_text(text)
        (ROOT/'reports/latest_action_pairs.json').write_text(text)
    def worker(seed,action,drop,repeat=False):
        name=f's{seed}_a{action}_'+('K' if drop is None else f'R{drop}')+('_repeat' if repeat else '')
        out=run/name
        job=dict(seed=seed,action_slot=action,drop_slot=drop,output=str(out),gpu=a.gpu,
                 library=str(a.library.resolve()),source_hashes=hashes)
        jobfile=run/(name+'.job.json')
        jobfile.write_text(json.dumps(job,indent=2))
        with (run/(name+'.log')).open('w') as log:
            proc=subprocess.Popen([str(ROOT/'envs/sim/bin/python'),'-u',
                str(ROOT/'scripts/action_pair_worker.py'),str(jobfile)],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                if proc.wait(timeout=240):
                    raise RuntimeError('Worker failed: '+name)
            finally:
                try: os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError: pass
        r=json.loads((out/'result.json').read_text())
        if r['status']!='complete' or r['forbidden_calls']:
            raise RuntimeError('Runtime access or execution gate failed')
        state['episodes'].append(dict(name=name,result_file=str(out/'result.json')))
        save()
        print(json.dumps({'completed':len(state['episodes']),'planned':state['planned_episodes'],'episode':name}),flush=True)
        return r
    start=time.monotonic()
    save()
    try:
        for seed in a.seeds:
            parent=None
            for action in a.actions:
                keep=worker(seed,action,None)
                identity={k:keep[k] for k in ('parent_hash','prefix_trace_hash','prefix_image_hashes',
                                             'current_hash','rng_hash','library_hash','order')}
                if parent is None: parent=identity
                if identity!=parent: raise RuntimeError('Action pairing changed pre-action history/state')
                recoveries=[]
                for slot in range(4):
                    recover=worker(seed,action,slot)
                    if {k:recover[k] for k in identity}!=parent:
                        raise RuntimeError('K/R prefix mismatch')
                    for k in ('action_hash','post_action_hash','post_action_rng_hash','events'):
                        if keep[k]!=recover[k]: raise RuntimeError('K/R continuation mismatch: '+k)
                    k=next(q for q in keep['outcomes'] if q['slot']==slot)
                    r=recover['outcomes'][0]
                    common=dict(family_id=f'action_latch_{seed}',parent_state_hash=keep['parent_hash'],
                        action_context_hash=keep['action_hash'],evidence_id=str(slot),
                        controller_hash=keep['library_hash'],rng_state_hash=keep['rng_hash'],
                        compute_seconds=0.,recovery_allowed=True,restore_validated=True,restore_error=0.)
                    delta=deletion_cost(BranchOutcome(**common,branch='K',success=k['success'],steps=k['recovery_steps']),
                        BranchOutcome(**common,branch='R',success=r['success'],steps=r['recovery_steps']),
                        CostConfig(10000,1.,step_weight=.1,compute_weight=0.),restore_tolerance=0.)
                    row=dict(family_id=common['family_id'],seed=seed,action_slot=action,evidence_slot=slot,
                        ordinal=keep['order'].index(slot),delta_cost=delta,keep=k,recover=r,
                        parent_hash=keep['parent_hash'],action_hash=keep['action_hash'],
                        evidence_image=str(Path(keep['job']['output'])/f'evidence_{slot}.png'),
                        public_action=str(Path(keep['job']['output'])/'public_action.npz'),
                        keep_result=str(Path(keep['job']['output'])/'result.json'),
                        recover_result=str(Path(recover['job']['output'])/'result.json'),
                        paired_prefix_and_continuation_validated=True)
                    state['labels'].append(row)
                    recoveries.append(recover)
                repeated=worker(seed,action,action,repeat=True)
                original=recoveries[action]
                if any(original[k]!=repeated[k] for k in ('parent_hash','post_action_hash','events','outcomes')):
                    raise RuntimeError('Repeated R branch disagrees')
                save()
        data=ROOT/'data/mechanism_pilot'/run.name
        data.mkdir(parents=True)
        (data/'labels.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in state['labels']))
        state.update(status='complete',data_file=str(data/'labels.jsonl'),elapsed_seconds=time.monotonic()-start,
                     all_pairs_matched=True,all_repeated_R_matched=True)
        save()
    except Exception:
        state.update(status='failed',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
        save()
        raise


if __name__=='__main__': main()
