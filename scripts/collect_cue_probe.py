"""Run and audit paired diagnostics. Outputs are explicitly excluded from training."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
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


def worker(job_path,log_path):
    with log_path.open('w') as log:
        p=subprocess.Popen([str(ROOT/'envs/sim/bin/python'),'-u',str(ROOT/'scripts/cue_probe_worker.py'),str(job_path)],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            rc=p.wait(timeout=300)
            if rc:
                raise RuntimeError(f'Worker exited {rc}; see {log_path}')
        finally:
            try:
                os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass


def main():
    import numpy as np
    from recoverability.snapshot import max_error,digest
    from recoverability.branch_protocol import BranchOutcome,CostConfig,deletion_cost
    parser=argparse.ArgumentParser()
    parser.add_argument('--programs',type=Path,required=True)
    parser.add_argument('--gpu',default='1')
    args=parser.parse_args()
    gate=json.loads((ROOT/'reports/latest_rebuild_validation.json').read_text())
    if gate['status']!='complete' or not gate.get('all_accepted'):
        raise RuntimeError('Fresh-process rebuild gate must pass before collecting pairs')
    program_report=json.loads(args.programs.read_text())
    if program_report['status']!='complete' or not program_report['programs']:
        raise RuntimeError('No complete compiled programs')
    programs=program_report['programs']
    controller_files=['src/recoverability/cue_reader.py','scripts/cue_probe_worker.py',
                      'src/recoverability/snapshot.py','src/recoverability/cue_probe_task.py']
    common_hash=hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in controller_files)).hexdigest()
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_cue_branches')
    run.mkdir(parents=True)
    config=CostConfig(step_horizon=10000,compute_scale_seconds=100,compute_weight=0)
    status=dict(status='running',run_dir=str(run),pairs=[],branches_completed=0,
                dataset_scope='diagnostic_only_not_for_training',cost=asdict(config),
                cost_note='Physics ticks; wall time reported separately, excluded from pilot labels',
                rebuild_validation=gate['run_dir'],controller_source_hash=common_hash)
    def save():
        text=json.dumps(status,indent=2)
        (run/'status.json').write_text(text)
        (ROOT/'reports/latest_cue_branches.json').write_text(text)
    save()
    start=time.monotonic()
    try:
        for program in programs:
            paired_parent_hashes=[]
            for command in ('preserve','erase'):
                branch_results={}
                arrays={}
                # R_repeat checks the actual post-intervention contact continuation.
                for label in ('K','D','R','R_repeat'):
                    branch='R' if label=='R_repeat' else label
                    path=run/f"seed_{program['seed']}_{command}_{label}"
                    job=dict(program=program,gpu=args.gpu,command=command,branch=branch,
                             output=str(path),controller_hash=digest((common_hash,program['stages'])))
                    job_path=path.with_suffix('.job.json')
                    job_path.write_text(json.dumps(job,indent=2))
                    print(json.dumps({'stage':'branch','seed':program['seed'],'command':command,'branch':label}),flush=True)
                    worker(job_path,path.with_suffix('.log'))
                    branch_results[label]=json.loads((path/'outcome.json').read_text())
                    with np.load(path/'evaluator_trace.npz',allow_pickle=False) as data:
                        arrays[label]={k:data[k] for k in data.files}
                    status['branches_completed']+=1
                    save()
                k=branch_results['K']
                required=('parent_state_hash','rng_state_hash','evidence_id','controller_hash','action_context_hash')
                hashes_match=all(all(row[key]==k[key] for key in required) for row in branch_results.values())
                parent_error=max(max_error(arrays['K']['parent'],a['parent']) for a in arrays.values())
                prefix_error=max(max_error(arrays['K']['prefix_trace'],a['prefix_trace']) for a in arrays.values())
                image_error=max(max_error(arrays['K']['parent_head'].astype(float),a['parent_head'].astype(float))
                                for a in arrays.values())
                continuation_error=max_error(arrays['R']['recovery_trace'],arrays['R_repeat']['recovery_trace'])
                repeated_outcome=all(branch_results['R'][key]==branch_results['R_repeat'][key]
                                     for key in ('success','answer','recovery_used','steps'))
                accepted=hashes_match and parent_error<=1e-5 and prefix_error<=1e-3 and image_error==0 and \
                         continuation_error<=1e-3 and repeated_outcome
                restore_error=max(parent_error,prefix_error,continuation_error)
                def outcome(row):
                    keys=('family_id','parent_state_hash','action_context_hash','evidence_id','controller_hash',
                          'rng_state_hash','branch','success','steps','recovery_allowed')
                    return BranchOutcome(**{key:row[key] for key in keys},compute_seconds=0.,
                                         restore_validated=accepted,restore_error=restore_error)
                delta=deletion_cost(outcome(k),outcome(branch_results['R']),config,restore_tolerance=1e-3) if accepted else None
                record=dict(family_id=k['family_id'],seed=program['seed'],command=command,
                            split='diagnostic',training_eligible=False,accepted=accepted,
                            parent_state_hash=k['parent_state_hash'],parent_error=parent_error,
                            prefix_error=prefix_error,image_max_error=image_error,
                            continuation_error=continuation_error,hashes_match=hashes_match,
                            delta=delta,branches={b:branch_results[b] for b in ('K','D','R')})
                status['pairs'].append(record)
                paired_parent_hashes.append(k['parent_state_hash'])
                save()
                if not accepted:
                    raise RuntimeError('Rejected pair: state, observation, RNG or continuation mismatch')
            if len(set(paired_parent_hashes))!=1:
                raise RuntimeError('Action conditions did not share a common pre-intervention parent')
        status.update(status='complete',all_accepted=all(r['accepted'] for r in status['pairs']),
                      elapsed_seconds=time.monotonic()-start)
        save()
        # This namespace intentionally cannot be mistaken for train/validation/test data.
        data_dir=ROOT/'data/diagnostics'/run.name
        data_dir.mkdir(parents=True)
        (data_dir/'pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in status['pairs']))
        (data_dir/'README.txt').write_text('Calibration only. Privileged trajectory compilation; no VLA or learned scorer.\n'
                                          'Not eligible for training or paper success-rate comparisons.\n')
        print(json.dumps({'status':'complete','pairs':len(status['pairs']),'run_dir':str(run)}),flush=True)
    except Exception:
        status.update(status='failed',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
        save()
        raise


if __name__=='__main__':
    main()
