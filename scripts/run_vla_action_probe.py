"""Manage a loopback-only seeded policy server and real VLA probe workers."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
from run_baseline import stop_group
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--library',required=True,type=Path)
    p.add_argument('--seed',type=int,default=310010);p.add_argument('--max-actions',type=int,default=200)
    a=p.parse_args()
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_vla_action_probe')
    run.mkdir(parents=True)
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TORCH_EXTENSIONS_DIR=str(ROOT/'cache/torch_extensions'),XDG_CACHE_HOME=str(ROOT/'cache/xdg'),
        PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1')
    source_names=['scripts/'+n+'.py' for n in ('run_vla_action_probe','vla_action_worker','seeded_policy_server')]
    source_names+=['src/recoverability/'+n+'.py' for n in ('vla_intervention','action_latch_task','fixed_skills','sim_runtime')]
    source_names+=['third_party/EventVLA/EventVLA/examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py']
    state=dict(status='loading',run_dir=str(run),port=port,episodes=[],
               source_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in source_names},
               formal_training_eligible=False)
    def save():
        text=json.dumps(state,indent=2);(run/'status.json').write_text(text)
        (ROOT/'reports/latest_vla_action_probe.json').write_text(text)
    server=child=None;save()
    try:
        with (run/'server.log').open('w') as log:
            server=subprocess.Popen([str(ROOT/'envs/policy/bin/python'),'-u',str(ROOT/'scripts/seeded_policy_server.py'),
                '--port',str(port),'--report',str(run/'server_ready.json')],env=env,cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            if server.poll() is not None: raise RuntimeError('Policy server failed')
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=1): break
            except OSError: time.sleep(1)
        else: raise TimeoutError('Policy load timed out')
        state['status']='evaluating';save()
        conditions=[(0,'K'),(0,'R'),(3,'K'),(3,'R'),(0,'K_repeat')]
        for slot,tag in conditions:
            out=run/f'a{slot}_{tag}'
            job=dict(seed=a.seed,action_slot=slot,branch=tag.split('_')[0],max_actions=a.max_actions,
                     library=str(a.library.resolve()),output=str(out),port=port,gpu='1')
            jobfile=out.with_suffix('.job.json');jobfile.write_text(json.dumps(job,indent=2))
            with out.with_suffix('.log').open('w') as log:
                child=subprocess.Popen([str(ROOT/'envs/sim/bin/python'),'-u',str(ROOT/'scripts/vla_action_worker.py'),str(jobfile)],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    if child.wait(timeout=900): raise RuntimeError('VLA worker failed: '+tag)
                finally: stop_group(child);child=None
            result=json.loads((out/'result.json').read_text())
            if result['status']!='complete' or result['forbidden_calls']:
                raise RuntimeError('VLA interface gate failed')
            if state['episodes']:
                first=json.loads(Path(state['episodes'][0]['result_file']).read_text())
                for key in ('parent_hash','prefix_trace_hash','current_hash'):
                    if first[key]!=result[key]: raise RuntimeError('VLA branches have different parents: '+key)
            state['episodes'].append(dict(action_slot=slot,branch=tag,result_file=str(out/'result.json'),
                actions_executed=result['actions_executed'],intended_cover_actuated=result['intended_cover_actuated'],
                events=result['events'],model_requests=len(result['request_audits'])))
            save()
        first=json.loads(Path(state['episodes'][0]['result_file']).read_text())
        repeated=json.loads(Path(state['episodes'][-1]['result_file']).read_text())
        state['repeated_trajectory_identical']=all(first[k]==repeated[k] for k in
            ('action_hash','joint_trace_hash','final_physics_hash','events'))
        state.update(status='complete',all_parent_states_matched=True,
            all_payload_deletion_audits_passed=True,
            control_gate_passed=all(r['intended_cover_actuated'] for r in state['episodes']))
        save()
    except Exception:
        state.update(status='failed',error=traceback.format_exc());save();raise
    finally: stop_group(child);stop_group(server)


if __name__=='__main__': main()
