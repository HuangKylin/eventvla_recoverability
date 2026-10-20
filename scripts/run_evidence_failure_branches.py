"""One native failure, one exact replay, two predeclared content probes, then stop."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from conservative_execution import load_profile, conservative_sample, rest_between_jobs
from resume_native_controls import atomic_json, resource_sample
from run_baseline import stop_group
from run_evidence_diagnostic import sha
from recoverability.evidence_branch_resume import prepare_resume


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--resume',type=Path)
    a=p.parse_args()
    manifest=json.loads(a.manifest.read_text())
    source=(ROOT/manifest['source']).resolve()
    census=json.loads((source.parent/'status.json').read_text())
    if census['status']!='complete' or not census['process_cleanup_completed']:
        raise ValueError('Native batch must be complete and cleaned')
    failures=[x for x in census['episodes'] if not x['native_success']]
    if len(failures)!=1 or Path(failures[0]['result_file']).parent!=source:
        raise ValueError('Manifest must select the unique recorded native failure')
    if sha(source/'result.json')!=failures[0]['result_sha256']:
        raise ValueError('Native result changed')
    for name,value in census['source_hashes'].items():
        if sha(ROOT/name)!=value:raise ValueError('Frozen native source changed: '+name)
    if sha(Path(manifest['checkpoint']))!=census['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    parent=json.loads((source/'parent.json').read_text())
    branches=manifest['branches']
    if len(branches)!=3 or branches[0]['name']!='unchanged' or branches[0]['replacements']:
        raise ValueError('Require exact replay followed by two probes')
    if manifest['training_allowed'] or manifest['task_expansion_allowed']:
        raise ValueError('Scope cannot expand')
    lock=(ROOT/'runs/native_controls_resume.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inherited,ancestors=prepare_resume(ROOT,a.resume,source,manifest) if a.resume else ([],set())
    for old in (ROOT/'runs').glob('*_evidence_failure_branches/status.json'):
        if json.loads(old.read_text()).get('source')==str(source) and old.parent.resolve() not in ancestors:
            raise ValueError('Another batch exists for this parent; resume the latest recorded batch explicitly')
    audit_env=os.environ.copy()
    audit_env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for row in inherited:
        jobpath=Path(row['branch_dir']).with_suffix('.job.json')
        subprocess.run([str(ROOT/'envs/sim/bin/python'),str(ROOT/'scripts/audit_evidence_failure_branch.py'),str(jobpath)],
            cwd=ROOT,env=audit_env,check=True,timeout=180)
    profile=load_profile(ROOT)
    item=conservative_sample(resource_sample(a.gpu),a.gpu,profile)
    gpu=next(g for g in item['gpus'] if int(g['index'])==a.gpu)
    if item['alarms'] or float(gpu['memory_MiB'])>2048 or float(gpu['utilization'])>10:
        raise RuntimeError('GPU/system unavailable: '+json.dumps(item))
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_evidence_failure_branches')
    run.mkdir()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    atomic_json(run/'selection_manifest.json',manifest)
    additions={}
    for name in ['scripts/run_evidence_failure_branches.py','scripts/audit_evidence_failure_branch.py',
                 'src/recoverability/evidence_branch_audit.py','scripts/pace_evidence_census.py',
                 'src/recoverability/evidence_branch_resume.py']:
        dest=run/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes());additions[name]=sha(dest)
    state=dict(status='loading',pid=os.getpid(),gpu=a.gpu,port=port,run_dir=str(run),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source=str(source),
        started_utc=datetime.now(timezone.utc).isoformat(),episodes=inherited,source_hashes=additions,
        manifest_sha256=sha(run/'selection_manifest.json'),formal_training_eligible=False,
        process_cleanup_completed=False)
    if a.resume:
        state['resumed_from']=str(a.resume.resolve())
        state['inherited_completed_branches']=[x['name'] for x in inherited]
    def save():
        atomic_json(run/'status.json',state)
        atomic_json(ROOT/'reports/latest_evidence_failure_branches.json',state)
    server=child=pacer=None
    alarms=0;last_sample=0
    resources=(run/'resources.jsonl').open('a',buffering=1)
    def check():
        nonlocal alarms,last_sample
        if time.monotonic()-last_sample>=profile['monitor_interval_seconds']:
            sample=conservative_sample(resource_sample(a.gpu),a.gpu,profile)
            resources.write(json.dumps(sample)+'\n');resources.flush();os.fsync(resources.fileno())
            alarms=alarms+1 if sample['alarms'] else 0;last_sample=time.monotonic()
            if alarms>=profile['consecutive_alarm_limit']:raise RuntimeError('Resource guard')
        if server and server.poll() is not None:raise RuntimeError('Policy server exited')
        if pacer and pacer.poll() is not None:raise RuntimeError('Cooling process exited')
    def interrupted(signum,frame):raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        CUDA_HOME='/usr/local/cuda-12.8',TORCH_EXTENSIONS_DIR=str(ROOT/'cache/torch_extensions'),
        XDG_CACHE_HOME=str(ROOT/'cache/xdg'),PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        EVENTVLA_RAY_DENOISER='none',EVENTVLA_RT_SPP='128',TOKENIZERS_PARALLELISM='false')
    env.pop('RECOVERABILITY_MEMORY_POLICY',None)
    save();print(run,flush=True)
    try:
        with (run/'server.log').open('w') as log:
            server=subprocess.Popen([str(ROOT/'envs/policy/bin/python'),'-u',str(ROOT/'scripts/seeded_policy_server.py'),
                '--port',str(port),'--report',str(run/'server_ready.json')],cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['server_pid']=server.pid;save()
        with (run/'cooling.log').open('w') as log:
            pacer=subprocess.Popen([sys.executable,str(ROOT/'scripts/pace_evidence_census.py'),'--run',str(run)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['pacer_pid']=pacer.pid;save()
        deadline=time.monotonic()+600
        while not (run/'server_ready.json').exists():
            check()
            if time.monotonic()>deadline:raise TimeoutError('Model loading')
            time.sleep(1)
        while True:
            check()
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=1):break
            except OSError:
                if time.monotonic()>deadline:raise TimeoutError('Policy socket')
                time.sleep(1)
        for spec in branches:
            if any(row['name']==spec['name'] for row in inherited):continue
            if state['episodes']:
                state['status']='inter_job_rest';save()
                rest_between_jobs(profile['inter_job_rest_seconds'],check)
            out=run/spec['name']
            job=dict(task=parent['task'],seed=parent['seed'],gpu=a.gpu,port=port,output=str(out),
                source=str(source),mode='paired',replacements=spec['replacements'])
            jp=out.with_suffix('.job.json');atomic_json(jp,job)
            state.update(status='evaluating',current_job=str(jp));save()
            with out.with_suffix('.log').open('w') as log:
                child=subprocess.Popen([str(ROOT/'envs/sim/bin/python'),'-u',str(ROOT/'scripts/evidence_diagnostic_worker.py'),str(jp)],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['worker_pid']=child.pid;save();deadline=time.monotonic()+3600
                while child.poll() is None:
                    check()
                    if time.monotonic()>deadline:raise TimeoutError('Worker deadline')
                    heartbeat=out/'result.json'
                    if heartbeat.exists() and time.time()-heartbeat.stat().st_mtime>240:raise TimeoutError('Worker heartbeat')
                    time.sleep(2)
                if child.returncode:raise RuntimeError('Worker failed: '+str(out))
            stop_group(child);child=None
            audit_env=env|{'CUDA_VISIBLE_DEVICES':''}
            with (run/'audit.log').open('a') as log:
                audit=subprocess.run([str(ROOT/'envs/sim/bin/python'),str(ROOT/'scripts/audit_evidence_failure_branch.py'),str(jp)],
                    cwd=ROOT,env=audit_env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            if audit.returncode:raise RuntimeError('Paired audit failed: '+str(out))
            result=json.loads((out/'result.json').read_text())
            verified=json.loads((out/'paired_audit.json').read_text())
            state['episodes'].append(dict(name=spec['name'],native_success=result['native_success'],
                native_failed=result['native_failed'],actions=result['actions_executed'],
                final_step=result['env_step'],final_state=result['final_evaluator_state'],
                result_sha256=sha(out/'result.json'),audit=verified,branch_dir=str(out)))
            save()
        state.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
                     next_action='interpret bounded probes; no automatic expansion or training');save()
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());save();raise
    finally:
        stop_group(pacer);stop_group(child);stop_group(server)
        state['process_cleanup_completed']=True;save();resources.close()


if __name__=='__main__':main()
