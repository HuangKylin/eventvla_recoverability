from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from conservative_execution import load_profile,conservative_sample,rest_between_jobs

PROFILE=load_profile(Path(__file__).resolve().parents[1])


def healthy():
    return dict(gpus=[dict(index='1',temperature_C='50',total_MiB='73415',memory_MiB='16000')],
                memory=dict(MemAvailable_KiB=120*1024*1024),temperatures=[dict(celsius=40)],alarms=[])


def test_observation_does_not_modify_raw_sample_and_keeps_existing_alarms():
    sample=healthy();sample['alarms']=['existing_fault'];before=deepcopy(sample)
    result=conservative_sample(sample,1,PROFILE)
    assert result['alarms']==['existing_fault'] and sample==before


@pytest.mark.parametrize('kind',('gpu_heat','cpu_heat','host_memory','gpu_memory','missing_gpu','invalid_gpu'))
def test_guard_rejects_resource_pressure_or_unknown_measurements(kind):
    sample=healthy()
    if kind=='gpu_heat':sample['gpus'][0]['temperature_C']='75'
    if kind=='cpu_heat':sample['temperatures'][0]['celsius']=85
    if kind=='host_memory':sample['memory']['MemAvailable_KiB']=31*1024*1024
    if kind=='gpu_memory':sample['gpus'][0]['memory_MiB']='70000'
    if kind=='missing_gpu':sample['gpus']=[]
    if kind=='invalid_gpu':sample['gpus'][0]['temperature_C']='NaN'
    assert conservative_sample(sample,1,PROFILE)['alarms']


def test_job_rest_checks_during_interval_and_preserves_failure():
    now=[0.];checks=[]
    rest_between_jobs(60,lambda:checks.append(now[0]),clock=lambda:now[0],sleep=lambda n:now.__setitem__(0,now[0]+n))
    assert now[0]==60 and checks[0]==0 and checks[-1]==60 and max(b-a for a,b in zip(checks,checks[1:]))<=2
    def fail():raise RuntimeError('Stop own job')
    with pytest.raises(RuntimeError):rest_between_jobs(60,fail,clock=lambda:now[0],sleep=lambda n:None)


def test_execution_profile_cannot_smuggle_model_settings(tmp_path):
    import json
    (tmp_path/'configs').mkdir();profile=dict(PROFILE,inference_precision='int8')
    (tmp_path/'configs/conservative_execution.json').write_text(json.dumps(profile))
    with pytest.raises(ValueError):load_profile(tmp_path)
