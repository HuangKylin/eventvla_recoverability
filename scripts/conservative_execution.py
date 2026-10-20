"""Wall-clock pacing and telemetry limits; never changes experiment inputs or math."""
import json
import math
from pathlib import Path
import time


def load_profile(root):
    profile=json.loads((Path(root)/'configs/conservative_execution.json').read_text())
    required={'profile','inter_job_rest_seconds','monitor_interval_seconds','consecutive_alarm_limit',
              'gpu_temperature_stop_C','cpu_temperature_stop_C','minimum_host_available_GiB','minimum_gpu_free_GiB'}
    if set(profile)!=required or profile['profile']!='conservative_wall_clock_v1':
        raise ValueError('Unexpected execution profile; numerical/model settings are not accepted here')
    for key,value in profile.items():
        if key!='profile' and (type(value) is not int or value<=0):raise ValueError('Invalid execution limit: '+key)
    return profile


def conservative_sample(sample,gpu,profile):
    result=dict(sample);alarms=list(sample.get('alarms',[]))
    def finite(value):
        number=float(value)
        if not math.isfinite(number):raise ValueError('Nonfinite telemetry')
        return number
    try:
        selected=next(row for row in sample['gpus'] if int(row['index'])==gpu)
        if finite(selected['temperature_C'])>=profile['gpu_temperature_stop_C']:alarms.append('conservative_gpu_temperature')
        if finite(selected['total_MiB'])-finite(selected['memory_MiB'])<profile['minimum_gpu_free_GiB']*1024:
            alarms.append('conservative_gpu_headroom')
    except (KeyError,ValueError,TypeError,StopIteration):alarms.append('selected_gpu_telemetry_unavailable')
    try:
        if finite(sample['memory']['MemAvailable_KiB'])<profile['minimum_host_available_GiB']*1024*1024:
            alarms.append('conservative_host_headroom')
        temperatures=[finite(row['celsius']) for row in sample['temperatures']]
        if not temperatures:raise ValueError('Missing CPU temperature')
        if max(temperatures)>=profile['cpu_temperature_stop_C']:alarms.append('conservative_cpu_temperature')
    except (KeyError,ValueError,TypeError):alarms.append('host_telemetry_unavailable')
    result.update(alarms=sorted(set(alarms)),execution_profile=profile['profile'])
    return result


def rest_between_jobs(seconds,check,*,clock=time.monotonic,sleep=time.sleep):
    """Called with no simulation worker alive; no simulated time or RNG is consumed."""
    until=clock()+seconds
    while True:
        check()
        remaining=until-clock()
        if remaining<=0:return
        sleep(min(2,remaining))
