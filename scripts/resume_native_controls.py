"""Resume only unfinished controls, preserving original results and worker code.

The monitor is observational except for stopping this runner's own processes
after sustained resource alarms. It cannot prevent a firmware/driver hard hang.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import traceback

from run_baseline import stop_group

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def resource_sample(gpu):
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemAvailable', 'MemTotal', 'SwapFree', 'SwapTotal'):
            memory[key + '_KiB'] = int(value.split()[0])
    sample = dict(time_utc=datetime.now(timezone.utc).isoformat(), memory=memory,
                  load_average=os.getloadavg(), gpus=[], temperatures=[], alarms=[])
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=6, check=True)
        for line in result.stdout.splitlines():
            fields = [part.strip() for part in line.split(',')]
            item = dict(zip(('index', 'memory_MiB', 'total_MiB', 'utilization', 'temperature_C', 'power_W'), fields))
            sample['gpus'].append(item)
            if int(item['index']) == gpu and float(item['temperature_C']) >= 90:
                sample['alarms'].append('selected_gpu_temperature_at_least_90C')
    except Exception as exc:
        sample['gpu_error'] = repr(exc)
        sample['alarms'].append('gpu_telemetry_unavailable')
    for hwmon in Path('/sys/class/hwmon').glob('hwmon*'):
        try:
            name = (hwmon / 'name').read_text().strip()
            if name != 'coretemp':
                continue
            for sensor in hwmon.glob('temp*_input'):
                value = int(sensor.read_text()) / 1000
                label = sensor.with_name(sensor.name.replace('_input', '_label'))
                sample['temperatures'].append(dict(device=name, label=label.read_text().strip() if label.exists() else sensor.name, celsius=value))
                if value >= 95:
                    sample['alarms'].append('cpu_temperature_at_least_95C')
        except (OSError, ValueError):
            continue
    if memory['MemAvailable_KiB'] < 8 * 1024 * 1024:
        sample['alarms'].append('host_available_memory_below_8GiB')
    sample['alarms'] = sorted(set(sample['alarms']))
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seeds', type=int, nargs='+', default=[100000, 100001])
    parser.add_argument('--conditions', nargs='+', default=['native', 'position', 'color', 'native_repeat'],
                        choices=['native', 'position', 'color', 'native_repeat'])
    args = parser.parse_args()
    previous = args.previous.resolve()
    old = json.loads((previous / 'status.json').read_text())
    lock = (ROOT / 'runs/native_controls_resume.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, expected in old['source_hashes'].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('Source changed since original run: ' + name)
    completed = {}
    for episode in old['episodes']:
        path = Path(episode['result_file'])
        result = json.loads(path.read_text())
        if result['status'] != 'complete':
            raise RuntimeError('Cannot import unfinished result: ' + str(path))
        completed[(episode['seed'], episode['condition'])] = dict(episode,
            reused_from=str(previous), result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    initial = resource_sample(args.gpu)
    chosen = next(row for row in initial['gpus'] if int(row['index']) == args.gpu)
    if initial['alarms'] or float(chosen['memory_MiB']) > 2048 or float(chosen['utilization']) > 10:
        raise RuntimeError('Selected GPU or system is not idle/healthy: ' + json.dumps(initial))
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_native_controls_resumed')
    run.mkdir()
    names = list(old['source_hashes']) + ['scripts/resume_native_controls.py']
    hashes = {}
    for name in names:
        content = (ROOT / name).read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
        dest = run / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               CUDA_HOME='/usr/local/cuda-12.8', TORCH_EXTENSIONS_DIR=str(ROOT / 'cache/torch_extensions'),
               XDG_CACHE_HOME=str(ROOT / 'cache/xdg'), PYTHONPATH=str(ROOT / 'src'), PYTHONUNBUFFERED='1')
    state = dict(status='loading', run_dir=str(run), pid=os.getpid(), gpu=args.gpu, port=port,
                 resumed_from=str(previous), source_hashes=hashes, episodes=list(completed.values()),
                 formal_training_eligible=False, seeds=args.seeds, conditions=args.conditions,
                 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 started_utc=datetime.now(timezone.utc).isoformat(),
                 protocol=old['protocol'], monitoring_interval_seconds=10,
                 guard='Stop own processes after three consecutive alarm samples; not protection against hard hangs.')
    def save():
        atomic_json(run / 'status.json', state)
        atomic_json(ROOT / 'reports/latest_native_controls.json', state)
    stop_monitor = threading.Event()
    abort_monitor = threading.Event()
    monitor_reason = []
    process_ids = {}
    def monitor():
        alarm_streak = 0
        with (run / 'resources.jsonl').open('a', buffering=1) as stream:
            while not stop_monitor.is_set():
                try:
                    sample = resource_sample(args.gpu)
                    sample['owned_processes'] = {}
                    for label, pid in list(process_ids.items()):
                        try:
                            lines = Path(f'/proc/{pid}/status').read_text().splitlines()
                            sample['owned_processes'][label] = dict(pid=pid, fields=[line for line in lines if line.startswith(('VmRSS:', 'VmSwap:', 'Threads:', 'State:'))])
                        except OSError:
                            pass
                    alarm_streak = alarm_streak + 1 if sample['alarms'] else 0
                    sample['consecutive_alarm_samples'] = alarm_streak
                    stream.write(json.dumps(sample) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    if alarm_streak >= 3:
                        monitor_reason[:] = sample['alarms']
                        abort_monitor.set()
                except Exception:
                    monitor_reason[:] = ['resource_monitor_failed: ' + traceback.format_exc()]
                    abort_monitor.set()
                stop_monitor.wait(10)
    def check():
        if abort_monitor.is_set():
            raise RuntimeError('Resource guard stopped run: ' + repr(monitor_reason))
        if server is not None and server.poll() is not None:
            raise RuntimeError('Policy server exited unexpectedly')
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    server = child = None
    thread = threading.Thread(target=monitor, daemon=True)
    save()
    thread.start()
    print(str(run), flush=True)
    try:
        with (run / 'server.log').open('w') as log:
            server = subprocess.Popen([str(ROOT / 'envs/policy/bin/python'), '-u', str(ROOT / 'scripts/seeded_policy_server.py'),
                                       '--port', str(port), '--report', str(run / 'server_ready.json')],
                                      cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        process_ids['policy_server'] = server.pid
        state['server_pid'] = server.pid
        save()
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            check()
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError('Policy server readiness timeout')
        for seed in args.seeds:
            prior_rollout = completed.get((seed, 'rollout'))
            parent = Path(prior_rollout['result_file']).parent if prior_rollout else run / f'{seed}_rollout'
            for condition in ['rollout'] + args.conditions:
                if (seed, condition) in completed:
                    continue
                check()
                if condition != 'rollout' and not (parent / 'parent.json').exists():
                    state.setdefault('skipped', []).append(dict(seed=seed, condition=condition, reason='Native prefix failed'))
                    save()
                    continue
                out = run / f'{seed}_{condition}'
                job = dict(seed=seed, condition=condition, output=str(out), gpu=args.gpu, port=port,
                           max_actions=300, source=None if condition == 'rollout' else str(parent))
                jobpath = out.with_suffix('.job.json')
                atomic_json(jobpath, job)
                state.update(status='evaluating', current_job=str(jobpath))
                save()
                with out.with_suffix('.log').open('w') as log:
                    child = subprocess.Popen([str(ROOT / 'envs/sim/bin/python'), '-u', str(ROOT / 'scripts/native_control_worker.py'), str(jobpath)],
                                             cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    process_ids['worker'] = child.pid
                    state['worker_pid'] = child.pid
                    save()
                    deadline = time.monotonic() + 2400
                    while child.poll() is None:
                        check()
                        if time.monotonic() > deadline:
                            raise TimeoutError('Worker timed out: ' + str(out))
                        time.sleep(2)
                    if child.returncode:
                        raise RuntimeError('Worker failed: ' + str(out))
                    stop_group(child)
                    child = None
                    process_ids.pop('worker', None)
                path = out / 'result.json'
                result = json.loads(path.read_text())
                if result['status'] != 'complete':
                    raise RuntimeError('Worker did not produce complete result')
                state['episodes'].append({key: result[key] for key in ('native_success', 'intended_cover_actuated',
                    'actions_executed', 'parent_match', 'events', 'action_hash', 'final_physics_hash', 'forbidden_calls') if key in result} |
                    dict(seed=seed, condition=condition, result_file=str(path), model_requests=len(result['request_audits']),
                         result_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                save()
        state.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
        state.pop('current_job', None)
        state.pop('worker_pid', None)
        save()
    except BaseException:
        state.update(status='failed', error=traceback.format_exc(), failed_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise
    finally:
        stop_group(child)
        stop_group(server)
        stop_monitor.set()
        thread.join(timeout=10)
        try:
            result = subprocess.run(['journalctl', '-k', '-b', '--since', state['started_utc'], '--no-pager'],
                                    capture_output=True, text=True, timeout=10)
            (run / 'kernel_since_start.log').write_text(result.stdout + result.stderr)
        except Exception as exc:
            (run / 'kernel_since_start.log').write_text(repr(exc))
        state['process_cleanup_completed'] = True
        save()


if __name__ == '__main__':
    main()
