"""Build the pinned cuRobo release inside the project's simulation environment."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / 'envs/sim/bin/python'
UV = '/home/user/.local/bin/uv'
ENV = os.environ.copy()
ENV.update(CUDA_HOME='/usr/local/cuda-12.8', TORCH_CUDA_ARCH_LIST='12.0',
           MAX_JOBS='2', CUDA_VISIBLE_DEVICES='1', UV_HTTP_TIMEOUT='300',
           UV_CACHE_DIR=str(ROOT / 'cache/uv'),
           TORCH_EXTENSIONS_DIR=str(ROOT / 'cache/torch_extensions'),
           PATH='/usr/local/cuda-12.8/bin:' + ENV.get('PATH', ''))


def state(stage, **extra):
    data = dict(stage=stage, pid=os.getpid(),
                time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    (ROOT / 'reports/sim_extensions_status.json').write_text(json.dumps(data, indent=2))
    print(json.dumps(data), flush=True)


def run(args):
    print('RUN', args, flush=True)
    subprocess.run(args, env=ENV, check=True)


def main():
    state('waiting_for_base_environment', status='running')
    deadline = time.monotonic() + 2700
    while time.monotonic() < deadline:
        current = json.loads((ROOT / 'reports/environment_install_status.json').read_text())
        if current['state'] == 'failed':
            raise RuntimeError('Base environment installation failed')
        if current['state'] == 'complete':
            break
        time.sleep(5)
    else:
        raise TimeoutError('Base environment was not ready within 45 minutes')
    state('extension_dependencies', status='running')
    run([UV, 'pip', 'install', '--python', str(PYTHON),
         'setuptools_scm', 'pybind11', 'numpy==1.26.4', 'scipy==1.10.1',
         'numpy-quaternion==2023.0.4', 'yourdfpy>=0.0.53', 'warp-lang==1.12.0',
         'scikit-image==0.24.0', 'opencv-python==4.11.0.86', 'importlib_resources'])
    state('curobo_build', status='running')
    run([UV, 'pip', 'install', '--python', str(PYTHON), '--no-build-isolation',
         '--no-deps', '--editable', str(ROOT / 'third_party/curobo')])
    state('complete', status='complete')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        state('failed', status='failed', error=traceback.format_exc())
        raise
