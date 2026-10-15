"""Install isolated, Blackwell-compatible environments; retain logs and status."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[1]
UV = Path('/home/user/.local/bin/uv')
ENV = os.environ.copy()
ENV.update(UV_CACHE_DIR=str(ROOT / 'cache/uv'), UV_HTTP_TIMEOUT='300',
           UV_LINK_MODE='hardlink', CUDA_HOME='/usr/local/cuda-12.8',
           MAX_JOBS='8', TORCH_CUDA_ARCH_LIST='12.0',
           PATH='/usr/local/cuda-12.8/bin:' + ENV.get('PATH', ''))
STATE = ROOT / 'reports/environment_install_status.json'


def status(stage, state, **extra):
    payload = dict(stage=stage, state=state, pid=os.getpid(),
                   time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(STATE)
    print(json.dumps(payload), flush=True)


def install(name, packages, extra=()):
    py = ROOT / 'envs' / name / 'bin/python'
    cmd = [str(UV), 'pip', 'install', '--python', str(py), *extra, *packages]
    print('RUN', cmd, flush=True)
    subprocess.run(cmd, check=True, env=ENV)


def main():
    for name in ['policy', 'sim']:
        status('torch_' + name, 'running')
        install(name, ['torch==2.7.1', 'torchvision==0.22.1'],
                ['--index-url', 'https://download.pytorch.org/whl/cu128'])
    status('policy_dependencies', 'running')
    install('policy', [
        'pip', 'setuptools==80.9.0', 'wheel', 'packaging', 'ninja',
        'transformers==4.57.0', 'accelerate==1.5.2', 'numpy==1.26.4',
        'tiktoken', 'einops', 'scipy', 'pillow', 'tensorboard', 'matplotlib',
        'websocket-client==1.8.0', 'websockets', 'msgpack', 'msgpack-numpy',
        'pydantic==2.10.6', 'pyarrow==14.0.1', 'av==12.3.0',
        'qwen-vl-utils', 'omegaconf', 'wandb', 'rich', 'diffusers', 'timm',
        'tyro', 'transformers_stream_generator==0.0.4',
        'huggingface-hub==0.36.2', 'safetensors', 'pytest'
    ])
    install('policy', [str(ROOT / 'third_party/EventVLA/EventVLA')], ['--editable'])
    status('sim_dependencies', 'running')
    install('sim', [
        'pip', 'setuptools==80.9.0', 'wheel', 'packaging', 'ninja',
        'numpy==1.26.4', 'scipy==1.10.1', 'transforms3d==0.4.2',
        'sapien==3.0.0b1', 'mplib==0.2.1', 'gymnasium==0.29.1',
        'trimesh==4.4.3', 'open3d==0.18.0', 'imageio==2.34.2',
        'pydantic', 'zarr<3', 'h5py', 'pyglet<2', 'wandb', 'moviepy<2',
        'termcolor', 'av', 'matplotlib', 'ffmpeg-python', 'toppra',
        'pyyaml', 'huggingface-hub==0.36.2', 'requests', 'pytest',
        'websocket-client==1.8.0', 'websockets', 'msgpack', 'msgpack-numpy',
        'omegaconf', 'rich', 'accelerate==1.5.2'
    ])
    for name in ['policy', 'sim']:
        freeze = subprocess.check_output([str(UV), 'pip', 'freeze', '--python',
                    str(ROOT / 'envs' / name / 'bin/python')], env=ENV, text=True)
        (ROOT / 'reports' / (name + '_installed.txt')).write_text(freeze)
    status('base_environments', 'complete',
           pending=['cuRobo build', 'runtime imports', 'rendering', 'model inference'])


if __name__ == '__main__':
    try:
        main()
    except Exception:
        status('installation', 'failed', error=traceback.format_exc())
        raise
