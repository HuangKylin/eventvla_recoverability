"""Verify downloaded model completeness and finite action inference on real RGB."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'third_party/EventVLA/EventVLA'
sys.path.insert(0, str(CODE))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
REPORT = ROOT / 'reports/smoke_policy.json'


def main():
    import numpy as np
    import torch
    import yaml
    from eventvla.model.framework.base_framework import baseframework
    result = dict(status='running', pid=os.getpid(), stage='resource_validation')

    def update(**values):
        result.update(values)
        REPORT.write_text(json.dumps(result, indent=2))
        print(json.dumps(values), flush=True)

    try:
        for group in ['policy', 'backbone']:
            state = json.loads((ROOT / 'reports' / ('download_' + group + '_status.json')).read_text())
            if state['status'] != 'complete':
                raise RuntimeError('Unverified resource: ' + group)
        original = ROOT / 'checkpoints/EventVLA/RoboTwin-MeM'
        runtime = ROOT / 'checkpoints/runtime/EventVLA-RoboTwin-MeM'
        (runtime / 'final_model').mkdir(parents=True, exist_ok=True)
        config = yaml.safe_load((original / 'config.yaml').read_text())
        config['framework']['qwenvl']['base_vlm'] = str(ROOT / 'checkpoints/Qwen3-VL-4B-Instruct')
        config['framework']['qwenvl']['attn_implementation'] = 'sdpa'
        (runtime / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        (runtime / 'dataset_statistics.json').write_bytes((original / 'dataset_statistics.json').read_bytes())
        ckpt = runtime / 'final_model/pytorch_model.pt'
        if not ckpt.exists():
            ckpt.symlink_to(original / 'final_model/pytorch_model.pt')
        update(stage='load_model', checkpoint=str(ckpt), attention='sdpa',
               torch_version=torch.__version__, cuda_version=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0),
               checkpoint_history_capacity=config['framework']['memory_buffer']['qwen_memory_injection']['max_keyframe_images'])
        start = time.monotonic()
        model = baseframework.from_pretrained(str(ckpt), strict_checkpoint=True)
        model = model.to(device='cuda', dtype=torch.bfloat16).eval()
        torch.cuda.synchronize()
        update(load_seconds=time.monotonic()-start,
               checkpoint_load_report=model.checkpoint_load_report)
        sample = np.load(ROOT / 'reports/smoke_cover_blocks_hard_observation.npz')
        current = [sample['head'], sample['left'], sample['right']]
        trials = []
        for count in [0, config['framework']['memory_buffer']['qwen_memory_injection']['max_keyframe_images']]:
            ex = dict(image=current * 4, lang='Inspect the blocks and open the covers in red, green, blue, yellow order.',
                      state=sample['state'], memory_keyframe_images=[sample['head']] * count,
                      memory_keyframe_steps=list(range(-count, 0)))
            update(stage='inference', history_images=count)
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            out = model.predict_action(examples=[ex])
            torch.cuda.synchronize()
            shapes = {}
            for name, val in out.items():
                if isinstance(val, np.ndarray) or torch.is_tensor(val):
                    a = val.detach().float().cpu().numpy() if torch.is_tensor(val) else val
                    if np.issubdtype(a.dtype, np.number) and not np.isfinite(a).all():
                        raise ValueError('Non-finite model output: ' + name)
                    shapes[name] = list(a.shape)
            trials.append(dict(history_images=count, seconds=time.monotonic()-start,
                               peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                               output_shapes=shapes))
        update(status='complete', stage='finished', trials=trials,
               scope='Inference smoke only; repeated real frames test the input pipeline, not task success')
    except Exception:
        update(status='failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
