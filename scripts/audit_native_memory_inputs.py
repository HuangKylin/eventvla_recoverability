"""Validate all six first requests without connecting a model or creating a scene."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('previous', type=Path)
    args = parser.parse_args(); os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'third_party/EventVLA/EventVLA'))
    import numpy as np
    from PIL import Image
    from recoverability.native_memory import intervene
    from recoverability.vla_intervention import audit_payload
    previous = json.loads(args.previous.read_text())
    sources = {e['seed']: Path(e['result_file']).parent for e in previous['episodes'] if e['condition'] == 'prepare'}
    seeds = sorted(sources)
    if len(seeds) != 2: raise ValueError('Expected two parent seeds')
    path = ROOT/'third_party/EventVLA/EventVLA/examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py'
    spec = importlib.util.spec_from_file_location('memory_offline_client', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rows = []
    for seed in seeds:
        source = sources[seed]; donor_source = sources[next(s for s in seeds if s != seed)]
        parent = json.loads((source/'parent.json').read_text())
        baseline = json.loads((source/'input_audit/report.json').read_text())
        for condition in ['keep', 'drop_all', 'mismatch']:
            model = module.ModelClient.__new__(module.ModelClient)
            model.__dict__.update(pickle.loads((source/'client_state.pkl').read_bytes()))
            donor = pickle.loads((donor_source/'client_state.pkl').read_bytes())['keyframe_image_memory']
            report = intervene(model, condition, donor)
            images = [model._resize_image(np.asarray(Image.open(source/f'parent_{v}.png')))
                      for v in ['head_camera', 'left_camera', 'right_camera']]
            model._record_temporal_images(images=images, step=parent['step'])
            anchors, metas = model._build_temporal_anchor_images(parent['step'])
            example = dict(image=anchors, image_metas=metas)
            model._inject_keyframe_memory_fields(example, parent['step'])
            audit = audit_payload(dict(examples=[example]), report['forbidden_image_hashes'])
            get_anchors = lambda a: [r for r in a['images'] if r['channel'] == 'image']
            if get_anchors(audit) != get_anchors(baseline['audit']): raise RuntimeError('Anchor mismatch')
            events = [r['sha256'] for r in audit['images'] if r['channel'] == 'memory_keyframe_images']
            if events != report['expected_event_image_hashes']: raise RuntimeError('Event image mismatch')
            if example['memory_keyframe_steps'] != ([] if condition == 'drop_all' else parent['event_steps']):
                raise RuntimeError('Event steps changed')
            if example['memory_keyframe_image_metas'] != ([] if condition == 'drop_all' else baseline['event_metas']):
                raise RuntimeError('Event metadata changed')
            if condition == 'keep' and audit['images'] != baseline['audit']['images']:
                raise RuntimeError('Keep differs from actual preceding native request')
            rows.append(dict(seed=seed, condition=condition, input_audit=audit, intervention=report,
                             temporal_anchors_unchanged=True, event_metadata_verified=True))
    result = dict(status='verified', rows=rows, formal_training_eligible=False)
    output = ROOT/'reports/native_memory_input_preflight.json'
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(status='verified', variants=len(rows), output=str(output))))


if __name__ == '__main__': main()
