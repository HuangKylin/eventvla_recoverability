"""CPU-only reconstruction of the first input from a trusted parent snapshot."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('parent', type=Path)
    args = parser.parse_args(); source = args.parent.resolve()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(ROOT / 'src'))
    sys.path.insert(0, str(ROOT / 'third_party/EventVLA/EventVLA'))
    import numpy as np
    from PIL import Image, ImageDraw
    from recoverability.vla_intervention import audit_payload, purge_evidence
    path = ROOT / 'third_party/EventVLA/EventVLA/examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py'
    spec = importlib.util.spec_from_file_location('offline_client', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model = module.ModelClient.__new__(module.ModelClient)
    model.__dict__.update(pickle.loads((source / 'client_state.pkl').read_bytes()))
    parent = json.loads((source / 'parent.json').read_text()); step = parent['step']
    if not model.pending_commit_done and model._pending_commit_step() == step:
        raise ValueError('Pending commit at first step requires explicit reconstruction')
    views = ('head_camera', 'left_camera', 'right_camera')
    images = [model._resize_image(np.asarray(Image.open(source/f'parent_{v}.png'))) for v in views]
    model._record_temporal_images(images=images, step=step)
    anchors, metas = model._build_temporal_anchor_images(step)
    example = dict(image=anchors, image_metas=metas)
    model._inject_keyframe_memory_fields(example, step)
    audit = audit_payload(dict(examples=[example]), ())
    native = source.parent / (str(parent['seed'])+'_native') / 'result.json'
    if not native.exists(): raise ValueError('Need actual first native request for input verification')
    result = json.loads(native.read_text())
    actual = result['request_audits'][0]
    if audit['images'] != actual['images'] or actual['step'] != step:
        raise ValueError('Reconstructed images differ from actual first request')
    deletion_checks = []
    event_steps = list(example['memory_keyframe_steps'])
    for removed in [[]] + [[s] for s in event_steps] + [event_steps]:
        variant = module.ModelClient.__new__(module.ModelClient)
        variant.__dict__.update(pickle.loads((source / 'client_state.pkl').read_bytes()))
        if variant.action_mode != 'abs':
            raise ValueError('Clearing cross-chunk action state requires separate non-absolute-mode validation')
        intervention = purge_evidence(variant, removed)
        variant._record_temporal_images(images=images, step=step)
        vimages, vmetas = variant._build_temporal_anchor_images(step)
        vexample = dict(image=vimages, image_metas=vmetas)
        variant._inject_keyframe_memory_fields(vexample, step)
        vaudit = audit_payload(dict(examples=[vexample]), intervention['forbidden_image_hashes'])
        expected = [s for s in event_steps if s not in removed]
        if vexample['memory_keyframe_steps'] != expected:
            raise ValueError('Deleted timestep remains or retained timestep changed')
        anchor_records = lambda a: [x for x in a['images'] if x['channel'] == 'image']
        if anchor_records(vaudit) != anchor_records(audit):
            raise ValueError('Deleting event memories changed temporal anchors')
        if not removed and vaudit['images'] != audit['images']:
            raise ValueError('Keep intervention changed model input images')
        deletion_checks.append(dict(deleted_steps=list(removed), retained_event_steps=expected,
            anchor_images=vaudit['anchor_images'], event_images=vaudit['event_images'],
            image_payload_bytes=vaudit['image_payload_bytes'], forbidden_hits=vaudit['forbidden_hits'],
            anchor_images_unchanged=True, action_cache_cleared=intervention['action_cache_cleared'],
            pending_commit_cleared=intervention['pending_commit_cleared']))
    output = source/'input_audit'; output.mkdir(exist_ok=True)
    def sheet(name, images, metas):
        width, height = images[0].shape[1], images[0].shape[0]
        canvas = Image.new('RGB', (width*3, (height+24)*((len(images)+2)//3)), 'white')
        draw = ImageDraw.Draw(canvas)
        for i,(image,meta) in enumerate(zip(images, metas)):
            x=(i%3)*width;y=(i//3)*(height+24)
            canvas.paste(Image.fromarray(image),(x,y+24))
            draw.text((x+3,y+4),str(meta.get('view',i))+' '+str(meta.get('time_role','event'))+' '+str(meta.get('delta',meta.get('step',''))),fill='black')
        canvas.save(output/name)
    sheet('temporal_anchors.png',anchors,metas)
    memory = example['memory_keyframe_images']
    sheet('event_images.png', memory, example['memory_keyframe_image_metas'])
    report=dict(status='verified', parent_step=step, actual_request_images_match=True,
        anchor_metas=metas, event_metas=example['memory_keyframe_image_metas'],
        event_steps=example['memory_keyframe_steps'], audit=audit, deletion_checks=deletion_checks,
        formal_training_eligible=False,
        note='Exact first-input reconstruction and offline deletion audit only; no post-deletion policy inference or recovery; not K/R labels.')
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(output=str(output), parent_step=step, actual_request_images_match=True)))


if __name__ == '__main__': main()
