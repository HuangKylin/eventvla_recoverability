"""Audit slot interventions through the actual upstream client, without GPU inference."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'third_party/EventVLA/EventVLA'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parents',type=Path,required=True)
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    from PIL import Image
    from recoverability.slot_protocol import CONDITIONS,PROTOCOL,apply_condition,parent_randomness_key,paired_request_seed
    from recoverability.vla_intervention import audit_payload
    source_file=ROOT/'third_party/EventVLA/EventVLA/examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py'
    spec=importlib.util.spec_from_file_location('slot_audit_client',source_file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    manifest=json.loads(args.parents.read_text());checks={};rows=[]
    for seed in (410000,410001):
        source=Path(next(e['result_file'] for e in manifest['episodes'] if e['seed']==seed and e['condition']=='prepare')).parent
        parent=json.loads((source/'parent.json').read_text());step=parent['step']
        actual=json.loads((source/'input_audit/report.json').read_text())
        for condition in CONDITIONS:
            model=module.ModelClient.__new__(module.ModelClient)
            with (source/'client_state.pkl').open('rb') as stream:model.__dict__.update(pickle.load(stream))
            intervention=apply_condition(model,condition)
            images=[model._resize_image(np.asarray(Image.open(source/f'parent_{v}.png'))) for v in ('head_camera','left_camera','right_camera')]
            model._record_temporal_images(images,step)
            anchors,metas=model._build_temporal_anchor_images(step)
            example=dict(image=anchors,image_metas=metas);model._inject_keyframe_memory_fields(example,step)
            audit=audit_payload(dict(examples=[example]),intervention['forbidden_image_hashes'])
            channels=lambda records,name:[x for x in records if x['channel']==name]
            before=channels(actual['audit']['images'],'memory_keyframe_images');after=channels(audit['images'],'memory_keyframe_images')
            candidate=intervention['specification']['candidate_index'];prefix=f'{seed}_{condition}_'
            checks[prefix+'four_images']=audit['event_images']==4
            checks[prefix+'temporal_unchanged']=channels(audit['images'],'image')==channels(actual['audit']['images'],'image')
            checks[prefix+'slot_metadata_unchanged']=example['memory_keyframe_steps']==actual['event_steps'] and example['memory_keyframe_image_metas']==actual['event_metas']
            checks[prefix+'only_candidate_changes']=[i for i,(a,b) in enumerate(zip(before,after)) if a!=b]==([] if candidate is None else [candidate])
            checks[prefix+'empty_has_no_old_pixels']=audit['forbidden_hits']==0
            checks[prefix+'equal_cache_reset']=intervention['action_cache_cleared'] and intervention['pending_commit_cleared']
            key=parent_randomness_key(parent)
            rows.append(dict(seed=seed,condition=condition,specification=intervention['specification'],
                event_images=audit['event_images'],first_input=audit,
                request_index_seed_preview=[paired_request_seed(key,i) for i in range(4)]))
    report=dict(status='passed' if all(checks.values()) else 'failed',protocol=PROTOCOL,
                checks=checks,variants=rows,actual_upstream_client=True,policy_inference_executed=False,
                formal_training_eligible=False)
    path=ROOT/'reports/native_slot_input_audit.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],checks=len(checks),variants=len(rows),report=str(path))))
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
