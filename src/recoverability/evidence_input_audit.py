"""Validate actual native policy images against their delivered RGB archive."""
from functools import lru_cache
from pathlib import Path
from recoverability.evidence_diagnostic import inference_seed


def audit_native_payload(payload, observation_root, instruction, episode_seed, request_index):
    import numpy as np
    from PIL import Image
    if payload['paired_seed'] != inference_seed(episode_seed, request_index):
        raise ValueError('Wrong request-index RNG seed')
    if len(payload['examples']) != 1:
        raise ValueError('Unexpected batch size')
    example=payload['examples'][0]
    if example['lang'] != instruction:
        raise ValueError('Native instruction changed')
    if any(k in example for k in ('evaluator_state','actual_first_block_id','actual_first_visit_side_ids')):
        raise ValueError('Evaluator metadata leaked')
    t=int(example['timestep'])
    if t<0:raise ValueError('Invalid current time')
    root=Path(observation_root)
    views=('head_camera','left_camera','right_camera')
    checked=[]
    @lru_cache(maxsize=64)
    def expected_image(step,view,width,height):
        with Image.open(root/f'{step:06d}'/f'{view}.png') as image:
            return np.asarray(image.resize((width,height)))
    for channel,meta_field in [('image','image_metas'),('memory_keyframe_images','memory_keyframe_image_metas')]:
        images,metas=example.get(channel,[]),example.get(meta_field,[])
        if len(images)!=len(metas):raise ValueError('Image/source metadata count mismatch')
        for im,meta in zip(images,metas):
            if channel=='memory_keyframe_images':
                source=int(meta['source_timestep'])
            elif meta['absolute_index'] is not None:
                source=int(meta['absolute_index'])
            else:
                source=max(0,t+int(meta['delta']))
            if not 0<=source<=t:raise ValueError('Future source image')
            view_index=int(meta['view_index'])
            if not 0<=view_index<len(views):raise ValueError('Invalid camera index')
            arr=np.asarray(im)
            if arr.ndim!=3 or arr.shape[2]!=3 or arr.dtype!=np.uint8:
                raise ValueError('Unexpected native image format')
            try:
                expected=expected_image(source,views[view_index],arr.shape[1],arr.shape[0])
            except FileNotFoundError as exc:
                raise ValueError('Image source was not archived as observed') from exc
            if not np.array_equal(arr,expected):raise ValueError('Image pixels differ from legal observation')
            checked.append(dict(channel=channel,source_step=source,view_index=view_index))
    if example.get('memory_keyframe_count')!=len(example.get('memory_keyframe_images',[])):
        raise ValueError('Memory count mismatch')
    if example.get('memory_keyframe_steps')!=[m['source_timestep'] for m in example.get('memory_keyframe_image_metas',[])]:
        raise ValueError('Memory source steps mismatch')
    return dict(step=t,request_index=request_index,images_checked=len(checked),sources=checked)
