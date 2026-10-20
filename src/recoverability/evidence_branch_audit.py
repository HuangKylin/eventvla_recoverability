"""Audit fixed-slot content interventions using their explicit true image sources."""
from copy import deepcopy
from pathlib import Path
import hashlib
import pickle


def equal(a, b):
    import numpy as np
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def audit_first_request(reference, actual, source, replacements):
    """All fields except the declared image slots must remain exactly identical."""
    import numpy as np
    from PIL import Image
    expected = deepcopy(reference)
    for replacement in replacements:
        slot = replacement['slot']
        old = expected['examples'][0]['memory_keyframe_images'][slot]
        with Image.open(Path(source)/'observations'/f"{replacement['source_step']:06d}"/'head_camera.png') as im:
            expected['examples'][0]['memory_keyframe_images'][slot] = np.asarray(im.resize((old.shape[1], old.shape[0])))
    if not equal(expected, actual):
        raise ValueError('First request changed beyond declared memory image contents')
    return True


def audit_branch(source, branch, replacements, parent):
    import numpy as np
    from PIL import Image
    from recoverability.evidence_diagnostic import inference_seed, validate_replacements
    source, branch = Path(source), Path(branch)
    validate_replacements(replacements, parent['event_count'], parent['step'], set(parent['observed_steps']))
    files = sorted((branch/'requests').glob('*.pkl'))
    if not files:
        raise ValueError('No actual paired model request')
    first = parent['request_index']
    reference = pickle.loads((source/'requests'/f'{first:04d}.pkl').read_bytes())
    actual = pickle.loads(files[0].read_bytes())
    audit_first_request(reference, actual, source, replacements)
    mapping = {parent['event_steps'][r['slot']]: r['source_step'] for r in replacements}
    requests = images = 0
    for offset, file in enumerate(files):
        index = first + offset
        if file.stem != f'{index:04d}':
            raise ValueError('Missing paired request')
        payload = pickle.loads(file.read_bytes())  # Only this project's own generated artifacts.
        example = payload['examples'][0]
        if len(payload['examples']) != 1 or payload['paired_seed'] != inference_seed(parent['seed'], index):
            raise ValueError('Batch/RNG mismatch')
        if example['lang'] != parent['native_instruction']:
            raise ValueError('Instruction changed')
        if any(k in example for k in ('evaluator_state','actual_first_block_id','actual_other_block_id')):
            raise ValueError('Evaluator state in policy input')
        t = int(example['timestep'])
        if t < parent['step']:
            raise ValueError('Request before branch parent')
        for channel, key in [('image','image_metas'), ('memory_keyframe_images','memory_keyframe_image_metas')]:
            if len(example[channel]) != len(example[key]):
                raise ValueError('Image/metadata count mismatch')
            for pixels, meta in zip(example[channel], example[key]):
                if channel == 'image':
                    step = meta['absolute_index'] if meta['absolute_index'] is not None else max(0, t + int(meta['delta']))
                else:
                    step = mapping.get(meta['source_timestep'], meta['source_timestep'])
                if not 0 <= step <= t:
                    raise ValueError('Future source')
                view = ('head_camera','left_camera','right_camera')[meta['view_index']]
                archive = branch if step >= parent['step'] else source
                with Image.open(archive/'observations'/f'{step:06d}'/f'{view}.png') as im:
                    expected = np.asarray(im.resize((pixels.shape[1], pixels.shape[0])))
                if not np.array_equal(pixels, expected):
                    raise ValueError('Actual pixels do not match declared legal observation')
                images += 1
        if example['memory_keyframe_count'] != len(example['memory_keyframe_images']):
            raise ValueError('Changed memory count')
        if example['memory_keyframe_steps'] != [m['source_timestep'] for m in example['memory_keyframe_image_metas']]:
            raise ValueError('Inconsistent memory steps')
        requests += 1
    return dict(first_request_exact_except_declared_images=True, requests=requests, images=images,
                metadata_semantics='original_timestamps_held_fixed_actual_sources_explicitly_audited',
                deployed_retrieval_validated=False)
