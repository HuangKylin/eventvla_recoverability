"""Predeclared fixed-count input calibration; not formal K/R supervision."""
from copy import deepcopy
import hashlib
import json
import numpy as np
from .native_items import intervene_item
from .vla_intervention import image_hash, purge_evidence

PROTOCOL = 'fixed_four_slots_request_rng_v1'
CONDITIONS = ['native_legacy', 'slots_legacy', 'slots_paired', 'slots_paired_repeat'] + [
    f'{fill}_item_{index}' for fill in ('gray', 'black') for index in range(4)]


def specification(condition):
    if condition not in CONDITIONS:
        raise ValueError('Unknown slot calibration condition')
    erased = condition.startswith(('gray_item_', 'black_item_'))
    return dict(capacity=None if condition == 'native_legacy' else 4,
                rng='timestep_legacy' if condition.endswith('_legacy') else 'request_index_v1',
                candidate_index=int(condition[-1]) if erased else None,
                padding=127 if condition.startswith('gray_item_') else (0 if erased else None))


def parent_randomness_key(parent):
    """Public predecision image/history identity; no condition, target, or elapsed time."""
    fields={key:parent[key] for key in ('observation_hash','history_hash','event_hash','native_instruction')}
    return hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()


def paired_request_seed(parent_key, request_index, replicate=0):
    if len(parent_key)!=64 or any(c not in '0123456789abcdef' for c in parent_key):
        raise ValueError('Expected a SHA-256 parent key')
    if type(request_index) is not int or request_index<0 or type(replicate) is not int or replicate<0:
        raise ValueError('Request index and replicate must be nonnegative integers')
    raw=f'{PROTOCOL}:{parent_key}:{replicate}:{request_index}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4],'big')


def apply_condition(client, condition):
    spec=specification(condition)
    if client.action_mode!='abs' or len(client.keyframe_image_memory)!=4:
        raise ValueError('Require four native events and absolute actions')
    original_capacity=int(client.max_keyframe_images)
    if original_capacity<4:raise ValueError('Native capacity cannot hold four events')
    candidate=spec['candidate_index']
    if candidate is None:
        change=purge_evidence(client,[])
    else:
        entry=client.keyframe_image_memory[candidate]
        step=int(entry['step'])
        if len(entry['images'])!=1 or np.asarray(entry['images'][0]).dtype!=np.uint8:
            raise ValueError('Require a uint8 head-camera candidate')
        replacement=np.full_like(entry['images'][0],spec['padding'])
        if image_hash(replacement)==image_hash(entry['images'][0]):raise ValueError('Candidate already equals padding')
        # Keep only the declared slot metadata. No old image is copied into the vacancy.
        empty={key:deepcopy(value) for key,value in entry.items() if key!='images'}
        empty['images']=[replacement]
        change=intervene_item(client,step,drop=True)
        client.keyframe_image_memory.insert(candidate,empty)
        change.update(empty_slot=candidate,padding_value=spec['padding'],padding_hash=image_hash(replacement),
                      metadata_semantics='Vacant slot retains its original timestamp/view metadata; padding is not an observation')
    if spec['capacity'] is not None:client.max_keyframe_images=spec['capacity']
    change.update(protocol=PROTOCOL,condition=condition,specification=spec,
                  original_capacity=original_capacity,active_capacity=int(client.max_keyframe_images),
                  retained_event_count=len(client.keyframe_image_memory),formal_training_eligible=False)
    return change
