"""Event-image interventions at a validated, closed native chunk boundary."""
from copy import deepcopy
import hashlib
import pickle
import numpy as np

from .vla_intervention import image_hash, purge_evidence


def metadata_hash(entries):
    return hashlib.sha256(pickle.dumps([
        {k: v for k, v in e.items() if k != 'images'} for e in entries
    ], protocol=5)).hexdigest()


def intervene(client, condition, donor_entries=None):
    if condition not in ('keep', 'drop_all', 'mismatch'):
        raise ValueError('Unknown memory condition')
    if client.action_mode != 'abs':
        raise ValueError('This cache-reset protocol requires absolute actions')
    entries = client.keyframe_image_memory
    steps = [int(e['step']) for e in entries]
    if len(entries) != 4 or len(set(steps)) != 4 or steps != sorted(steps):
        raise ValueError('Need four ordered inspection events')
    if any(len(e['images']) != 1 for e in entries):
        raise ValueError('Expected one head image per event')
    if set(steps) & {int(r['step']) for r in client.image_history}:
        raise ValueError('Event evidence overlaps temporal anchors')
    before = [image_hash(e['images'][0]) for e in entries]
    metadata_before = metadata_hash(entries)
    replacements = None
    if condition == 'mismatch':
        if donor_entries is None or len(donor_entries) != len(entries):
            raise ValueError('Need a complete donor inspection history')
        donor_steps = [int(e['step']) for e in donor_entries]
        if donor_steps != sorted(set(donor_steps)):
            raise ValueError('Donor history must be ordered and unique')
        replacements = []
        for recipient, donor in zip(entries, donor_entries):
            if len(donor['images']) != 1:
                raise ValueError('Donor view count differs')
            old, new = np.asarray(recipient['images'][0]), np.asarray(donor['images'][0])
            if old.shape != new.shape or old.dtype != new.dtype:
                raise ValueError('Mismatch must preserve image shape and dtype')
            replacements.append(new.copy())
        if before == [image_hash(im) for im in replacements]:
            raise ValueError('Donor makes no image intervention')
    # Equal cache reset for all groups; no pending cross-boundary commitment.
    result = purge_evidence(client, steps if condition == 'drop_all' else [])
    if replacements is not None:
        client.keyframe_image_memory = deepcopy(client.keyframe_image_memory)
        for entry, image in zip(client.keyframe_image_memory, replacements):
            entry['images'] = [image]
    after = [image_hash(e['images'][0]) for e in client.keyframe_image_memory]
    metadata_after = metadata_hash(client.keyframe_image_memory)
    if condition != 'drop_all' and metadata_before != metadata_after:
        raise RuntimeError('Retained event metadata changed')
    result.update(condition=condition, original_event_image_hashes=before,
        expected_event_image_hashes=after,
        forbidden_image_hashes=sorted(set(before) - set(after)),
        original_event_steps=steps, metadata_before=metadata_before,
        metadata_after=metadata_after, retained_metadata_unchanged=condition != 'drop_all',
        shared_image_hashes=sorted(set(before) & set(after)))
    return result
