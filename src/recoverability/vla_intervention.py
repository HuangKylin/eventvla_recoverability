"""Explicit EventVLA client intervention; never changes upstream default runs."""
import hashlib
import numpy as np


def image_hash(image):
    a=np.ascontiguousarray(image)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()


def purge_evidence(client, steps):
    """Delete entire observation records, including all three camera views.

Clears action/commit caches computed before deletion. Server memory is reset
separately by the deterministic wrapper on EVERY inference, for both K and R.
The first frame must be cue-free; refusing deletion of it avoids inventing a
replacement visual observation. The current frame must also be cue-free.
"""
    steps=set(int(s) for s in steps)
    if client.initial_step in steps:
        raise ValueError('The initial observation must be cue-free in this protocol')
    forbidden=[]
    for record in client.image_history:
        if int(record['step']) in steps:
            forbidden.extend(image_hash(im) for im in record['images'])
    for entry in client.keyframe_image_memory:
        if int(entry['step']) in steps:
            forbidden.extend(image_hash(im) for im in entry['images'])
    retained=[r for r in client.image_history if int(r['step']) not in steps]
    client.image_history.clear()
    client.image_history.extend(retained)
    client.num_image_history=len(client.image_history)
    client.keyframe_image_memory=[e for e in client.keyframe_image_memory if int(e['step']) not in steps]
    client.raw_actions=None
    client.plan_start_step=None
    client.prev_action=None
    client._clear_pending_commit()
    if client.action_ensembler is not None:
        client.action_ensembler.reset()
    return dict(deleted_steps=sorted(steps),forbidden_image_hashes=sorted(set(forbidden)),
                remaining_event_steps=[e['step'] for e in client.keyframe_image_memory],
                remaining_temporal_steps=[r['step'] for r in client.image_history],
                action_cache_cleared=client.raw_actions is None,pending_commit_cleared=client.pending_commit_done)


def audit_payload(payload, forbidden):
    """Audit actual serialized model inputs, not just memory-list contents."""
    records=[]
    for example in payload['examples']:
        for name in ('image','memory_keyframe_images'):
            for image in example.get(name,[]):
                sha=image_hash(image)
                records.append(dict(channel=name,sha256=sha,bytes=int(np.asarray(image).nbytes)))
    hits=[r for r in records if r['sha256'] in set(forbidden)]
    if hits:
        raise RuntimeError('Deleted evidence bypassed intervention through model image inputs')
    return dict(anchor_images=sum(r['channel']=='image' for r in records),
                event_images=sum(r['channel']=='memory_keyframe_images' for r in records),
                image_payload_bytes=sum(r['bytes'] for r in records),forbidden_hits=len(hits),images=records)
