"""Explain a narrow protocol-4/5 ndarray writeability hash discrepancy.

Only call on this experiment's trusted local client snapshots. No simulator,
model, GPU, or snapshot file is modified by this audit.
"""
import hashlib
from pathlib import Path
import pickle


def digest(value):
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def verify_event_snapshot(path, *, original_hash, restored_hash):
    payload = Path(path).read_bytes()
    state = pickle.loads(payload)
    events = state['keyframe_image_memory']
    loaded_hash = digest(events)
    before = [(image.dtype.str, image.shape, image.tobytes()) for event in events for image in event['images']]
    flags = [image.flags.writeable for event in events for image in event['images']]
    for event in events:
        for image in event['images']:
            image.setflags(write=False)
    readonly_hash = digest(events)
    after = [(image.dtype.str, image.shape, image.tobytes()) for event in events for image in event['images']]
    return dict(verified=loaded_hash == restored_hash and readonly_hash == original_hash and before == after,
                snapshot_sha256=hashlib.sha256(payload).hexdigest(),
                restored_events_hash=loaded_hash, readonly_events_hash=readonly_hash,
                original_events_hash=original_hash, restored_writeable_flags=flags,
                image_values_unchanged=before == after,
                explanation='Protocol-4 snapshot restoration makes originally readonly NumPy images writeable; protocol-5 hashing records that flag.')
