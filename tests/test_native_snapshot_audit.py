import importlib.util
from pathlib import Path
import pickle

import numpy as np

path = Path(__file__).resolve().parents[1] / 'scripts/native_snapshot_audit.py'
spec = importlib.util.spec_from_file_location('native_snapshot_audit', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def snapshot(tmp_path):
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    image.setflags(write=False)
    events = [dict(step=97, images=[image], metas=[dict(source_timestep=97)], confidence=1.0)]
    original = module.digest(events)
    path = tmp_path / 'client_state.pkl'
    path.write_bytes(pickle.dumps(dict(keyframe_image_memory=events), protocol=4))
    restored = module.digest(pickle.loads(path.read_bytes())['keyframe_image_memory'])
    return path, original, restored


def test_readonly_roundtrip_explains_hash_without_ignoring_image_values(tmp_path):
    path, original, restored = snapshot(tmp_path)
    assert original != restored
    report = module.verify_event_snapshot(path, original_hash=original, restored_hash=restored)
    assert report['verified']
    assert report['image_values_unchanged']


def test_pixel_change_cannot_be_excused_as_writeability(tmp_path):
    path, original, restored = snapshot(tmp_path)
    state = pickle.loads(path.read_bytes())
    state['keyframe_image_memory'][0]['images'][0][0,0,0] += 1
    path.write_bytes(pickle.dumps(state,protocol=4))
    changed = module.digest(state['keyframe_image_memory'])
    assert not module.verify_event_snapshot(path, original_hash=original, restored_hash=changed)['verified']


def test_timestamp_change_cannot_be_excused_as_writeability(tmp_path):
    path, original, restored = snapshot(tmp_path)
    state = pickle.loads(path.read_bytes())
    state['keyframe_image_memory'][0]['step'] += 1
    path.write_bytes(pickle.dumps(state,protocol=4))
    changed = module.digest(state['keyframe_image_memory'])
    assert not module.verify_event_snapshot(path, original_hash=original, restored_hash=changed)['verified']
