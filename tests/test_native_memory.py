from collections import deque
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from recoverability.native_memory import intervene, metadata_hash
from recoverability.vla_intervention import image_hash, audit_payload


def client():
    c = SimpleNamespace(action_mode='abs', initial_step=0,
        image_history=deque([dict(step=800, images=[np.zeros((2, 2, 3), dtype=np.uint8)])]),
        keyframe_image_memory=[dict(step=(i+1)*100, confidence=.8+i*.01,
            metas=[dict(source_timestep=(i+1)*100, view='head')],
            images=[np.full((2, 2, 3), i+1, dtype=np.uint8)]) for i in range(4)],
        raw_actions=np.ones((50, 14)), plan_start_step=750, prev_action=np.ones(14),
        action_ensembler=None, pending_commit_done=False)
    c._clear_pending_commit = lambda: setattr(c, 'pending_commit_done', True)
    return c


@pytest.mark.parametrize('condition', ['keep', 'drop_all', 'mismatch'])
def test_cache_reset_equal_and_temporal_untouched(condition):
    c = client(); history = c.image_history[0]; donor = deepcopy(c.keyframe_image_memory)
    for e in donor: e['images'][0][:] += 10
    r = intervene(c, condition, donor)
    assert c.image_history[0] is history
    assert c.raw_actions is None and c.plan_start_step is None and c.prev_action is None
    assert c.pending_commit_done and r['action_cache_cleared']
    assert len(c.keyframe_image_memory) == (0 if condition == 'drop_all' else 4)


def test_mismatch_changes_pixels_only_and_does_not_alias_donor():
    c = client(); donor = deepcopy(c.keyframe_image_memory); meta = metadata_hash(c.keyframe_image_memory)
    for e in donor:
        e['step'] += 5; e['confidence'] = .1; e['images'][0][:] += 10
    r = intervene(c, 'mismatch', donor)
    assert metadata_hash(c.keyframe_image_memory) == meta
    assert r['expected_event_image_hashes'] == [image_hash(e['images'][0]) for e in donor]
    donor[0]['images'][0][:] = 99
    assert image_hash(c.keyframe_image_memory[0]['images'][0]) == r['expected_event_image_hashes'][0]


def test_partial_shared_images_not_falsely_marked_leak():
    c = client(); donor = deepcopy(c.keyframe_image_memory)
    donor[-1]['images'][0][:] = 99
    old = c.keyframe_image_memory[-1]['images'][0].copy()
    r = intervene(c, 'mismatch', donor)
    assert len(r['shared_image_hashes']) == 3
    payload = dict(examples=[dict(image=[], memory_keyframe_images=[e['images'][0] for e in c.keyframe_image_memory])])
    assert audit_payload(payload, r['forbidden_image_hashes'])['forbidden_hits'] == 0
    payload['examples'][0]['image'] = [old]
    with pytest.raises(RuntimeError): audit_payload(payload, r['forbidden_image_hashes'])


def test_reject_invalid_donor_before_mutating_client():
    c = client(); donor = deepcopy(c.keyframe_image_memory); donor[0]['images'][0] = np.zeros((1, 1, 3))
    with pytest.raises(ValueError): intervene(c, 'mismatch', donor)
    assert c.raw_actions is not None and not c.pending_commit_done
    with pytest.raises(ValueError): intervene(c, 'mismatch', c.keyframe_image_memory)


def test_reject_temporal_evidence_overlap():
    c = client(); c.image_history[0]['step'] = 100
    with pytest.raises(ValueError): intervene(c, 'drop_all')
