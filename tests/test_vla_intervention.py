from collections import deque
from types import SimpleNamespace
import numpy as np
import pytest
from recoverability.vla_intervention import purge_evidence,audit_payload


def client():
    c=SimpleNamespace(initial_step=0,image_history=deque([
        dict(step=0,images=[np.zeros((2,2,3),dtype=np.uint8)]),
        dict(step=10,images=[np.full((2,2,3),n,dtype=np.uint8) for n in (11,12,13)]),
        dict(step=20,images=[np.full((2,2,3),21,dtype=np.uint8)])]),
        keyframe_image_memory=[dict(step=10,images=[np.full((2,2,3),11,dtype=np.uint8)])],
        raw_actions=np.ones((50,14)),plan_start_step=5,prev_action=np.ones(14),
        pending_commit_done=False,action_ensembler=None)
    c._clear_pending_commit=lambda:setattr(c,'pending_commit_done',True)
    return c


def test_deletion_reaches_temporal_wrist_and_action_caches():
    c=client();removed=c.image_history[1]['images']
    audit=purge_evidence(c,[10])
    assert [x['step'] for x in c.image_history]==[0,20]
    assert not c.keyframe_image_memory and c.raw_actions is None and c.pending_commit_done
    # Event-only deletion would have missed the wrist view; the payload gate
    # must catch a deliberately reintroduced copy through the anchor channel.
    with pytest.raises(RuntimeError,match='bypassed'):
        audit_payload({'examples':[{'image':[removed[2]]}]},audit['forbidden_image_hashes'])
    assert audit_payload({'examples':[{'image':c.image_history[0]['images']}]},audit['forbidden_image_hashes'])['forbidden_hits']==0


def test_first_frame_requires_explicit_cue_free_protocol():
    with pytest.raises(ValueError,match='cue-free'): purge_evidence(client(),[0])
