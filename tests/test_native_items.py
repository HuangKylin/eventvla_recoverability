from copy import deepcopy
from types import SimpleNamespace
import pytest
from test_native_memory import client
from recoverability.native_items import intervene_item, event_signature, candidate_slot, replace_item_content
from recoverability.native_memory import metadata_hash
from recoverability.vla_intervention import audit_payload


def test_single_deletion_preserves_other_images_metadata_and_history():
    c=client(); history=list(c.image_history); original=deepcopy(c.keyframe_image_memory)
    r=intervene_item(c, 200, drop=True)
    assert event_signature(c.keyframe_image_memory)==event_signature([original[i] for i in (0,2,3)])
    assert list(c.image_history)==history
    assert r['deleted_steps']==[200] and r['noncandidate_unchanged']
    assert c.raw_actions is None and c.plan_start_step is None and c.pending_commit_done
    payload={'examples':[{'image':[], 'memory_keyframe_images':[e['images'][0] for e in c.keyframe_image_memory]}]}
    assert audit_payload(payload,r['forbidden_image_hashes'])['forbidden_hits']==0
    payload['examples'][0]['image']=[original[1]['images'][0]]
    with pytest.raises(RuntimeError):audit_payload(payload,r['forbidden_image_hashes'])


def test_keep_clears_same_caches_without_modifying_events():
    c=client(); before=event_signature(c.keyframe_image_memory)
    r=intervene_item(c, 200, drop=False)
    assert event_signature(c.keyframe_image_memory)==before
    assert not r['forbidden_image_hashes'] and c.raw_actions is None and c.pending_commit_done


@pytest.mark.parametrize('failure',['missing','recent','duplicate'])
def test_unsafe_candidate_rejected_before_mutation(failure):
    c=client()
    if failure=='recent':c.image_history[0]['step']=200
    if failure=='duplicate':c.keyframe_image_memory[2]['images']=c.keyframe_image_memory[1]['images']
    with pytest.raises(ValueError):intervene_item(c, 999 if failure=='missing' else 200, drop=True)
    assert c.raw_actions is not None and not c.pending_commit_done and len(c.keyframe_image_memory)==4


def test_routing_uses_public_past_rgb_and_rejects_ambiguous_location():
    e=client().keyframe_image_memory[0]
    assert candidate_slot(e,lambda slot,rgb:SimpleNamespace(answer='red' if slot==2 else None))==2
    for answer in (None,'red'):
        with pytest.raises(ValueError):candidate_slot(e,lambda slot,rgb:SimpleNamespace(answer=answer))


def test_content_probe_preserves_count_order_metadata_and_other_events():
    c=client();before=deepcopy(c.keyframe_image_memory);meta=metadata_hash(before)
    replacement=c.image_history[0]['images'][0].copy()
    result=replace_item_content(c,200,replacement)
    assert len(c.keyframe_image_memory)==4 and metadata_hash(c.keyframe_image_memory)==meta
    assert event_signature([c.keyframe_image_memory[i] for i in (0,2,3)])==event_signature([before[i] for i in (0,2,3)])
    assert result['synthetic_input_diagnostic'] and result['forbidden_image_hashes']
    replacement[:]=99
    assert not (c.keyframe_image_memory[1]['images'][0]==99).any()
