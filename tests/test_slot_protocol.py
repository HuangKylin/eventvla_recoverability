from copy import deepcopy
import numpy as np
import pytest
from test_native_memory import client
from recoverability.native_memory import metadata_hash
from recoverability.native_items import event_signature
from recoverability.slot_protocol import apply_condition,paired_request_seed,parent_randomness_key
from recoverability.vla_intervention import audit_payload


@pytest.mark.parametrize('fill,value',[('gray',127),('black',0)])
@pytest.mark.parametrize('index',range(4))
def test_vacancy_keeps_four_slots_and_removes_only_candidate_pixels(fill,value,index):
    c=client();c.max_keyframe_images=5;old=deepcopy(c.keyframe_image_memory);history=deepcopy(c.image_history)
    result=apply_condition(c,f'{fill}_item_{index}')
    assert c.max_keyframe_images==4 and len(c.keyframe_image_memory)==4
    assert metadata_hash(c.keyframe_image_memory)==metadata_hash(old)
    assert event_signature([e for i,e in enumerate(c.keyframe_image_memory) if i!=index])==event_signature([e for i,e in enumerate(old) if i!=index])
    assert np.array_equal(c.image_history[0]['images'][0],history[0]['images'][0])
    assert np.all(c.keyframe_image_memory[index]['images'][0]==value)
    payload=dict(examples=[dict(image=[],memory_keyframe_images=[e['images'][0] for e in c.keyframe_image_memory])])
    assert audit_payload(payload,result['forbidden_image_hashes'])['event_images']==4
    payload['examples'][0]['image']=[old[index]['images'][0]]
    with pytest.raises(RuntimeError):audit_payload(payload,result['forbidden_image_hashes'])
    assert c.raw_actions is None and c.plan_start_step is None and c.pending_commit_done


@pytest.mark.parametrize('condition,capacity',[('native_legacy',5),('slots_legacy',4),('slots_paired',4),('slots_paired_repeat',4)])
def test_keep_controls_only_change_declared_capacity(condition,capacity):
    c=client();c.max_keyframe_images=5;before=event_signature(c.keyframe_image_memory)
    result=apply_condition(c,condition)
    assert event_signature(c.keyframe_image_memory)==before and not result['forbidden_image_hashes']
    assert c.max_keyframe_images==capacity and c.raw_actions is None


def test_randomness_ignores_condition_candidate_clock_and_target():
    parent=dict(observation_hash='a',history_hash='b',event_hash='c',native_instruction='inspect')
    key=parent_randomness_key(parent)
    varied=dict(parent,step=9999,condition='recover',candidate_index=3,target_slot=2)
    assert key==parent_randomness_key(varied)
    stream=[paired_request_seed(key,i) for i in range(20)]
    assert stream==[paired_request_seed(parent_randomness_key(varied),i) for i in range(20)]
    assert len(set(stream))==20 and all(0<=seed<2**32 for seed in stream)
    assert stream!=[paired_request_seed(key,i,replicate=1) for i in range(20)]
    assert key!=parent_randomness_key(dict(parent,event_hash='new'))


@pytest.mark.parametrize('index',[-1,1.5,True])
def test_invalid_random_stream_index_rejected(index):
    with pytest.raises(ValueError):paired_request_seed('a'*64,index)


def test_recent_candidate_is_rejected_before_mutation():
    c=client();c.max_keyframe_images=5;c.image_history[0]['step']=200
    before=event_signature(c.keyframe_image_memory)
    with pytest.raises(ValueError):apply_condition(c,'gray_item_1')
    assert c.max_keyframe_images==5 and event_signature(c.keyframe_image_memory)==before and c.raw_actions is not None
