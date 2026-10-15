import numpy as np
import pytest
from recoverability.fixed_skills import SLOT_ROIS
from recoverability.versioned_memory import VersionedMemory


def image_for(slot, rgb=(0,255,0)):
    image=np.full((240,320,3),128,dtype=np.uint8)
    x0,y0,x1,y1=SLOT_ROIS[slot]
    x=int((x0+x1)*160)
    image[155:175,x-8:x+8]=rgb
    return image


def stream(memory):
    for slot in range(4):
        memory.observe(slot,0,image_for(slot))
    memory.proposed_advance([0,1,2,3])
    for slot in range(4):
        memory.observe(slot,1,image_for(slot,(255,0,0)))


def test_fifo_naturally_loses_old_versions_at_capacity_four():
    memory=VersionedMemory('fifo',4)
    stream(memory)
    assert memory.keys()==[(s,1) for s in range(4)]
    assert memory.answer(0,0)==(None,0)
    assert not memory.recovery_possible_from_public_state(0,0)
    assert memory.answer(0,1)==('red',1)


def test_action_rule_keeps_cues_that_public_action_makes_irrecoverable():
    memory=VersionedMemory('action_aware',4)
    stream(memory)
    assert memory.keys()==[(s,0) for s in range(4)]
    assert memory.answer(0,0)==('green',1)
    assert memory.answer(0,1)==(None,0)
    assert memory.recovery_possible_from_public_state(0,1)


@pytest.mark.parametrize('method',['fifo','uniform','oldest','action_aware'])
@pytest.mark.parametrize('capacity',[0,2,4,8])
def test_bounded_modes_have_no_semantic_archive_or_excess_images(method,capacity):
    memory=VersionedMemory(method,capacity)
    stream(memory)
    assert memory.semantic=={}
    assert len(memory.entries)<=capacity
    assert memory.storage()['image_payload_bytes']<=capacity*240*320*3
    assert set(vars(memory))=={'method','capacity','seed','entries','semantic','versions','ordinal'}


def test_full_archive_and_structured_baselines_can_keep_all_evidence():
    for method in ('archive','structured'):
        memory=VersionedMemory(method,4)
        stream(memory)
        assert len(memory.keys())==8
        assert memory.answer(2,0)[0]=='green'
        assert memory.answer(2,1)[0]=='red'
        assert memory.storage()['bounded'] is False


def test_observation_cannot_claim_a_future_unannounced_version():
    memory=VersionedMemory('fifo',4)
    with pytest.raises(ValueError,match='public apparatus version'):
        memory.observe(0,1,image_for(0))


def test_input_image_alias_cannot_modify_retained_evidence():
    memory=VersionedMemory('fifo',1)
    rgb=image_for(0)
    memory.observe(0,0,rgb)
    rgb[:]=0
    assert memory.answer(0,0)[0]=='green'


def test_oldest_baseline_matches_action_rule_on_this_simple_overwrite_schedule():
    oldest,aware=VersionedMemory('oldest',4),VersionedMemory('action_aware',4)
    stream(oldest)
    stream(aware)
    assert oldest.keys()==aware.keys()
