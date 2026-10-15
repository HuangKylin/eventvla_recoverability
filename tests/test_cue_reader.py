import numpy as np
import pytest

from recoverability.cue_reader import ProbeMemory,read_colour


def cue(colour):
    image=np.full((240,320,3),128,dtype=np.uint8)
    image[150:180,35:60]=colour
    return image


@pytest.mark.parametrize('name,colour',[
    ('red',(255,0,0)),('green',(0,255,0)),('blue',(0,0,255)),('yellow',(255,255,0)),
])
def test_colour_read_from_pixels(name,colour):
    assert read_colour(cue(colour)).answer==name


def test_neutral_and_covered_observations_abstain():
    assert read_colour(cue((128,128,128))).answer is None


def test_delete_removes_previous_semantic_answer_and_all_persistent_image_bytes():
    memory=ProbeMemory(cue((0,255,0)))
    assert memory.answer()=='green'
    assert memory.nbytes>0
    memory.drop()
    assert memory.answer() is None
    assert memory.nbytes==0
    assert vars(memory)=={'image':None}


def test_memory_owns_its_image_and_does_not_alias_evaluator_observation():
    image=cue((0,0,255))
    memory=ProbeMemory(image)
    image[:]=128
    assert memory.answer()=='blue'


def test_conflicting_cues_in_calibrated_region_abstain():
    image=cue((255,0,0))
    image[150:180,60:85]=[0,255,0]
    assert read_colour(image).answer is None


def test_wrong_image_encoding_is_rejected():
    with pytest.raises(ValueError,match='uint8 RGB'):
        read_colour(cue((255,0,0)).astype(float)/255)
