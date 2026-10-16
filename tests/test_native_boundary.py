import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from recoverability.native_boundary import assess_parent, AbsoluteLiftTracker


def parent(**changes):
    xyz = np.array([[x, -.14, .741] for x in [-.33, -.11, .11, .33]])
    args = dict(positions=xyz.copy(), initial_positions=xyz.copy(), block_positions=xyz.copy(),
                grippers=[1., 1.], phase='final', inspect_pointer=4, final_pointer=0,
                failed=False, step=760, plan_start=710, plan_length=50, settled_steps=3)
    args.update(changes)
    return assess_parent(**args)


def test_valid_closed_chunk_boundary():
    assert parent()['eligible']


@pytest.mark.parametrize('change', [dict(step=759), dict(step=761), dict(grippers=[1., .1]),
    dict(final_pointer=1), dict(failed=True), dict(settled_steps=2), dict(plan_start=None)])
def test_reject_early_or_ineligible_boundary(change):
    assert not parent(**change)['eligible']


def test_stage_final_does_not_mean_cover_is_down():
    xyz = np.array([[x, -.14, .741] for x in [-.33, -.11, .11, .33]])
    xyz[3, 2] += .0795
    assert not parent(positions=xyz)['eligible']
    xyz[3] = [.33, -.03, .741]
    assert not parent(positions=xyz)['eligible']


def test_absolute_reference_detects_old_false_negative():
    tracker = AbsoluteLiftTracker([.741] * 4)
    for _ in range(3): tracker.update([.741, .741, .741, .8205])
    assert tracker.first_target(3)


def test_transient_and_simultaneous_lifts_are_not_unique_target():
    tracker = AbsoluteLiftTracker([.741] * 4)
    tracker.update([.9, .741, .741, .741])
    tracker.update([.741] * 4)
    assert not tracker.events
    for _ in range(3): tracker.update([.9, .9, .741, .741])
    assert tracker.events[0]['slots'] == [0, 1]
    assert not tracker.first_target(0)


def test_nonfinite_geometry_rejected():
    with pytest.raises(ValueError): parent(grippers=[float('nan'), 1.])
