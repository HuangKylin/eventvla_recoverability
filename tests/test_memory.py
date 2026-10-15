import numpy as np
import pytest

from recoverability.memory import image_payload_bytes, select_entries


def event(step, value=0):
    return {'step': step, 'images': [np.full((2, 3, 3), value, dtype=np.uint8)]}


def test_streaming_uniform_matches_full_archive_priority_selection():
    # A bounded implementation must produce the same reservoir as a reference
    # with all candidates, even though it never receives discarded candidates.
    all_events = [event(i) for i in range(300)]
    bounded = []
    for entry in all_events:
        bounded = select_entries([*bounded, entry], 4, strategy='uniform', seed=17)
        assert len(bounded) <= 4
    reference = select_entries(all_events, 4, strategy='uniform', seed=17)
    assert [e['step'] for e in bounded] == [e['step'] for e in reference]
    assert image_payload_bytes(bounded) == 4 * 18


def test_fifo_preserves_upstream_behavior_for_ordered_commits():
    bounded = []
    for step in [1, 4, 4, 9, 12, 17]:
        bounded = select_entries([*bounded, event(step, step)], 3)
    assert [e['step'] for e in bounded] == [9, 12, 17]


def test_duplicate_commit_updates_image_and_consumes_one_slot():
    kept = select_entries([event(1, 1), event(4, 4), event(1, 9)], 2)
    assert [e['step'] for e in kept] == [1, 4]
    assert np.all(kept[0]['images'][0] == 9)


def test_high_value_old_evidence_can_reject_new_candidate():
    kept = select_entries([event(1), event(4), event(9)], 2,
                          strategy='scored', scores=[0.9, 0.5, -0.1])
    assert [e['step'] for e in kept] == [1, 4]


def test_zero_budget_does_not_keep_hidden_images():
    assert select_entries([event(1)], 0) == []


@pytest.mark.parametrize('scores', [[float('nan')], [float('inf')], [], None])
def test_invalid_scores_are_rejected(scores):
    with pytest.raises(ValueError):
        select_entries([event(1)], 1, strategy='scored', scores=scores)
