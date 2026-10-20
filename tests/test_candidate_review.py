import pytest
from recoverability.candidate_review import (
    PublicVisitArchive, enumerate_candidate, screen_side_rule, single_query_cost,
)


def test_geometry_and_action_both_change_the_required_evidence():
    left = {0, 1, 2, 3}
    assert screen_side_rule(left, 'close_left_screen') != screen_side_rule(left, 'close_right_screen')
    assert screen_side_rule(left, 'close_left_screen') != screen_side_rule({0, 2, 4, 6}, 'close_left_screen')
    with pytest.raises(ValueError, match='Unknown action'):
        screen_side_rule(left, 'close_unseen_screen')


def test_rule_can_saturate_headroom_despite_action_effect():
    report = enumerate_candidate()
    assert report['layout_partitions'] == 70
    assert report['action_conditions'] == report['rule_oracle_equal_conditions'] == 140
    assert report['averages']['rule_expected_loss'] == pytest.approx(0.05)
    assert report['averages']['fifo_expected_loss'] == pytest.approx(0.275)
    assert report['averages']['archive_information_loss'] == 0


def test_information_cost_is_single_query_and_validates_inputs():
    assert single_query_cost(set(), set(range(8))) == 1
    assert single_query_cost(set(range(8)), set(range(8))) == 0
    with pytest.raises(ValueError):
        single_query_cost({8}, set())


def test_archive_preserves_every_observed_payload_without_answer_metadata():
    archive = PublicVisitArchive()
    archive.append(0, b'first-rgb')
    archive.append(1, b'other-rgb')
    assert archive.read(0) == b'first-rgb'
    assert archive.read(1) == b'other-rgb'
    assert archive.payload_bytes == 18
    with pytest.raises(KeyError):
        archive.read(2)
    with pytest.raises(ValueError, match='duplicate'):
        archive.append(0, b'changed')
