import copy
import numpy as np
import pytest
from recoverability.research_gate import (
    enumerate_answer_ambiguity, match_public_action, rank_diagnostic, scaleup_decision,
)


def test_public_program_does_not_accept_unknown_or_ambiguous_action():
    zero = np.zeros((16, 14))
    one = np.ones((16, 14))
    assert match_public_action(one, {0: zero, 3: one})[0] == 3
    with pytest.raises(ValueError, match='Unknown or ambiguous'):
        match_public_action(one/2, {0: zero, 3: one})
    with pytest.raises(ValueError, match='Unknown or ambiguous'):
        match_public_action(zero, {0: zero, 1: zero})


def test_unique_palette_removes_single_deletion_information_loss():
    native = enumerate_answer_ambiguity(False)
    independent = enumerate_answer_ambiguity(True)
    assert native['fully_inferable_cases'] == native['assignment_deletion_cases'] == 96
    assert independent['fully_inferable_cases'] == 0
    assert independent['possible_missing_values_histogram'] == {4: 1024}


def test_priority_is_computed_from_outcomes_and_rejects_bad_pair():
    rows = [dict(family_id='f', parent_hash='p', action_hash='a', evidence_slot=i,
                 paired_prefix_and_continuation_validated=True,
                 keep=dict(success=True, recovery_steps=0),
                 recover=dict(success=i != 2, recovery_steps=100),
                 delta_cost=0.001 + (i == 2)) for i in range(4)]
    result = rank_diagnostic(rows, 0)
    assert result['diagnostic_best_slots'] == [2]
    assert result['public_rule_matches_best'] is False
    assert result['marginal_priority_gap'] == pytest.approx(1)
    broken = copy.deepcopy(rows)
    broken[2]['delta_cost'] = 0
    with pytest.raises(ValueError, match='disagrees'):
        rank_diagnostic(broken, 0)
    broken = copy.deepcopy(rows)
    broken[2]['parent_hash'] = 'different'
    with pytest.raises(ValueError, match='Mixed'):
        rank_diagnostic(broken, 0)


def test_no_go_requires_all_evidence_and_is_not_a_training_gate():
    values = dict(rule_matches=8, groups=8, rgb_correct=16, rgb_total=16,
                  structured_closed_loop_perfect=True, rule_closed_loop_perfect=True)
    assert scaleup_decision(**values) == 'no_go_current_simple_color_task_scaleup'
    for key, value in [('rule_matches', 7), ('rgb_correct', 15),
                       ('structured_closed_loop_perfect', False), ('rule_closed_loop_perfect', False)]:
        assert scaleup_decision(**dict(values, **{key: value})).startswith('inconclusive')
    with pytest.raises(ValueError, match='Missing'):
        scaleup_decision(**dict(values, groups=0))
