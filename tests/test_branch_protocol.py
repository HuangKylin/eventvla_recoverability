from dataclasses import replace

import pytest

from recoverability.branch_protocol import BranchOutcome, CostConfig, deletion_cost, validate_split_records


CONFIG = CostConfig(step_horizon=1000, compute_scale_seconds=100)
KEEP = BranchOutcome('scene-1', 'state-a', 'action-a', 'cue-a', 'controller-a', 'rng-a',
                     'K', True, 100, 1.0, True, True, 0.0)
RECOVER = replace(KEEP, branch='R', steps=200)


def test_same_success_with_costly_recovery_protects_evidence():
    assert deletion_cost(KEEP, RECOVER, CONFIG, restore_tolerance=1e-6) > 0


@pytest.mark.parametrize('field', ['family_id', 'parent_state_hash', 'action_context_hash',
                                   'evidence_id', 'controller_hash', 'rng_state_hash'])
def test_mismatched_pair_is_not_training_data(field):
    with pytest.raises(ValueError, match='Unmatched'):
        deletion_cost(KEEP, replace(RECOVER, **{field: 'different'}), CONFIG, restore_tolerance=1e-6)


def test_failed_deletion_dominates_fast_execution():
    slow_keep = replace(KEEP, steps=100000, compute_seconds=100000)
    fast_failure = replace(RECOVER, success=False, steps=0, compute_seconds=0)
    assert deletion_cost(slow_keep, fast_failure, CONFIG, restore_tolerance=1e-6) > 0


def test_no_recovery_branch_cannot_replace_recovery_label():
    with pytest.raises(ValueError):
        deletion_cost(KEEP, replace(RECOVER, branch='D', recovery_allowed=False), CONFIG, restore_tolerance=1e-6)


def test_invalid_restore_is_rejected_even_if_success_matches():
    with pytest.raises(ValueError, match='restoration'):
        deletion_cost(KEEP, replace(RECOVER, restore_error=0.1), CONFIG, restore_tolerance=1e-6)


def test_counterfactual_siblings_cannot_cross_splits():
    with pytest.raises(ValueError, match='leaks'):
        validate_split_records([{'family_id': 'scene-1', 'branch': 'K', 'split': 'train'},
                                {'family_id': 'scene-1', 'branch': 'R', 'split': 'test'}])
