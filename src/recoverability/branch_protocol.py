"""Validate paired rollout labels; this module does not generate simulated data."""
from dataclasses import dataclass
import hashlib
import math
from collections.abc import Iterable


@dataclass(frozen=True)
class CostConfig:
    step_horizon: int
    compute_scale_seconds: float
    step_weight: float = 0.10
    compute_weight: float = 0.02

    def __post_init__(self):
        if (not isinstance(self.step_horizon, int) or isinstance(self.step_horizon, bool)
                or self.step_horizon <= 0 or not math.isfinite(self.compute_scale_seconds)
                or self.compute_scale_seconds <= 0):
            raise ValueError('Positive finite cost scales are required')
        if not all(math.isfinite(x) and x >= 0 for x in (self.step_weight, self.compute_weight)):
            raise ValueError('Cost weights must be finite and non-negative')
        if self.step_weight + self.compute_weight >= 1:
            raise ValueError('Failure must dominate the bounded auxiliary costs')


@dataclass(frozen=True)
class BranchOutcome:
    family_id: str
    parent_state_hash: str
    action_context_hash: str
    evidence_id: str
    controller_hash: str
    rng_state_hash: str
    branch: str
    success: bool
    steps: int
    compute_seconds: float
    recovery_allowed: bool
    restore_validated: bool
    restore_error: float


def cost(outcome: BranchOutcome, config: CostConfig) -> float:
    if not isinstance(outcome.success, bool):
        raise ValueError('Success must be Boolean')
    if not isinstance(outcome.steps, int) or isinstance(outcome.steps, bool) or outcome.steps < 0:
        raise ValueError('Step count must be a non-negative integer')
    if not math.isfinite(outcome.compute_seconds) or outcome.compute_seconds < 0:
        raise ValueError('Compute time must be finite and non-negative')
    return (float(not outcome.success)
            + config.step_weight * min(outcome.steps / config.step_horizon, 1)
            + config.compute_weight * min(outcome.compute_seconds / config.compute_scale_seconds, 1))


def deletion_cost(keep: BranchOutcome, recover: BranchOutcome, config: CostConfig,
                  *, restore_tolerance: float) -> float:
    if not math.isfinite(restore_tolerance) or restore_tolerance < 0:
        raise ValueError('A finite non-negative restore tolerance is required')
    if keep.branch != 'K' or recover.branch != 'R':
        raise ValueError('The training label requires K and R, not the no-recovery diagnostic branch')
    for field in ('family_id', 'parent_state_hash', 'action_context_hash',
                  'evidence_id', 'controller_hash', 'rng_state_hash'):
        a, b = getattr(keep, field), getattr(recover, field)
        if not a or a != b:
            raise ValueError('Unmatched paired branches: ' + field)
    for outcome in (keep, recover):
        if not outcome.recovery_allowed:
            raise ValueError('Both training branches must permit the shared recovery policy')
        if (not outcome.restore_validated or not math.isfinite(outcome.restore_error)
                or outcome.restore_error < 0 or outcome.restore_error > restore_tolerance):
            raise ValueError('Unvalidated or inaccurate state restoration')
    return cost(recover, config) - cost(keep, config)


def split_for_family(family_id: str, seed: int = 42) -> str:
    """All action conditions and sibling branches use the same parent family ID."""
    if not family_id:
        raise ValueError('A parent scene family ID is required')
    value = int.from_bytes(hashlib.sha256(f'{seed}:{family_id}'.encode()).digest()[:8], 'big') % 100
    return 'train' if value < 80 else 'validation' if value < 90 else 'test'


def validate_split_records(records: Iterable[dict]) -> None:
    seen = {}
    for row in records:
        family, split = row['family_id'], row['split']
        if not family or split not in {'train', 'validation', 'test'}:
            raise ValueError('Missing family or invalid split')
        if family in seen and seen[family] != split:
            raise ValueError('Parent scene family leaks across splits: ' + family)
        seen[family] = split
