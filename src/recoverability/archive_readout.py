"""Append-only observed-image archive and auditable, per-decision read budgets.

This is an input contract, not a trained retriever or a security sandbox. A caller
must trace *all* proposal inputs and charge encoding/search/policy costs. Runtime
integration must prevent untraced reads; a supplied trace alone cannot prove that.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from recoverability.vla_intervention import image_hash


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class Observation:
    key: str
    step: int
    view: str
    sha256: str


class ObservedArchive:
    """Store delivered observations without eviction; expose only public metadata.

    Completeness refers to append() calls, not to simulator states or unobserved
    frames. Persisting the full online stream is a separate integration task.
    """
    def __init__(self):
        self._records = {}
        self._images = {}

    def append(self, step, view, image):
        if not isinstance(step, int) or step < 0 or not isinstance(view, str) or not view:
            raise ValueError('nonnegative integer timestamp and public view required')
        pixels = np.array(image, copy=True)
        if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
            raise ValueError('archive expects unchanged uint8 RGB observations')
        sha = image_hash(pixels)
        key = digest([step, view, sha])
        self._records[key] = Observation(key, step, view, sha)
        pixels.flags.writeable = False
        self._images[key] = pixels
        return key

    def metadata(self, decision_step):
        return tuple(sorted((r for r in self._records.values() if r.step <= decision_step),
                            key=lambda r: (r.step, r.view, r.key)))

    def fingerprint(self, decision_step):
        return digest([r.__dict__ for r in self.metadata(decision_step)])

    def read(self, keys, decision_step):
        # Repeated anchor references are valid and must be charged repeatedly.
        for key in keys:
            if key not in self._records or self._records[key].step > decision_step:
                raise ValueError('unknown or future observation')
        return [self._images[key].copy() for key in keys]


def select_public(archive, candidates, count, decision_step, strategy):
    """Metadata-only controls. No seed, answer, target slot or future state input."""
    if not isinstance(count, int) or count < 0 or len(set(candidates)) != len(candidates):
        raise ValueError('invalid count or duplicate candidates')
    public = {r.key: r for r in archive.metadata(decision_step)}
    if any(key not in public for key in candidates):
        raise ValueError('unknown or future candidate')
    ordered = sorted(candidates, key=lambda key: (public[key].step, public[key].view, key))
    count = min(count, len(ordered))
    if strategy == 'recent':
        return tuple(ordered[-count:]) if count else ()
    if strategy == 'uniform':
        # Centre for one read, endpoints plus even temporal coverage otherwise.
        indices = ([len(ordered) // 2] if count == 1 else
                   [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)])
        return tuple(ordered[i] for i in indices)
    raise ValueError('unknown public control')


@dataclass(frozen=True)
class InputTrace:
    """Actual processor output; sequence_tokens includes text AND visual tokens."""
    stage: str
    archive_fingerprint: str
    decision_step: int
    observation_keys: tuple
    visual_tokens: int
    sequence_tokens: int
    preprocessing_seconds: float
    context_sha256: str

    def validate(self, archive):
        if len(self.context_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.context_sha256):
            raise ValueError('missing public text/state/model context fingerprint')
        if self.archive_fingerprint != archive.fingerprint(self.decision_step):
            raise ValueError('archive fingerprint mismatch')
        archive.read(self.observation_keys, self.decision_step)
        if (not isinstance(self.visual_tokens, int) or self.visual_tokens < 0 or
                not isinstance(self.sequence_tokens, int) or self.sequence_tokens < self.visual_tokens or
                not math.isfinite(self.preprocessing_seconds) or self.preprocessing_seconds < 0):
            raise ValueError('invalid measured processor cost')


@dataclass(frozen=True)
class ReadBudget:
    image_reads: int
    visual_tokens: int
    sequence_tokens: int
    policy_calls: int

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in self.__dict__.values()):
            raise ValueError('budget values must be nonnegative integers')


class ReadLedger:
    """Validate cumulative reads, including repeated anchors across forward calls.

    Does not equate token count to wall-clock compute. Unknown policy/encoder
    timings stay unknown and block an end-to-end cost claim.
    """
    def __init__(self, archive, decision_step, budget):
        self.archive, self.step, self.budget = archive, decision_step, budget
        self.traces = []

    def charge(self, trace):
        trace.validate(self.archive)
        if trace.decision_step != self.step:
            raise ValueError('mixed decision steps')
        traces = [*self.traces, trace]
        used = ReadBudget(sum(len(t.observation_keys) for t in traces),
                          sum(t.visual_tokens for t in traces),
                          sum(t.sequence_tokens for t in traces), len(traces))
        if any(getattr(used, field) > getattr(self.budget, field) for field in used.__dict__):
            raise ValueError('cumulative read budget exceeded')
        self.traces.append(trace)
        return used


@dataclass(frozen=True)
class Proposal:
    """A traced pre-selection forward; never reuse a memory-conditioned action."""
    trace: InputTrace
    seed: int
    action_sha256: str
    policy_seconds: float | None
    source: str


def validate_proposal(proposal, archive, decision_step, allowed_anchor_keys,
                      selection_started, expected_context_sha256, require_measured_policy=True):
    proposal.trace.validate(archive)
    if selection_started or proposal.trace.stage != 'proposal':
        raise ValueError('proposal must precede selection')
    if proposal.trace.decision_step != decision_step:
        raise ValueError('stale or future proposal')
    if proposal.trace.observation_keys != tuple(allowed_anchor_keys):
        raise ValueError('proposal used inputs outside the common anchor context')
    if proposal.trace.context_sha256 != expected_context_sha256:
        raise ValueError('proposal public text/state/model context differs')
    if (not isinstance(proposal.seed, int) or len(proposal.action_sha256) != 64 or
            any(c not in '0123456789abcdef' for c in proposal.action_sha256)):
        raise ValueError('missing proposal provenance')
    if proposal.policy_seconds is not None and (not math.isfinite(proposal.policy_seconds) or proposal.policy_seconds < 0):
        raise ValueError('invalid policy timing')
    if require_measured_policy and (proposal.source != 'live_anchor_only_forward' or proposal.policy_seconds is None):
        raise ValueError('synthetic/cached/unmeasured proposal cannot enter closed-loop evaluation')
    return digest(proposal.__dict__ | {'trace': proposal.trace.__dict__})
