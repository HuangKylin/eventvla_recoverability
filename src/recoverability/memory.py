"""Bounded selectors operating on the upstream EventVLA image-memory entries.

No discarded images, features or semantic summaries are retained here. Uniform
sampling uses deterministic independent priorities, so it needs no full archive.
The optional scores input is an interface for a later trained scorer; no trained
scorer or recoverability labels are implemented by this module.
"""
import hashlib
import math
from collections.abc import Mapping, Sequence


def select_entries(entries: Sequence[Mapping], capacity: int, *, strategy='fifo',
                   seed: int = 42, scores: Sequence[float] | None = None) -> list:
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
        raise ValueError('capacity must be a non-negative integer')
    if strategy not in {'fifo', 'uniform', 'scored'}:
        raise ValueError('Unsupported memory strategy: ' + str(strategy))
    if strategy == 'scored':
        if scores is None or len(scores) != len(entries):
            raise ValueError('One score is required per candidate')
        if not all(math.isfinite(float(v)) for v in scores):
            raise ValueError('Scores must be finite')
    # Revisiting the same committed timestep updates its payload, not capacity.
    by_step = {}
    for i, entry in enumerate(entries):
        step = entry.get('step')
        if not isinstance(step, int):
            raise ValueError('Each entry must have an integer step')
        by_step[step] = (entry, i)
    if capacity == 0:
        return []
    candidates = list(by_step.values())

    def key(pair):
        entry, original_index = pair
        step = entry['step']
        if strategy == 'fifo':
            return (step,)
        if strategy == 'uniform':
            priority = hashlib.blake2b(f'{seed}:{step}'.encode(), digest_size=16).digest()
            return (priority, step)
        return (float(scores[original_index]), step)

    kept = sorted(candidates, key=key, reverse=True)[:capacity]
    return [entry for entry, _ in sorted(kept, key=lambda pair: pair[0]['step'])]


def image_payload_bytes(entries: Sequence[Mapping]) -> int:
    """Array payload only; index metadata, temporal anchors and RSS are separate."""
    return sum(int(image.nbytes) for entry in entries for image in entry.get('images', []))
