"""Evaluator-only branch eligibility and absolute-reference cover events."""
import numpy as np


def assess_parent(*, positions, initial_positions, block_positions, grippers,
                  phase, inspect_pointer, final_pointer, failed,
                  step, plan_start, plan_length, settled_steps):
    positions = np.asarray(positions, dtype=float)
    initial = np.asarray(initial_positions, dtype=float)
    blocks = np.asarray(block_positions, dtype=float)
    if positions.shape != (4, 3) or initial.shape != (4, 3) or blocks.shape != (4, 3):
        raise ValueError('Expected four cover and block positions')
    if not all(np.isfinite(x).all() for x in (positions, initial, blocks, grippers)):
        raise ValueError('Nonfinite evaluator state')
    lift = positions[:, 2] - initial[:, 2]
    offsets = np.linalg.norm(positions[:, :2] - blocks[:, :2], axis=1)
    chunk_end = plan_start is not None and plan_length > 0 and step == plan_start + plan_length
    checks = dict(final_phase=phase == 'final' and inspect_pointer == 4,
                  before_first_final_open=final_pointer == 0 and not failed,
                  covers_down=bool(np.all(np.abs(lift) < .01)),
                  covers_over_blocks=bool(np.all(offsets < .03)),
                  grippers_open=bool(np.all(np.asarray(grippers) > .8)),
                  natural_chunk_end=bool(chunk_end), settled=settled_steps >= 3)
    return dict(eligible=all(checks.values()), checks=checks, cover_lifts_m=lift.tolist(),
                cover_block_offsets_m=offsets.tolist(), grippers=list(map(float, grippers)),
                step=int(step), plan_start=plan_start, plan_length=int(plan_length))


class AbsoluteLiftTracker:
    """Measure from initial resting heights, never from an elevated branch parent.

    Three consecutive physics samples reject one-sample jitter. Simultaneous
    first crossings remain a set; iteration order must not select the target.
    """
    def __init__(self, initial_z, threshold=.04, persistence=3):
        self.initial_z = np.asarray(initial_z, dtype=float)
        self.threshold = threshold
        self.persistence = persistence
        self.counts = np.zeros(len(self.initial_z), dtype=int)
        self.seen = set()
        self.tick = 0
        self.events = []

    def update(self, heights):
        lifts = np.asarray(heights, dtype=float) - self.initial_z
        if lifts.shape != self.initial_z.shape or not np.isfinite(lifts).all():
            raise ValueError('Invalid cover heights')
        self.tick += 1
        self.counts = np.where(lifts > self.threshold, self.counts + 1, 0)
        slots = [int(i) for i in range(len(lifts)) if self.counts[i] >= self.persistence and i not in self.seen]
        if slots:
            self.events.append(dict(tick=self.tick, slots=slots, lifts_m=[float(lifts[i]) for i in slots]))
            self.seen.update(slots)

    def first_target(self, target):
        return bool(self.events and self.events[0]['slots'] == [target])
