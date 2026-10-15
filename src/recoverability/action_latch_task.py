"""T3 mechanism task: lifting an armed cover resets its visual indicator.

This is an explicitly designed simulator mechanism, not an original RoboTwin
task. Only the environment reads cover poses. Controllers get RGB and joints.
"""
from contextlib import contextmanager
import numpy as np
from recoverability.cue_probe_task import revisit_or_erase_cue_probe_v0


class action_latch_v1(revisit_or_erase_cue_probe_v0):
    def load_actors(self):
        super().load_actors()
        self.latch_armed = False
        self.latch_events = []
        self.latch_ticks = 0
        self.latch_origin = None

    def arm_latches(self):
        if self.latch_events:
            raise RuntimeError('Rebuild the scene before another intervention')
        self.latch_origin = [np.asarray(c.get_pose().p).copy() for c in self.covers]
        self.latch_armed = True
        self.latch_ticks = 0

    def tick_latches(self):
        if not self.latch_armed:
            return
        self.latch_ticks += 1
        changed = {e['slot'] for e in self.latch_events}
        for slot, cover in enumerate(self.covers):
            lift = float(cover.get_pose().p[2] - self.latch_origin[slot][2])
            if slot not in changed and lift > 0.04:
                self._set_indicator_colour(slot, (0.5, 0.5, 0.5))
                self.latch_events.append(dict(slot=slot, tick=self.latch_ticks, lift_m=lift))


@contextmanager
def physical_latch_steps(env):
    """Instrument every physics tick, including inside upstream take_action()."""
    original = env.scene
    class SceneWithLatch:
        def __getattr__(self, name):
            return getattr(original, name)
        def step(self):
            original.step()
            env.tick_latches()
    env.scene = SceneWithLatch()
    try:
        yield
    finally:
        env.scene = original

