"""Eight independent versioned cues in a four-slot fixed workstation.

Only evaluator/apparatus code may import this module. The controller receives
RGB, robot proprioception, and public slot/version commands. Version advancement
is an explicit apparatus operation, not yet a learned robot manipulation.
"""
import numpy as np
from recoverability.cue_probe_task import revisit_or_erase_cue_probe_v0


class multi_cue_versions_v1(revisit_or_erase_cue_probe_v0):
    def load_actors(self):
        super().load_actors()
        generator = np.random.default_rng(self.probe_seed)
        self._answer_ids = generator.integers(0, 4, size=(2, 4))
        self._versions = [0] * 4
        for slot in range(4):
            self._set_indicator_colour(slot, self.color_tuple[self._answer_ids[0, slot]])
        self.phase = 'multi_cue_observe'

    def advance_indicator(self, slot):
        if slot not in range(4) or self._versions[slot] != 0:
            raise ValueError('An indicator can advance from version zero exactly once')
        self._versions[slot] = 1
        self._set_indicator_colour(slot, self.color_tuple[self._answer_ids[1, slot]])

    def score_answer(self, slot, version, answer):
        # Called only after the answer has left the controller.
        return answer == self.color_names[self._answer_ids[version, slot]]

    def check_success(self):
        # Per-query outcomes are scored by the evaluator, not by robot.step().
        return False
