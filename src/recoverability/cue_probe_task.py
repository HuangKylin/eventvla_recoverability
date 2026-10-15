"""Controlled T1 apparatus probe; not a RoboTwin benchmark or a VLA task score.

The covered coloured block represents a resettable visual indicator. A public
apparatus command either leaves its value intact or resets it to neutral grey.
Robot inspection is real contact simulation. The reset command is an explicit
high-level apparatus action, not a learned 14-DoF robot action.
"""
import numpy as np
from envs.cover_blocks_hard import cover_blocks_hard


class revisit_or_erase_cue_probe_v0(cover_blocks_hard):
    def setup_demo(self, **kwargs):
        self.probe_seed = int(kwargs.get('seed', 0))
        super().setup_demo(**kwargs)

    def _set_indicator_colour(self, slot, colour):
        from sapien.render import RenderBodyComponent
        changed = 0
        for component in self.blocks[slot].actor.get_components():
            if isinstance(component, RenderBodyComponent):
                for shape in component.render_shapes:
                    shape.material.set_base_color(list(colour) + [1.0])
                    changed += 1
        if not changed:
            raise RuntimeError('No indicator material was updated')

    def load_actors(self):
        super().load_actors()
        # Independent colours with replacement: other slots cannot reveal this
        # answer by eliminating colours from a known permutation.
        colour_ids = np.random.default_rng(self.probe_seed).integers(0, 4, size=4)
        for slot, colour_id in enumerate(colour_ids):
            self._set_indicator_colour(slot, self.color_tuple[colour_id])
            self.block_color_names[slot] = self.color_names[colour_id]
        self.phase = 'probe_inspection'
        self.probe_slot = 0
        self.probe_erased = False
        self.probe_command = None
        self.probe_answer = None
        self.probe_submitted = False
        self.probe_inspections = 0

    def _mark_inspection_complete(self, slot_id):
        # Inspection stays legal. Original cover_blocks_hard is not modified.
        self.probe_inspections += 1

    def update_state_transition(self):
        pass

    def apply_apparatus_command(self, command):
        if command not in {'preserve', 'erase'}:
            raise ValueError('Unknown public apparatus command')
        if self.probe_command is not None:
            raise ValueError('The intervention is one-shot; rebuild to branch again')
        self.probe_command = command
        if command == 'erase':
            self._set_indicator_colour(self.probe_slot, (0.5, 0.5, 0.5))
            self.probe_erased = True

    def submit_answer(self, answer):
        if self.probe_submitted:
            raise ValueError('Answer was already submitted')
        if answer is not None and answer not in self.color_names:
            raise ValueError('Answer is not a supported colour or abstention')
        self.probe_answer = answer
        self.probe_submitted = True
        return self.check_success()

    def check_success(self):
        # Evaluator-only truth. No truth field is passed to the colour reader.
        return bool(self.probe_submitted and
                    self.probe_answer == self.block_color_names[self.probe_slot])
