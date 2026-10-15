"""Public-observation controller for a calibrated fixed workstation.

Offline expert demonstrations are allowed. At evaluation there is one immutable
skill library for every seed/memory method. No per-test scene planning, object
pose, colour truth, seed, or task instance enters SharedRecoveryController.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from recoverability.cue_reader import read_colour


SLOT_ROIS = ((0.04, 0.45, 0.26, 0.95), (0.30, 0.45, 0.49, 0.95),
             (0.56, 0.45, 0.76, 0.95), (0.81, 0.45, 0.995, 0.95))


def measured_proprioception(robot):
    """Joint positions measured by the robot, plus its commanded gripper openings."""
    return np.asarray(robot.get_left_arm_real_jointState() +
                      robot.get_right_arm_real_jointState(), dtype=np.float64)


@dataclass(frozen=True)
class Skill:
    slot: int
    drives: np.ndarray
    observe_after_steps: int
    start_proprioception: np.ndarray


class SharedRecoveryController:
    def __init__(self, library, start_tolerance=0.15):
        self.path = Path(library).resolve()
        self.manifest = json.loads(self.path.read_text())
        if self.manifest['status'] != 'complete':
            raise ValueError('Skill library is incomplete')
        self.library_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.start_tolerance = start_tolerance
        self.skills = {}
        for item in self.manifest['skills']:
            file = self.path.parent / item['file']
            if hashlib.sha256(file.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Skill file checksum mismatch')
            with np.load(file, allow_pickle=False) as data:
                self.skills[item['slot']] = Skill(item['slot'], data['drives'].copy(),
                    item['observe_after_steps'], data['start_proprioception'].copy())
        if set(self.skills) != set(range(4)):
            raise ValueError('All four public slots require a calibrated skill')

    def select(self, slot, proprioception):
        if slot not in self.skills:
            raise ValueError('Unknown public slot')
        state = np.asarray(proprioception)
        skill = self.skills[slot]
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('Invalid robot proprioception')
        # Gripper state is excluded from the arm start guard; every skill opens
        # its gripper during its approach. No simulator object pose is consulted.
        indices = [0,1,2,3,4,5,7,8,9,10,11,12]
        error = float(np.max(np.abs(state[indices] - skill.start_proprioception[indices])))
        if error > self.start_tolerance:
            raise RuntimeError(f'Robot is outside the calibrated start envelope: {error:.6f} rad')
        return skill, error

    @staticmethod
    def read(slot, rgb):
        return read_colour(rgb, roi=SLOT_ROIS[slot])


@contextmanager
def deny_oracle_actions(env):
    """Runtime tripwires around expert geometry/planning entry points.

This is an interface audit, not an OS sandbox. Controller code has no env handle.
"""
    names = ('inspect_cover', 'grasp_actor', 'place_actor', 'get_grasp_pose',
             'choose_grasp_pose', 'get_place_pose', '_target_pose', 'get_keyframe_oracle_info')
    saved = {name: getattr(env, name) for name in names}
    robot_saved = {name: getattr(env.robot, name) for name in dir(env.robot)
                   if name.startswith(('left_plan_', 'right_plan_'))}
    audit = {'forbidden_calls': 0}
    def forbidden(*args, **kwargs):
        audit['forbidden_calls'] += 1
        raise RuntimeError('Runtime expert geometry or planning access is forbidden')
    for name in saved:
        setattr(env, name, forbidden)
    for name in robot_saved:
        setattr(env.robot, name, forbidden)
    try:
        yield audit
    finally:
        for name, value in saved.items():
            setattr(env, name, value)
        for name, value in robot_saved.items():
            setattr(env.robot, name, value)


def execute_skill(env, controller, slot, *, capture_closed=True):
    """Actuator/observation adapter. It never calls task or object geometry methods."""
    from recoverability.snapshot import apply_drive_vector
    from recoverability.sim_runtime import public_observation
    import time
    start = time.monotonic()
    skill, start_error = controller.select(slot, measured_proprioception(env.robot))
    visible = None
    # Motor targets are immutable demonstration-derived parameters, not memory
    # of task cues. Existing joint PD drives supply proprioceptive feedback.
    for index, row in enumerate(skill.drives):
        apply_drive_vector(env, row)
        env.scene.step()
        if index + 1 == skill.observe_after_steps:
            visible = public_observation(env.get_obs())['images']['head_camera']
    if visible is None:
        raise RuntimeError('Skill did not expose its scheduled observation')
    closed = public_observation(env.get_obs())['images']['head_camera'] if capture_closed else None
    read = controller.read(slot, visible)
    closed_read = None if closed is None else controller.read(slot, closed)
    return dict(image=visible, closed_image=closed, read=read,
                closed_read=closed_read, start_error_rad=start_error,
                physics_steps=len(skill.drives), seconds=time.monotonic()-start)
