"""Public recovery/commit protocol and proprioceptive fixed-workstation bridge.

Recovery is an explicit task mode entered BEFORE manipulation. It never clears
a failure flag. RGB, robot joints, and the shared demonstration library are the
only inputs to motion selection; object truth is reserved for external scoring.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np

ARM = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])


@dataclass
class RecoveryProtocol:
    mode: str = 'ready'
    physics_steps: int = 0
    suppressed_updates: int = 0
    max_physics_steps: int = 30000

    def begin(self, *, already_failed=False):
        if self.mode != 'ready' or already_failed:
            raise RuntimeError('Recovery must start before submission or failure')
        self.mode = 'recover'

    def tick(self):
        if self.mode != 'recover': raise RuntimeError('Recovery action outside recovery mode')
        if self.physics_steps >= self.max_physics_steps:
            raise RuntimeError('Recovery physics budget exhausted')
        self.physics_steps += 1

    def update(self, native_update):
        if self.mode == 'recover': self.suppressed_updates += 1
        elif self.mode == 'commit': return native_update()
        else: raise RuntimeError('Choose recovery or commit before a task update')

    def commit(self, *, colours_hidden, grippers_open):
        if self.mode not in ('ready', 'recover') or not colours_hidden or not grippers_open:
            raise RuntimeError('Submission requires closed observations and released grippers')
        self.mode = 'commit'


@contextmanager
def recovery_task_mode(env, protocol):
    native_update = env.update_state_transition
    original_scene = env.scene
    def update(): return protocol.update(native_update)
    class Scene:
        def __getattr__(self, name): return getattr(original_scene, name)
        def step(self):
            if protocol.mode == 'recover': protocol.tick()
            original_scene.step()
            if protocol.mode == 'recover': update()
    env.update_state_transition = update
    env.scene = Scene()
    try: yield
    finally:
        env.update_state_transition = native_update
        env.scene = original_scene


def joint_bridge(start, goal, *, dt=.004, max_speed=.4, settle_steps=200):
    """Smooth joint-space commands, with both grippers open; no geometric planner."""
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    if start.shape != (14,) or goal.shape != (14,) or not np.isfinite([start, goal]).all():
        raise ValueError('Need finite 14D proprioception')
    if dt <= 0 or max_speed <= 0 or settle_steps < 1: raise ValueError('Invalid bridge limits')
    delta = goal[ARM] - start[ARM]
    n = max(1, int(np.ceil(1.5*np.max(np.abs(delta))/(max_speed*dt))))
    t = np.arange(1, n+1, dtype=float)/n
    position = start[ARM][None, :] + (3*t*t-2*t*t*t)[:, None]*delta
    velocity = (6*t-6*t*t)[:, None]*delta/(n*dt)
    for q, dq in zip(position, velocity): yield q, dq
    for _ in range(settle_steps): yield goal[ARM].copy(), np.zeros(12)


def bridge_to(env, target, *, after_tick=None):
    from .fixed_skills import measured_proprioception
    start = measured_proprioception(env.robot)
    n = 0
    for q, dq in joint_bridge(start, target):
        env.robot.set_arm_joints(q[:6], dq[:6], 'left')
        env.robot.set_arm_joints(q[6:], dq[6:], 'right')
        env.robot.set_gripper(1., 'left'); env.robot.set_gripper(1., 'right')
        env.scene.step(); n += 1
        if after_tick is not None: after_tick()
    actual = measured_proprioception(env.robot)
    error = float(np.max(np.abs(actual[ARM]-np.asarray(target)[ARM])))
    if error > .15: raise RuntimeError(f'Joint bridge did not converge: {error}')
    return dict(physics_steps=n, start=start.tolist(), goal=np.asarray(target).tolist(), end_error_rad=error)


def sweep_recover(env, controller, *, observe, after_tick=None, after_slot=None, max_attempts=3):
    """Scan every public slot in fixed order. No true target/seed/colour argument."""
    from .fixed_skills import measured_proprioception
    from .snapshot import apply_drive_vector
    return_pose = measured_proprioception(env.robot).copy()
    if max_attempts not in (1,2,3):raise ValueError('Recovery attempts must be bounded to 1..3')
    records = []; bridges = []
    for slot in range(4):
        attempts=[]
        for attempt in range(max_attempts):
            bridges.append(bridge_to(env, controller.skills[slot].start_proprioception, after_tick=after_tick))
            skill, error = controller.select(slot, measured_proprioception(env.robot))
            visible = None
            for i, drive in enumerate(skill.drives):
                apply_drive_vector(env, drive); env.scene.step()
                if after_tick is not None: after_tick()
                if i+1 == skill.observe_after_steps: visible = observe('visible', slot)
            if visible is None: raise RuntimeError('No scheduled recovery observation')
            closed = observe('closed', slot)
            colour = controller.read(slot, visible['images']['head_camera'])
            closed_colour = controller.read(slot, closed['images']['head_camera'])
            attempts.append(dict(attempt=attempt+1,read=colour.__dict__,closed_read=closed_colour.__dict__))
            if closed_colour.answer is not None:raise RuntimeError(f'Cover did not hide evidence after attempt: {attempts}')
            if colour.answer is not None:break
        record = dict(slot=slot, read=colour.__dict__, closed_read=closed_colour.__dict__,
                      skill_physics_steps=len(skill.drives)*len(attempts), start_error_rad=error, attempts=attempts)
        records.append(record)
        # A bounded miss is an explicit unrecovered item, not invented evidence.
        # Unsafe closure still raises above; continue to the other public slots.
        if after_slot is not None: after_slot(record)
    bridges.append(bridge_to(env, return_pose, after_tick=after_tick))
    final = observe('returned', -1)
    hidden = all(controller.read(s, final['images']['head_camera']).answer is None for s in range(4))
    opened = bool(np.all(measured_proprioception(env.robot)[[6, 13]] > .8))
    return dict(records=records, bridges=bridges, colours_hidden=hidden,
                grippers_open=opened, final_proprioception=measured_proprioception(env.robot).tolist(),
                recovered_slots=[r['slot'] for r in records if r['read']['answer'] is not None],
                unrecovered_slots=[r['slot'] for r in records if r['read']['answer'] is None])
