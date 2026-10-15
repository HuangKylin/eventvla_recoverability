"""Same-scene snapshots for a deliberately narrow, verifiable controller contract.

Only fixed drive replay is supported. Native cuRobo workers, arbitrary VLA clients,
renderer RNG, and dynamically-created entities are NOT serialised by this module.
The caller must validate physical continuation before accepting paired labels.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import pickle
import random

import numpy as np
import torch


BASE_FIELDS = (
    'FRAME_IDX', 'plan_success', 'take_action_cnt', 'eval_success', 'max_reward',
    'stage_success_tag', 'left_cnt', 'right_cnt', 'left_joint_path', 'right_joint_path',
    'instruction', 'language_annotation', 'language_annotation_cache', 'now_obs', 'info',
)
COVER_FIELDS = (
    'phase', 'inspect_pointer', 'inspect_reveal_slot', 'final_pointer', 'fail_flag',
    'keyframe_steps', '_tracked_keyframe_slot', '_tracked_best_keyframe_step',
    '_tracked_best_keyframe_dist',
)
ROBOT_FIELDS = ('left_gripper_val', 'right_gripper_val', 'left_js', 'right_js')


def digest(value):
    # Internal in-memory hashing only. Never deserialize an untrusted pickle.
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def articulations(env):
    result = [env.robot.left_entity]
    other = getattr(env.robot, 'right_entity', None)
    if other is not None and all(other is not item for item in result):
        result.append(other)
    return result


def drive_vector(env):
    parts = []
    for art in articulations(env):
        for joint in art.get_active_joints():
            parts.extend((np.asarray(joint.get_drive_target()).ravel(),
                          np.asarray(joint.get_drive_velocity_target()).ravel()))
        parts.append(np.asarray(art.get_qf()).ravel())
    parts.append(np.asarray([env.robot.left_gripper_val, env.robot.right_gripper_val]))
    return np.concatenate(parts).astype(np.float64)


def apply_drive_vector(env, vector):
    cursor = 0
    for art in articulations(env):
        for joint in art.get_active_joints():
            n = len(joint.get_drive_target())
            joint.set_drive_target(vector[cursor:cursor+n].astype(np.float32))
            cursor += n
            joint.set_drive_velocity_target(vector[cursor:cursor+n].astype(np.float32))
            cursor += n
        n = len(art.get_qf())
        art.set_qf(vector[cursor:cursor+n].astype(np.float32))
        cursor += n
    env.robot.left_gripper_val, env.robot.right_gripper_val = vector[cursor:cursor+2].tolist()
    if cursor + 2 != len(vector):
        raise ValueError('Drive vector does not match this scene')


def physics_vector(env):
    """Evaluator-only fingerprint, including every entity and all dynamic velocities."""
    from sapien.physx import PhysxRigidDynamicComponent
    parts = []
    for entity in env.scene.get_entities():
        pose = entity.get_pose()
        parts.extend((pose.p, pose.q))
        for component in entity.get_components():
            if isinstance(component, PhysxRigidDynamicComponent):
                parts.extend((component.get_linear_velocity(), component.get_angular_velocity()))
    for art in articulations(env):
        parts.extend((art.get_qpos(), art.get_qvel(), art.get_root_linear_velocity(),
                      art.get_root_angular_velocity()))
    parts.append(drive_vector(env))
    return np.concatenate([np.asarray(p).ravel() for p in parts]).astype(np.float64)


def max_error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float('inf')
    return float(np.max(np.abs(a-b), initial=0))


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state().numpy().copy(),
                torch_cuda=[x.cpu().numpy().copy() for x in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(torch.from_numpy(state['torch_cpu'].copy()))
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all([torch.from_numpy(x.copy()) for x in state['torch_cuda']])


@dataclass
class Snapshot:
    scene_identity: int
    entity_ids: tuple
    native_physics: bytes
    poses: bytes
    drives: np.ndarray
    task_state: dict
    robot_state: dict
    controller_state: dict
    random_state: dict
    physical_state: np.ndarray
    parent_hash: str
    extra_fields: tuple


def capture(env, controller_state, *, mode, extra_fields=()):
    if mode != 'fixed_drive_replay':
        raise ValueError('Only fixed_drive_replay is supported; planner/VLA state is not captured')
    if env.save_data or getattr(env, 'eval_video_path', None):
        raise ValueError('Branching with active external video/data writers is unsupported')
    if getattr(env, 'crazy_random_light', False):
        raise ValueError('Mutable renderer lights are not part of the snapshot contract')
    fields = BASE_FIELDS + COVER_FIELDS + tuple(extra_fields)
    task_state = {key: deepcopy(getattr(env, key)) for key in fields if hasattr(env, key)}
    robot_state = {key: deepcopy(getattr(env.robot, key)) for key in ROBOT_FIELDS
                   if hasattr(env.robot, key)}
    controller = deepcopy(controller_state)
    physical, rng = physics_vector(env), rng_state()
    payload = (physical, task_state, robot_state, controller, rng)
    return Snapshot(id(env.scene), tuple(x.get_global_id() for x in env.scene.get_entities()),
                    bytes(env.scene.get_physx_system().pack()), bytes(env.scene.pack_poses()),
                    drive_vector(env), task_state, robot_state, controller, rng, physical,
                    digest(payload), tuple(extra_fields))


def restore(env, snapshot):
    if (id(env.scene) != snapshot.scene_identity or
            tuple(x.get_global_id() for x in env.scene.get_entities()) != snapshot.entity_ids):
        raise ValueError('Snapshot belongs to another scene or scene topology changed')
    env.scene.unpack_poses(snapshot.poses)
    env.scene.get_physx_system().unpack(snapshot.native_physics)
    apply_drive_vector(env, snapshot.drives)
    for key, value in snapshot.task_state.items():
        setattr(env, key, deepcopy(value))
    for key, value in snapshot.robot_state.items():
        setattr(env.robot, key, deepcopy(value))
    restore_rng(snapshot.random_state)
    return deepcopy(snapshot.controller_state), max_error(physics_vector(env), snapshot.physical_state)


@contextmanager
def forbid_planning(robot):
    """Fail closed if a replay branch accidentally consults an un-restored planner."""
    names = [name for name in dir(robot) if name.startswith(('left_plan_', 'right_plan_'))]
    original = {name: getattr(robot, name) for name in names}
    def forbidden(*args, **kwargs):
        raise RuntimeError('Planning is outside the fixed-drive snapshot contract')
    for name in names:
        setattr(robot, name, forbidden)
    try:
        yield
    finally:
        for name, method in original.items():
            setattr(robot, name, method)


class DriveRecorder:
    """A temporary scene proxy recording the drives before each real physics step."""
    def __init__(self, env):
        self.env, self.native_scene = env, env.scene
        self.rows = []
    def __getattr__(self, name):
        return getattr(self.native_scene, name)
    def step(self):
        self.rows.append(drive_vector(self.env))
        self.native_scene.step()
    def __enter__(self):
        self.env.scene = self
        return self
    def __exit__(self, *args):
        self.env.scene = self.native_scene


def replay(env, tape, *, sample_every=100, after_step=None):
    traces = []
    with forbid_planning(env.robot):
        for i, row in enumerate(tape):
            apply_drive_vector(env, row)
            env.scene.step()
            if after_step is not None:
                after_step(i)
            if i % sample_every == 0 or i == len(tape)-1:
                traces.append(physics_vector(env))
    return np.asarray(traces)
