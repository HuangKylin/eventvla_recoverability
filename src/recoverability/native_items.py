"""Single-event interventions and public-location recovery for label pilots.

Location is read from the candidate's past RGB BEFORE eviction. Only its slot
index survives as routing metadata; no colour, simulator truth or target index
is supplied to the recovery controller. Routing metadata must be budgeted by
all methods in later capacity experiments.
"""
import numpy as np

from .native_memory import metadata_hash
from .vla_intervention import image_hash, purge_evidence


def event_signature(entries):
    return [(int(e['step']), metadata_hash([e]), tuple(image_hash(im) for im in e['images']))
            for e in entries]


def intervene_item(client, candidate_step, *, drop):
    if client.action_mode != 'abs':
        raise ValueError('Single-event intervention requires absolute actions')
    entries = client.keyframe_image_memory
    steps = [int(e['step']) for e in entries]
    candidate_step = int(candidate_step)
    if steps != sorted(set(steps)) or steps.count(candidate_step) != 1:
        raise ValueError('Candidate must identify exactly one ordered event')
    if candidate_step == client.initial_step or any(int(r['step']) == candidate_step for r in client.image_history):
        raise ValueError('Candidate overlaps the first frame or temporal history')
    candidate = next(e for e in entries if int(e['step']) == candidate_step)
    candidate_hashes = [image_hash(im) for im in candidate['images']]
    other = [e for e in entries if int(e['step']) != candidate_step]
    if set(candidate_hashes) & {image_hash(im) for e in other for im in e['images']}:
        raise ValueError('Candidate image duplicated in another event; identity is ambiguous')
    before = event_signature(other)
    result = purge_evidence(client, [candidate_step] if drop else [])
    after = event_signature([e for e in client.keyframe_image_memory if int(e['step']) != candidate_step])
    if before != after:
        raise RuntimeError('Intervention modified a noncandidate event')
    result.update(candidate_step=candidate_step, candidate_image_hashes=candidate_hashes,
                  noncandidate_signature=before, noncandidate_unchanged=True,
                  intervention='drop_one' if drop else 'keep_one')
    return result


def candidate_slot(entry, reader):
    if len(entry['images']) != 1:
        raise ValueError('Expected one past head-camera image')
    visible = [slot for slot in range(4) if reader(slot, entry['images'][0]).answer is not None]
    if len(visible) != 1:
        raise ValueError('Past RGB must identify exactly one visible public slot')
    return visible[0]


def replace_item_content(client, candidate_step, replacement):
    """Diagnostic pixel substitution; preserves event count/order/metadata.

    The caller supplies an actually observed, cue-free image. The substituted
    record is a causal input probe, not a claim that this was the historical
    observation at the original timestamp. Never export it as a K/R label.
    """
    entries=[e for e in client.keyframe_image_memory if int(e['step'])==int(candidate_step)]
    if len(entries)!=1 or len(entries[0]['images'])!=1:raise ValueError('Need one head-camera candidate')
    old=np.asarray(entries[0]['images'][0]);new=np.asarray(replacement)
    if new.shape!=old.shape or new.dtype!=old.dtype:raise ValueError('Replacement shape/dtype differs')
    if image_hash(new)==image_hash(old):raise ValueError('Replacement must change content')
    result=intervene_item(client,candidate_step,drop=False)
    entries[0]['images']=[new.copy()]
    result.update(intervention='replace_one_content',forbidden_image_hashes=result['candidate_image_hashes'],
                  replacement_image_hash=image_hash(new),synthetic_input_diagnostic=True)
    return result


def recover_candidate(env, controller, slot, *, observe, after_tick=None, after_slot=None, max_attempts=3):
    """Attempt one caller-selected public slot, close it, then return the robot."""
    from .fixed_skills import measured_proprioception
    from .native_recovery import bridge_to
    from .snapshot import apply_drive_vector
    if slot not in range(4) or max_attempts not in (1, 2, 3):
        raise ValueError('Invalid public slot or retry budget')
    return_pose = measured_proprioception(env.robot).copy()
    bridges = []; attempts = []
    for attempt in range(max_attempts):
        bridges.append(bridge_to(env, controller.skills[slot].start_proprioception, after_tick=after_tick))
        skill, error = controller.select(slot, measured_proprioception(env.robot))
        visible = None
        for i, drive in enumerate(skill.drives):
            apply_drive_vector(env, drive); env.scene.step()
            if after_tick is not None: after_tick()
            if i + 1 == skill.observe_after_steps: visible = observe('visible', slot)
        if visible is None: raise RuntimeError('Missing scheduled observation')
        closed = observe('closed', slot)
        colour = controller.read(slot, visible['images']['head_camera'])
        closed_colour = controller.read(slot, closed['images']['head_camera'])
        attempts.append(dict(attempt=attempt+1, read=colour.__dict__, closed_read=closed_colour.__dict__))
        if closed_colour.answer is not None: raise RuntimeError('Evidence remained visible after recovery')
        if colour.answer is not None: break
    record = dict(slot=slot, read=colour.__dict__, closed_read=closed_colour.__dict__,
                  skill_physics_steps=len(skill.drives)*len(attempts), start_error_rad=error, attempts=attempts)
    if after_slot is not None: after_slot(record)
    bridges.append(bridge_to(env, return_pose, after_tick=after_tick))
    final = observe('returned', -1)
    return dict(records=[record], bridges=bridges,
                colours_hidden=all(controller.read(s, final['images']['head_camera']).answer is None for s in range(4)),
                grippers_open=bool(np.all(measured_proprioception(env.robot)[[6, 13]] > .8)),
                final_proprioception=measured_proprioception(env.robot).tolist(),
                recovered_slots=[slot] if colour.answer is not None else [],
                unrecovered_slots=[] if colour.answer is not None else [slot])
