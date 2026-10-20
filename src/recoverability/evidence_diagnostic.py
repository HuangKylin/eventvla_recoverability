"""Pure contracts for the bounded native evidence-validity diagnostic."""
TASKS = ('put_back_block_hard', 'rearrange_blocks_hard')
SEEDS = tuple(range(520000, 520006))


def validate_native_jobs(jobs):
    expected = [(task, seed) for task in TASKS for seed in SEEDS]
    if jobs != expected:
        raise ValueError('Native task/seed order must match the fixed twelve-job scope')


def candidate_ready(task, state, chunk_finished, pending_commit_done):
    if not chunk_finished or not pending_commit_done or state.get('fail_flag'):
        return False
    if task == TASKS[0]:
        return (state.get('first_center_return_achieved', False) and
                state.get('second_center_return_achieved', False) and
                state.get('press_cnt', 0) >= 2 and state.get('both_on_center', False))
    if task == TASKS[1]:
        return state.get('phase') == 'move_other_to_center'
    raise ValueError('Task outside fixed scope')


def validate_replacements(spec, event_count, parent_step, observed_steps):
    if len(spec) > event_count or len({x['slot'] for x in spec}) != len(spec):
        raise ValueError('Invalid replacement slots')
    for item in spec:
        if (type(item['slot']) is not int or not 0 <= item['slot'] < event_count or
                type(item['source_step']) is not int or item['source_step'] >= parent_step or
                item['source_step'] not in observed_steps):
            raise ValueError('Replacement must use a real, strictly past observed frame')


def inference_seed(episode_seed, request_index):
    if episode_seed not in SEEDS or not 0 <= request_index < 1000:
        raise ValueError('Inference seed outside fixed scope')
    return episode_seed * 1000 + request_index
