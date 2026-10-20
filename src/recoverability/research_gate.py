"""CPU diagnostics for deciding whether a task warrants more learned-memory data.

These functions are diagnostics, not an online policy or formal significance test.
The action matcher accepts public motor programs only. Outcome labels enter the
separate ranking diagnostic and must never enter a deployed selector.
"""
from itertools import permutations, product
import math
import numpy as np


def match_public_action(sampled_drives, templates, tolerance=1e-8):
    """Recognize a frozen skill from its motor samples, refusing unknown/tied input."""
    query = np.asarray(sampled_drives, dtype=np.float64)
    if query.ndim != 2 or query.shape[0] != 16 or not np.isfinite(query).all():
        raise ValueError('Expected 16 finite motor samples')
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid matching tolerance')
    distances = {}
    for slot, raw in templates.items():
        template = np.asarray(raw, dtype=np.float64)
        if template.shape != query.shape or not np.isfinite(template).all():
            raise ValueError('Invalid skill template')
        distances[int(slot)] = float(np.max(np.abs(template - query)))
    matches = [slot for slot, distance in distances.items() if distance <= tolerance]
    if len(matches) != 1:
        raise ValueError('Unknown or ambiguous public motor program')
    return matches[0], distances


def branch_cost(outcome):
    steps = outcome['recovery_steps']
    if type(outcome['success']) is not bool or type(steps) is not int or steps < 0:
        raise ValueError('Invalid branch outcome')
    return float(not outcome['success']) + 0.1 * min(steps / 10000, 1)


def rank_diagnostic(rows, public_rule_slot):
    """Compare protection priority to measured single-drop marginal costs.

Do not sum these costs to predict multi-deletion or sequential query outcomes.
"""
    if len(rows) != 4 or {r['evidence_slot'] for r in rows} != set(range(4)):
        raise ValueError('Need one complete four-candidate group')
    if len({(r['family_id'], r['action_hash'], r['parent_hash']) for r in rows}) != 1:
        raise ValueError('Mixed decision states')
    costs = {}
    for row in rows:
        if row['paired_prefix_and_continuation_validated'] is not True:
            raise ValueError('Unvalidated branch pair')
        observed = branch_cost(row['recover']) - branch_cost(row['keep'])
        if not math.isfinite(row['delta_cost']) or not math.isclose(
                observed, row['delta_cost'], rel_tol=0, abs_tol=1e-10):
            raise ValueError('Stored cost disagrees with branch outcomes')
        costs[row['evidence_slot']] = observed
    maximum = max(costs.values())
    best = [slot for slot, cost in costs.items() if math.isclose(cost, maximum, abs_tol=1e-10)]
    return dict(costs=costs, diagnostic_best_slots=best,
                public_rule_slot=public_rule_slot,
                public_rule_matches_best=public_rule_slot in best,
                marginal_priority_gap=maximum - costs[public_rule_slot],
                scope='single_drop_priority_only_not_capacity_one_closed_loop_value')


def enumerate_answer_ambiguity(with_replacement):
    """Exact finite-state enumeration, not simulation or independent test episodes.

Assume all other three colors were read correctly and the public generator's
palette/uniqueness constraint is known. Count possible values of the lost color.
"""
    states = list(product(range(4), repeat=4) if with_replacement else permutations(range(4)))
    histogram = {}
    for missing in range(4):
        possibilities = {}
        for state in states:
            visible = tuple(value for i, value in enumerate(state) if i != missing)
            possibilities.setdefault(visible, set()).add(state[missing])
        for state in states:
            visible = tuple(value for i, value in enumerate(state) if i != missing)
            count = len(possibilities[visible])
            histogram[count] = histogram.get(count, 0) + 1
    return dict(color_assignments=len(states), assignment_deletion_cases=len(states)*4,
                possible_missing_values_histogram=histogram,
                fully_inferable_cases=histogram.get(1, 0),
                assumptions='known palette/generator; other three colors read correctly',
                scope='exact_symbolic_enumeration_not_robot_rollouts')


def scaleup_decision(*, rule_matches, groups, rgb_correct, rgb_total,
                     structured_closed_loop_perfect, rule_closed_loop_perfect):
    if min(groups, rgb_total) <= 0:
        raise ValueError('Missing evidence is not a negative result')
    stop = (rule_matches == groups and rgb_correct == rgb_total
            and structured_closed_loop_perfect and rule_closed_loop_perfect)
    return ('no_go_current_simple_color_task_scaleup' if stop
            else 'inconclusive_requires_specific_failure_analysis')
