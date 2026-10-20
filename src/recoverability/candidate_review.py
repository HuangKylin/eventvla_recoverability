"""Finite task-logic checks and a public-key RGB archive; no simulator/model API."""
from itertools import combinations
import math


def single_query_cost(retained, inaccessible, count=8, reachable_cost=0.1):
    """Expected loss for one uniform future query, not eight sequential queries."""
    retained, inaccessible = set(retained), set(inaccessible)
    valid = set(range(count))
    if count < 1 or not retained <= valid or not inaccessible <= valid:
        raise ValueError('Invalid public evidence index')
    if not math.isfinite(reachable_cost) or not 0 <= reachable_cost < 1:
        raise ValueError('Inspectable cost must be below unavailable-information cost')
    return sum(0 if i in retained else 1 if i in inaccessible else reachable_cost
               for i in range(count))/count


def screen_side_rule(public_left_visits, proposed_action, count=8):
    left = set(public_left_visits)
    if count != 8 or len(left) != 4 or not left <= set(range(count)):
        raise ValueError('Expected four public visits on each side')
    if proposed_action == 'close_left_screen':
        return left
    if proposed_action == 'close_right_screen':
        return set(range(count))-left
    raise ValueError('Unknown action; do not infer from hidden outcome')


def enumerate_candidate():
    universe = set(range(8))
    subsets = list(combinations(range(8), 4))
    rows = []
    for left in subsets:
        for action in ('close_left_screen', 'close_right_screen'):
            inaccessible = set(left) if action == 'close_left_screen' else universe-set(left)
            rule = screen_side_rule(left, action)
            oracle = min(single_query_cost(s, inaccessible) for s in subsets)
            row = dict(left_public_visits=list(left), action=action,
                       rule_retained=sorted(rule),
                       oracle_expected_loss=oracle,
                       rule_expected_loss=single_query_cost(rule, inaccessible),
                       fifo_expected_loss=single_query_cost((4, 5, 6, 7), inaccessible),
                       archive_information_loss=0.0)
            row['rule_equals_oracle'] = math.isclose(row['rule_expected_loss'], oracle, abs_tol=1e-12)
            rows.append(row)
    return dict(scope='exact_finite_single_query_model_not_robot_rollouts',
                layout_partitions=len(subsets), action_conditions=len(rows),
                retained_subsets_per_condition=len(subsets),
                rule_oracle_equal_conditions=sum(r['rule_equals_oracle'] for r in rows),
                action_changes_retained_set_pairs=len(subsets),
                same_action_has_different_layout_dependent_choices=len(subsets),
                averages={method: sum(r[method] for r in rows)/len(rows) for method in
                          ('oracle_expected_loss', 'rule_expected_loss', 'fifo_expected_loss', 'archive_information_loss')},
                information_boundary='Archive restores requested historical RGB exactly; downstream perception/control remains shared and untested',
                conditions=rows)


class PublicVisitArchive:
    """Reference bytes keyed by a visit identifier known at observation time.

An index is not an object identity/answer. No state truth is used; this only works
for the initial candidate's explicit visit query, not the visual-only revision.
"""
    def __init__(self):
        self.records = {}

    def append(self, visit, rgb_bytes):
        if type(visit) is not int or visit < 0 or visit in self.records:
            raise ValueError('Invalid or duplicate public visit')
        if not isinstance(rgb_bytes, bytes) or not rgb_bytes:
            raise ValueError('Expected immutable nonempty bytes')
        self.records[visit] = memoryview(rgb_bytes).tobytes()

    def read(self, visit):
        # Materialize an independent payload to avoid reporting pointer access
        # alone as the cost of handing a frame to a downstream consumer.
        return memoryview(self.records[visit]).tobytes()

    @property
    def payload_bytes(self):
        return sum(len(value) for value in self.records.values())
