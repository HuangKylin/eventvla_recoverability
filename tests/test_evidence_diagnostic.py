import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from recoverability.evidence_diagnostic import (
    TASKS, SEEDS, validate_native_jobs, candidate_ready, validate_replacements, inference_seed)


class DiagnosticContractTests(unittest.TestCase):
    def test_scope_rejects_cherry_picked_seed_or_order(self):
        jobs = [(t,s) for t in TASKS for s in SEEDS]
        validate_native_jobs(jobs)
        with self.assertRaises(ValueError):
            validate_native_jobs(jobs[1:]+jobs[:1])

    def test_no_mid_chunk_or_pending_write_parent(self):
        state = dict(phase='move_other_to_center', fail_flag=False)
        self.assertTrue(candidate_ready(TASKS[1], state, True, True))
        self.assertFalse(candidate_ready(TASKS[1], state, False, True))
        self.assertFalse(candidate_ready(TASKS[1], state, True, False))

    def test_premature_success_is_not_a_memory_parent(self):
        state = dict(phase='success', press_cnt=1, first_center_return_achieved=False,
                     second_center_return_achieved=False, both_on_center=False)
        self.assertFalse(candidate_ready(TASKS[0], state, True, True))

    def test_unobserved_current_future_and_duplicate_replacements_rejected(self):
        validate_replacements([dict(slot=0, source_step=5)], 2, 10, {0,5,8})
        for spec in [[dict(slot=0, source_step=i)] for i in (6,10,12)]+[
            [dict(slot=0, source_step=5), dict(slot=0, source_step=8)]]:
            with self.assertRaises(ValueError):
                validate_replacements(spec, 2, 10, {0,5,8})

    def test_pair_seed_uses_request_index_not_simulated_time(self):
        self.assertNotEqual(inference_seed(SEEDS[0], 0), inference_seed(SEEDS[0], 1))
        with self.assertRaises(ValueError):
            inference_seed(SEEDS[0], 1000)


if __name__ == '__main__':
    unittest.main()
