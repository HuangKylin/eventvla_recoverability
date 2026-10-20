"""Failure-oriented tests for the shared archive / proposal / read-cost contract."""
import unittest
from dataclasses import replace
import numpy as np
from recoverability.archive_readout import (ObservedArchive, InputTrace, ReadBudget,
    ReadLedger, Proposal, select_public, validate_proposal)


class ArchiveReadoutTests(unittest.TestCase):
    def setUp(self):
        self.archive = ObservedArchive()
        self.pixels = np.zeros((8, 8, 3), dtype=np.uint8)
        self.keys = tuple(self.archive.append(i, 'main', self.pixels + i) for i in range(4))
        self.trace = InputTrace('proposal', self.archive.fingerprint(3), 3,
                                (self.keys[0], self.keys[3]), 8, 20, 0.01, 'b' * 64)
        self.proposal = Proposal(self.trace, 7, 'a' * 64, 0.5, 'live_anchor_only_forward')

    def test_reads_cannot_mutate_archive_and_selection_does_not_evict(self):
        before = self.archive.fingerprint(3)
        selected = select_public(self.archive, self.keys, 1, 3, 'recent')
        self.archive.read(selected, 3)[0][:] = 99
        self.pixels[:] = 99
        self.assertEqual(before, self.archive.fingerprint(3))
        self.assertEqual(len(self.archive.metadata(3)), 4)
        self.assertEqual(int(self.archive.read([self.keys[3]], 3)[0][0, 0, 0]), 3)

    def test_future_unknown_and_duplicate_candidates_are_rejected(self):
        for keys in ((self.keys[3],), ('unknown',)):
            with self.assertRaises(ValueError): self.archive.read(keys, 2)
        with self.assertRaises(ValueError): select_public(self.archive, self.keys * 2, 2, 3, 'recent')
        self.assertEqual(select_public(self.archive, self.keys, 0, 3, 'uniform'), ())

    def test_proposal_cannot_see_selected_history_or_arrive_after_selection(self):
        for anchors, started in (((self.keys[3],), False), (self.trace.observation_keys, True)):
            with self.assertRaises(ValueError):
                validate_proposal(self.proposal, self.archive, 3, anchors, started, 'b' * 64)
        self.assertEqual(len(validate_proposal(self.proposal, self.archive, 3,
                                               self.trace.observation_keys, False, 'b' * 64)), 64)
        with self.assertRaises(ValueError):
            validate_proposal(self.proposal, self.archive, 3, self.trace.observation_keys, False, 'c' * 64)

    def test_missing_live_forward_and_wrong_time_or_archive_are_rejected(self):
        for proposal in (replace(self.proposal, source='cached_action'),
                         replace(self.proposal, policy_seconds=None),
                         replace(self.proposal, trace=replace(self.trace, archive_fingerprint='foreign'))):
            with self.assertRaises(ValueError):
                validate_proposal(proposal, self.archive, 3, self.trace.observation_keys, False, 'b' * 64)
        with self.assertRaises(ValueError):
            validate_proposal(self.proposal, self.archive, 4, self.trace.observation_keys, False, 'b' * 64)

    def test_duplicate_anchor_reads_and_proposal_call_are_charged_twice(self):
        ledger = ReadLedger(self.archive, 3, ReadBudget(4, 16, 40, 2))
        ledger.charge(self.trace)
        used = ledger.charge(replace(self.trace, stage='final'))
        self.assertEqual(used.image_reads, 4)
        self.assertEqual(used.visual_tokens, 16)
        with self.assertRaises(ValueError): ledger.charge(self.trace)
        self.assertEqual(len(ledger.traces), 2)  # Rejection is atomic.

    def test_each_budget_dimension_can_reject_an_otherwise_valid_request(self):
        for budget in (ReadBudget(1, 8, 20, 1), ReadBudget(2, 7, 20, 1),
                       ReadBudget(2, 8, 19, 1), ReadBudget(2, 8, 20, 0)):
            with self.assertRaises(ValueError): ReadLedger(self.archive, 3, budget).charge(self.trace)
        with self.assertRaises(ValueError): replace(self.trace, preprocessing_seconds=float('nan')).validate(self.archive)


if __name__ == '__main__': unittest.main()
