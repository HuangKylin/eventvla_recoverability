"""Guard against causal leakage and fabricated sufficiency labels."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('claim_audit', ROOT/'scripts/audit_evidence_claim.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EvidenceClaimAuditTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'configs/evidence_claim_audit_v1.json').read_text(encoding='utf-8'))
        self.source = json.loads((ROOT/self.config['source_report']).read_text(encoding='utf-8'))
        self.protocol = json.loads((ROOT/self.config['source_protocol']).read_text(encoding='utf-8'))

    def test_missing_box_is_not_mislabeled_as_stale_occupancy(self):
        result = module.audit(self.source, self.protocol, self.config)
        recent = result['occupancy_summary']['recent']
        self.assertEqual(recent['only_empty_box_evidence'], 0)
        self.assertEqual(result['posthoc_query_only_shortcut']['correct'], 9)
        self.assertFalse(result['stale_memory_caused_control_failure_established'])

    def test_future_frame_is_rejected(self):
        self.source['records'][0]['selected_frames'][0] = self.source['records'][0]['decision_frame']
        with self.assertRaisesRegex(ValueError, 'future retrieval'):
            module.audit(self.source, self.protocol, self.config)

    def test_duplicate_record_does_not_inflate_sample_count(self):
        self.source['records'].append(copy.deepcopy(self.source['records'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate method/probe'):
            module.audit(self.source, self.protocol, self.config)

    def test_unlabeled_box_is_not_implicitly_treated_as_empty(self):
        self.config['episodes']['2']['empty_box_frames'].remove(16)
        with self.assertRaisesRegex(ValueError, 'requires visual adjudication'):
            module.audit(self.source, self.protocol, self.config)


if __name__ == '__main__':
    unittest.main()
