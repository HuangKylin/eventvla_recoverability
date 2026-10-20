import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from recoverability.evidence_branch_audit import audit_first_request


class FirstRequestAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        p=self.root/'observations/000521';p.mkdir(parents=True)
        Image.fromarray(np.full((4,4,3),71,dtype=np.uint8)).save(p/'head_camera.png')
        self.reference={'examples':[{'lang':'original native instruction',
            'image':[np.zeros((2,2,3),dtype=np.uint8)],
            'memory_keyframe_images':[np.full((2,2,3),13,dtype=np.uint8)],
            'memory_keyframe_steps':[180]}], 'paired_seed':520001011}
        self.replacements=[{'slot':0,'source_step':521}]
        self.actual=copy.deepcopy(self.reference)
        self.actual['examples'][0]['memory_keyframe_images'][0][:]=71

    def tearDown(self):self.temp.cleanup()

    def test_declared_pixels_only_allowed(self):
        self.assertTrue(audit_first_request(self.reference,self.actual,self.root,self.replacements))

    def test_changed_anchor_rejected(self):
        self.actual['examples'][0]['image'][0][:]=12
        with self.assertRaises(ValueError):audit_first_request(self.reference,self.actual,self.root,self.replacements)

    def test_undocumented_timestamp_rejected(self):
        self.actual['examples'][0]['memory_keyframe_steps']=[521]
        with self.assertRaises(ValueError):audit_first_request(self.reference,self.actual,self.root,self.replacements)

    def test_changed_seed_rejected(self):
        self.actual['paired_seed']+=1
        with self.assertRaises(ValueError):audit_first_request(self.reference,self.actual,self.root,self.replacements)

    def test_unmodified_replay_allowed(self):
        self.assertTrue(audit_first_request(self.reference,copy.deepcopy(self.reference),self.root,[]))


if __name__=='__main__':unittest.main()
