from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from recoverability.evidence_input_audit import audit_native_payload
from recoverability.evidence_diagnostic import inference_seed


class ActualInputAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'000000').mkdir()
        rgb=np.zeros((8,8,3),dtype=np.uint8)
        Image.fromarray(rgb).save(self.root/'000000/head_camera.png')
        self.payload=dict(paired_seed=inference_seed(520000,0),examples=[dict(
            lang='native instruction',timestep=1,image=[rgb.copy()],
            image_metas=[dict(absolute_index=0,delta=None,view_index=0)],
            memory_keyframe_images=[rgb.copy()],memory_keyframe_count=1,
            memory_keyframe_steps=[0],memory_keyframe_image_metas=[dict(source_timestep=0,view_index=0)])])

    def audit(self):
        return audit_native_payload(self.payload,self.root,'native instruction',520000,0)

    def test_legal_native_anchor_and_event_pixels(self):
        self.assertEqual(self.audit()['images_checked'],2)

    def test_tampered_history_pixels_are_rejected(self):
        self.payload['examples'][0]['memory_keyframe_images'][0][0,0,0]=255
        with self.assertRaisesRegex(ValueError,'pixels differ'):self.audit()

    def test_future_memory_metadata_is_rejected(self):
        self.payload['examples'][0]['memory_keyframe_image_metas'][0]['source_timestep']=2
        with self.assertRaisesRegex(ValueError,'Future'):self.audit()

    def test_unobserved_frame_and_missing_metadata_are_rejected(self):
        original=deepcopy(self.payload)
        self.payload['examples'][0]['memory_keyframe_image_metas'][0]['source_timestep']=1
        with self.assertRaisesRegex(ValueError,'not archived'):self.audit()
        self.payload=original
        self.payload['examples'][0]['image_metas']=[]
        with self.assertRaisesRegex(ValueError,'count mismatch'):self.audit()


if __name__=='__main__':unittest.main()
