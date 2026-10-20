import copy
import json
from pathlib import Path
import tempfile
import unittest
from recoverability.evidence_branch_resume import prepare_resume,file_sha


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        self.old=self.root/'runs/old_evidence_failure_branches';self.old.mkdir(parents=True)
        self.source=self.root/'runs/native/failed';self.source.mkdir(parents=True)
        self.manifest={'branches':[{'name':'unchanged','replacements':[]},
            {'name':'witness','replacements':[{'slot':0,'source_step':521}]},
            {'name':'recency','replacements':[{'slot':0,'source_step':529}]}]}
        self.write(self.old/'selection_manifest.json',self.manifest)
        self.branch=self.old/'unchanged';self.branch.mkdir()
        self.write(self.branch/'result.json',{'status':'complete','native_success':False})
        audit={'first_request_exact_except_declared_images':True,
               'unchanged_replay':{k:True for k in ['actions_exact','final_physics_exact','final_evaluator_exact','outcome_exact']}}
        self.write(self.branch/'paired_audit.json',audit)
        self.write(self.branch.with_suffix('.job.json'),{'source':str(self.source),'replacements':[]})
        self.state={'status':'failed','source':str(self.source),'process_cleanup_completed':True,
            'manifest_sha256':file_sha(self.old/'selection_manifest.json'),'source_hashes':{},
            'episodes':[{'name':'unchanged','result_sha256':file_sha(self.branch/'result.json'),'audit':audit}]}
        self.save()

    def write(self,path,value):path.write_text(json.dumps(value),encoding='utf-8')
    def save(self):self.write(self.old/'status.json',self.state)
    def run_resume(self):return prepare_resume(self.root,self.old,self.source,self.manifest)
    def tearDown(self):self.tmp.cleanup()

    def test_only_completed_prefix_inherited(self):
        inherited,ancestors=self.run_resume()
        self.assertEqual([r['name'] for r in inherited],['unchanged'])
        self.assertEqual(Path(inherited[0]['branch_dir']),self.branch)
        self.assertEqual(ancestors,{self.old})

    def test_duplicate_completion_rejected(self):
        self.state['episodes']*=2;self.save()
        with self.assertRaises(ValueError):self.run_resume()

    def test_active_batch_rejected(self):
        self.state['process_cleanup_completed']=False;self.save()
        with self.assertRaises(ValueError):self.run_resume()

    def test_changed_result_rejected(self):
        self.write(self.branch/'result.json',{'status':'complete','native_success':True})
        with self.assertRaises(ValueError):self.run_resume()

    def test_changed_selection_rejected(self):
        self.manifest['branches'][1]['replacements'][0]['source_step']=520
        with self.assertRaises(ValueError):self.run_resume()

    def test_unverified_replay_rejected(self):
        self.state['episodes'][0]['audit']['unchanged_replay']['actions_exact']=False
        self.write(self.branch/'paired_audit.json',self.state['episodes'][0]['audit']);self.save()
        with self.assertRaises(ValueError):self.run_resume()

    def test_changed_job_rejected(self):
        self.write(self.branch.with_suffix('.job.json'),{'source':str(self.source),'replacements':[{'slot':0,'source_step':521}]})
        with self.assertRaises(ValueError):self.run_resume()


if __name__=='__main__':unittest.main()
