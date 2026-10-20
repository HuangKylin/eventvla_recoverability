"""Check actual branch inputs and require bit-exact unchanged continuation."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from recoverability.evidence_branch_audit import audit_branch
from resume_native_controls import atomic_json

def main():
    import numpy as np
    job=json.loads(Path(sys.argv[1]).read_text())
    source,branch=Path(job['source']),Path(job['output'])
    parent=json.loads((source/'parent.json').read_text())
    original=json.loads((source/'result.json').read_text())
    result=json.loads((branch/'result.json').read_text())
    if result['status']!='complete' or not all(result['parent_match'].values()):
        raise ValueError('Invalid completed parent replay')
    report=audit_branch(source,branch,job['replacements'],parent)
    if not job['replacements']:
        native=np.load(source/'executed_actions.npz',allow_pickle=False)['actions'][parent['step']:]
        paired=np.load(branch/'executed_actions.npz',allow_pickle=False)['actions']
        checks=dict(actions_exact=np.array_equal(native,paired),
            final_physics_exact=result['final_physics_hash']==original['final_physics_hash'],
            final_evaluator_exact=result['final_evaluator_state']==original['final_evaluator_state'],
            outcome_exact=result['native_success']==original['native_success'])
        if not all(checks.values()):raise ValueError('Unchanged continuation is not exact: '+repr(checks))
        report['unchanged_replay']=checks
    atomic_json(branch/'paired_audit.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
