"""Finish this explicitly launched experiment: collect gate -> train -> rollout."""
import argparse
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True);a=p.parse_args()
    deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        s=json.loads(a.collection.read_text())
        if s['status']=='failed': raise RuntimeError('Collection failed; no training')
        if s['status']=='complete': break
        time.sleep(5)
    else: raise TimeoutError('Collection did not complete')
    subprocess.run([str(ROOT/'envs/sim/bin/python'),'-u',str(ROOT/'scripts/train_pilot_scorer.py'),
                    '--collection',str(a.collection)],cwd=ROOT,check=True,timeout=300)
    report=json.loads((ROOT/'reports/latest_pilot_scorer.json').read_text())
    # SAPIEN camera readback stalled when two independent renderer processes
    # shared this GPU. Run simulator stages sequentially; the policy server may
    # coexist with its single simulator client, as in the baseline.
    subprocess.run([str(ROOT/'envs/tools/bin/python'),'-u',str(ROOT/'scripts/run_vla_action_probe.py'),
                    '--library',str(a.library)],cwd=ROOT,check=True,timeout=5400)
    subprocess.run([str(ROOT/'envs/tools/bin/python'),'-u',str(ROOT/'scripts/run_scored_action_eval.py'),
                    '--training-report',str(Path(report['run_dir'])/'report.json'),
                    '--library',str(a.library)],cwd=ROOT,check=True,timeout=3600)


if __name__=='__main__': main()
