"""Sanity-check honest archive competitors on the single-cue calibration probe.

These are offline probe checks, NOT fixed-budget VLA benchmark comparisons.
The archive is allowed to retain the historical observation even after erasure.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import numpy as np
    from PIL import Image
    from recoverability.cue_reader import read_colour
    p=argparse.ArgumentParser()
    p.add_argument('--branches',type=Path,required=True)
    p.add_argument('--programs',type=Path,required=True)
    args=p.parse_args()
    branches=json.loads(args.branches.read_text())
    programs=json.loads(args.programs.read_text())
    if branches['status']!='complete' or not branches.get('all_accepted'):
        raise RuntimeError('Only completed, accepted diagnostics can be audited')
    truth={record['seed']:record['truth_for_audit'] for record in programs['programs']}
    run=Path(branches['run_dir'])
    rows=[]
    for pair in branches['pairs']:
        image_file=run/f"seed_{pair['seed']}_{pair['command']}_K"/'evidence.png'
        pixels=np.asarray(Image.open(image_file).convert('RGB'))
        # Reader receives only historical RGB; truth is consulted below solely for scoring.
        archive_answer=read_colour(pixels).answer
        semantic_record=json.dumps({'slot':0,'colour':archive_answer},separators=(',',':')).encode()
        structured_answer=json.loads(semantic_record)['colour']
        rows.append(dict(seed=pair['seed'],command=pair['command'],
                         archive_success=archive_answer==truth[pair['seed']],
                         structured_success=structured_answer==truth[pair['seed']],
                         archived_images=1,raw_rgb_bytes=pixels.nbytes,
                         png_bytes=image_file.stat().st_size,semantic_payload_bytes=len(semantic_record)))
    report=dict(scope='offline_single_cue_diagnostic_only',rows=rows,
                archive_successes=sum(r['archive_success'] for r in rows),
                structured_successes=sum(r['structured_success'] for r in rows),cases=len(rows),
                warning='One candidate fits in any positive-capacity FIFO. Forced deletion is a label intervention; '
                        'this probe does not establish a memory-budget bottleneck or a learned-method advantage.',
                storage_note='Payload bytes only; Python objects, indexing and encoder cost are not included.')
    (ROOT/'reports/cue_archive_sanity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
