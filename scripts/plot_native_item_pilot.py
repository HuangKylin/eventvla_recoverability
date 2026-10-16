"""Standalone descriptive figure; no independent-sample or significance claim."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    rows=[json.loads(line) for line in (ROOT/'data/native_item_development_pilot/labels.jsonl').read_text().splitlines()]
    names=[r['parent_group'].removeprefix('development_parent_')+' / item '+str(r['candidate_index']) for r in rows]
    matrix=np.asarray([[r['success'][k] for k in ['K','D','R']] for r in rows],dtype=int)
    fig,(a,b)=plt.subplots(1,2,figsize=(12,6.2),gridspec_kw={'width_ratios':[1,1.4]})
    a.imshow(matrix,cmap=ListedColormap(['#e9a09c','#79beaa']),vmin=0,vmax=1,aspect='auto')
    a.set_xticks(range(3),['K: keep','D: delete','R: recover']);a.set_yticks(range(len(rows)),names)
    for i,row in enumerate(rows):
        for j in range(3):
            label='correct' if matrix[i,j] else 'not correct'
            if j==2 and not row['recovered']:label+='\n(recovery missed)'
            a.text(j,i,label,ha='center',va='center',fontsize=9)
    a.set_title('First target reveal outcome',pad=12)
    costs=[r['physics_delta_R_minus_K'] for r in rows]
    b.barh(range(len(rows)),costs,color='#6687b7');b.invert_yaxis();b.set_yticks(range(len(rows)),names)
    b.axvline(0,color='black',lw=.7);b.set_xlabel('Actual physics ticks: R minus K (including failed attempts)')
    b.set_title('Additional execution cost',pad=12)
    for i,value in enumerate(costs):b.text(value,i,' '+str(value),va='center',fontsize=9)
    b.margins(x=.22)
    fig.suptitle('Development pilot: 8 candidates from 2 parents',fontsize=15)
    fig.text(.5,.015,'One K rollout is shared within each parent. No full-task success, independent 8-scene estimate, or training claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    out=ROOT/'reports/native_item_images';out.mkdir(exist_ok=True)
    for suffix in ['png','pdf']:fig.savefig(out/f'item_pilot.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
