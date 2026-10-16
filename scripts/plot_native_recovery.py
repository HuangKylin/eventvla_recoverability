"""Render saved RGB evidence; no synthetic images or outcome interpolation."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
state=json.loads((ROOT/'reports/latest_native_recovery_gate.json').read_text())
if state['status']!='complete':raise ValueError('Both gates must finish before plotting')
run=ROOT/'runs'/Path(state['run_dir']).name
fig,axes=plt.subplots(2,5,figsize=(16,7.5))
for row,entry in enumerate(sorted(state['episodes'],key=lambda e:e['seed'])):
    folder=run/str(entry['seed']);result=json.loads((folder/'result.json').read_text())
    for col in range(5):
        file=folder/(f'visible_{col}_head_camera.png' if col<4 else 'returned_-1_head_camera.png')
        ax=axes[row,col];ax.imshow(Image.open(file));ax.set_xticks([]);ax.set_yticks([])
        title=f"Slot {col+1}: {result['recovery']['records'][col]['read']['answer'] or 'unrecovered'}" if col<4 else 'Returned: all colours hidden'
        ax.set_title(title,fontsize=10)
        if col==0:ax.set_ylabel(f"Seed {entry['seed']}",fontsize=11)
        for spine in ax.spines.values():spine.set_visible(False)
fig.suptitle('Shared physical recovery from native EventVLA parents',fontsize=15)
fig.text(.5,.025,'Same immutable skill library; fixed scan order; RGB + proprioception only. Two diagnostic scenes, not a success-rate estimate.',ha='center',fontsize=10)
fig.subplots_adjust(left=.04,right=.995,top=.86,bottom=.10,wspace=.05,hspace=.42)
out=ROOT/'reports/native_recovery_images';out.mkdir(exist_ok=True)
fig.savefig(out/'shared_recovery.png',dpi=160)
fig.savefig(out/'shared_recovery.pdf')
print(out/'shared_recovery.png')

# Optional closed-loop panel once all six policy branches have completed.
status_path=ROOT/'reports/latest_native_recovery_vla.json'
if status_path.exists():
    policy=json.loads(status_path.read_text())
    if policy['status']=='complete':
        trial=ROOT/'runs'/Path(policy['run_dir']).name
        fig,axes=plt.subplots(2,3,figsize=(12,9.0))
        for row,seed in enumerate(sorted(policy['seeds'])):
            for col,condition in enumerate(('keep','recover','recover_discard')):
                folder=trial/f'{seed}_{condition}';result=json.loads((folder/'result.json').read_text())
                ax=axes[row,col];ax.imshow(Image.open(folder/'final_head_camera.png'));ax.set_xticks([]);ax.set_yticks([])
                label={'keep':'Keep original','recover':'Recover + retain','recover_discard':'Recover + discard'}[condition]
                outcome=('correct reveal' if result['first_reveal_correct'] else 'no lift within horizon' if not result['lift_events'] else 'wrong first lift' if not result['first_lift_correct'] else 'correct lift; no reveal')
                ax.set_title(f"{label}: {outcome}\n{result['actions_executed']} policy actions",fontsize=10)
                if col==0:ax.set_ylabel(f'Seed {seed}',fontsize=11)
                for spine in ax.spines.values():spine.set_visible(False)
        fig.suptitle('Frozen EventVLA: evidence recovery and matched discard controls',fontsize=14)
        fig.text(.5,.025,'Two diagnostic parents. Runs stop at first wrong lift or first reveal; these are not full-task success rates.',ha='center',fontsize=9)
        fig.subplots_adjust(left=.04,right=.995,top=.86,bottom=.10,wspace=.05,hspace=.42)
        fig.savefig(out/'recovery_closed_loop.png',dpi=160);fig.savefig(out/'recovery_closed_loop.pdf')
