"""Make a report figure from completed, validated terminal observations."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_native_memory import artifact


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('validation', type=Path)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    report = json.loads(args.validation.read_text())
    if report['status'] != 'verified': raise ValueError('Need validated results')
    run = artifact(report['run_dir']); seeds = sorted({r['seed'] for r in report['rows']})
    conditions = ['keep', 'drop_all', 'mismatch']
    labels = ['Keep original events', 'Remove all events', 'Swap donor event images']
    fig, axes = plt.subplots(len(seeds), 3, figsize=(12, 7), squeeze=False)
    for i, seed in enumerate(seeds):
        for j, condition in enumerate(conditions):
            row = next(r for r in report['rows'] if r['seed'] == seed and r['condition'] == condition)
            ax = axes[i, j]
            ax.imshow(plt.imread(artifact(row['result_file']).parent/'final_head_camera.png'))
            ax.set_xticks([]); ax.set_yticks([])
            slots = ', '.join(str(s+1) for s in row['first_lift_slots']) or 'none'
            verdict = 'correct' if row['first_lift_correct'] else 'wrong'
            ax.set_title(f'{labels[j]}\nFirst lift: #{slots} ({verdict})', fontsize=11,
                         color='#18703d' if row['first_lift_correct'] else '#a32632')
            if j == 0:
                ax.set_ylabel(f'Seed {seed}\nTrue red: #{row["target_slot"]+1}\nDonor red: #{row["donor_target_slot"]+1}',
                              rotation=0, labelpad=48, fontsize=10, va='center')
    fig.suptitle('Native EventVLA: event-memory interventions', fontsize=16)
    fig.text(.5, .025, 'Covers numbered left to right, 1-4. Terminal frames differ in time: wrong first lifts were stopped early.\n'
             'Two selected parent states; this is a first-choice diagnostic, not a full-task success benchmark.',
             ha='center', fontsize=9, color='#333333')
    fig.subplots_adjust(left=.14, right=.985, bottom=.13, top=.88, hspace=.33, wspace=.10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=160)
    fig.savefig(args.output.with_suffix('.pdf'))
    plt.close(fig)


if __name__ == '__main__': main()
