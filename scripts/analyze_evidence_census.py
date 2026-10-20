"""Summarize completed census episodes; never promote proxy errors to labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from recoverability.evidence_diagnostic import TASKS, SEEDS, inference_seed
from recoverability.evidence_input_audit import audit_native_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    state = json.loads(((args.run/'status.json') if args.run else
                       ROOT/'reports/latest_evidence_native_census.json').read_text())
    if state['status'] != 'complete' and not args.allow_partial:
        raise ValueError('Census incomplete; use --allow-partial for an explicit partial report')
    rows, identities = [], set()
    for record in state['episodes']:
        key = record['task'], record['seed']
        if key in identities or key[0] not in TASKS or key[1] not in SEEDS:
            raise ValueError('Duplicate or out-of-scope episode')
        identities.add(key)
        path = Path(record['result_file'])
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Result outside project')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['result_sha256']:
            raise ValueError('Completed result changed')
        result = json.loads(path.read_text())
        if result['status'] != 'complete' or result['forbidden_calls'] != 0:
            raise ValueError('Invalid completed episode')
        request_count = 0
        image_count = 0
        for expected_index, audit in enumerate(result['request_audits']):
            # Pickles here are trusted locally produced policy requests only.
            payload = pickle.loads((path.parent/'requests'/f'{expected_index:04d}.pkl').read_bytes())
            example = payload['examples'][0]
            verification = audit_native_payload(payload, path.parent/'observations',
                                                result['instruction'], record['seed'], expected_index)
            image_count += verification['images_checked']
            if payload['paired_seed'] != inference_seed(record['seed'], expected_index):
                raise ValueError('Inference RNG pairing changed')
            if example['lang'] != result['instruction']:
                raise ValueError('Native instruction changed')
            if example['timestep'] != audit['step']:
                raise ValueError('Request timestep mismatch')
            if len(example.get('memory_keyframe_image_metas', [])) != len(example.get('memory_keyframe_images', [])):
                raise ValueError('Memory images lack source metadata')
            for field in ('image_metas', 'memory_keyframe_image_metas'):
                for meta in example.get(field, []):
                    if int(meta.get('source_timestep', example['timestep'])) > example['timestep']:
                        raise ValueError('Future image in actual model input')
            if any(k in example for k in ('evaluator_state','actual_first_block_id','actual_first_visit_side_ids')):
                raise ValueError('Evaluator metadata leaked into actual model input')
            request_count += 1
        if result.get('native_success_without_both_returns'):
            category = 'native_success_without_required_move_return_sequence'
        elif result['native_success']:
            category = 'native_success_no_control_failure_to_repair'
        elif result['candidate_saved']:
            category = 'failure_after_candidate_boundary_cause_not_yet_established'
        else:
            category = 'failure_before_eligible_memory_boundary'
        final = result['final_evaluator_state']
        rows.append(dict(task=record['task'], seed=record['seed'], category=category,
            native_success=result['native_success'], candidate_saved=result['candidate_saved'],
            candidate_step=result.get('candidate_step'), actions=result['actions_executed'],
            requests=request_count, images_verified_against_observed_rgb=image_count,
            final_phase=final['phase'], final_press_cnt=final['press_cnt'],
            elapsed_seconds=result['elapsed_seconds'], result_file=str(path)))
    summaries = {}
    for task in TASKS:
        group = [r for r in rows if r['task']==task]
        summaries[task] = dict(completed=len(group), planned=len(SEEDS),
            native_success=sum(r['native_success'] for r in group),
            candidates_saved=sum(r['candidate_saved'] for r in group),
            categories=dict(Counter(r['category'] for r in group)))
    report = dict(status='complete' if state['status']=='complete' else 'partial',
        run_dir=state['run_dir'], completed_native_episodes=len(rows), planned_native_episodes=12,
        summaries=summaries, episodes=rows, audited_policy_requests=sum(r['requests'] for r in rows),
        images_verified_against_observed_rgb=sum(r['images_verified_against_observed_rgb'] for r in rows),
        diagnostic_branches_executed=0, memory_causation_established=False,
        training_eligible=False, original_research_goal_achieved=False,
        note='Seeds are repeated development trials, not twelve independent task families. put_back has fixed geometry.',
        next_action='Inspect first eligible actual control failures; otherwise stop this bounded diagnostic route.')
    output = Path(state['run_dir'])/'analysis.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    (ROOT/'reports/latest_evidence_census_analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'}))


if __name__ == '__main__':
    main()
