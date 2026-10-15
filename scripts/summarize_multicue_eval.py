"""Produce a factual pilot summary, including counter-evidence to learned eviction."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--report',type=Path,default=ROOT/'reports/latest_multicue_eval.json')
    args=p.parse_args()
    source=json.loads(args.report.read_text())
    if source['status']!='complete' or not source.get('all_prefixes_matched'):
        raise RuntimeError('Only a completed paired experiment can be summarized')
    episodes=[json.loads(Path(row['result_file']).read_text()) for row in source['episodes']]
    if len({r['library_hash'] for r in episodes}) != 1:
        raise RuntimeError('The shared skill library changed across evaluation scenes')
    recovery_queries=[q for r in episodes for q in r['queries'] if q['source']=='recovery']
    failures=Counter()
    for episode in episodes:
        for query in episode['queries']:
            if not query['correct']:
                failures['missing_evidence_abstention' if query['answer'] is None and query['source']=='missing'
                         else 'perception_or_recovery_error']+=1
    index={(row['method'],row['capacity']):row for row in source['aggregates']}
    fifo4=index['fifo',4]
    aware4=index['action_aware',4]
    oldest4=index['oldest',4]
    paired=[]
    for seed in source['seeds']:
        rows={r['job']['method']:r for r in episodes if r['job']['seed']==seed and r['job']['capacity']==4}
        paired.append(dict(seed=seed,action_rule_minus_fifo=rows['action_aware']['correct_queries']-rows['fifo']['correct_queries'],
                           action_rule_minus_oldest=rows['action_aware']['correct_queries']-rows['oldest']['correct_queries']))
    summary=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status='pilot_complete',
        run_dir=source['run_dir'],episodes=len(episodes),independent_colour_scenes=len(source['seeds']),
        calibration_seed=json.loads(Path(source['library']).read_text())['calibration_seed'],
        cue_observations=sum(len(r['observations']) for r in episodes),
        initial_cue_reads_correct=sum(r['perception_correct'] for r in episodes),
        total_query_decisions=sum(r['query_count'] for r in episodes),
        recovery_executions=sum(r['recovery_calls'] for r in episodes),
        successful_recovery_queries=sum(q['correct'] for q in recovery_queries),
        runtime_forbidden_calls=sum(r['forbidden_calls'] for r in episodes),
        all_prefixes_matched=True,failures=dict(failures),paired_scene_differences=paired,
        aggregate_results=source['aggregates'],
        shared_motor_parameter_bytes=episodes[0]['shared_motor_parameter_bytes'],
        elapsed_seconds=source['elapsed_seconds'],
        conclusions=dict(capacity_pressure_observed=fifo4['correct_queries']<fifo4['queries'],
            fixed_workstation_recovery_verified=bool(recovery_queries) and all(q['correct'] for q in recovery_queries)
                and all(r['forbidden_calls']==0 and r['perception_correct']==8 for r in episodes),
            action_rule_exceeds_oldest=aware4['correct_queries']>oldest4['correct_queries'],
            learned_scorer_trained=False,research_first_goal_achieved=False),
        caveats=['Fixed geometry; seeds change colours only',
                 'Offline expert calibration, then no per-test geometry/planning access',
                 'Public apparatus advance is not a learned robot action',
                 'Final outcomes are colour queries, not end-to-end VLA manipulation success',
                 'Unknown historical values are abstentions; there is no guessing baseline',
                 'Queries within a scene are not independent samples',
                 'Payload bytes exclude Python containers and model/runtime overhead',
                 'Fixed-skill execution uses sparse observation points; wall time is not EventVLA throughput'])
    (ROOT/'reports/multicue_validation.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
