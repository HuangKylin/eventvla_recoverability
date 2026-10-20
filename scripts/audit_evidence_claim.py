"""Reassess existing retrieval evidence without model inference or new data.

This is a retrospective development audit, not a new task evaluation. The
annotation sidecar is evaluator-only; it never supplies a retrieval decision.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(source, protocol, annotation):
    records = source['records']
    groups = defaultdict(list)
    expected_methods = set(protocol['methods'])
    stride = protocol['observation_stride']
    seen = set()
    for record in records:
        ep, t, method = (record[k] for k in ('episode', 'decision_frame', 'method'))
        key = (ep, t, method)
        if key in seen:
            raise ValueError(f'duplicate method/probe: {key}')
        seen.add(key)
        if ep not in protocol['episodes'] or method not in expected_methods:
            raise ValueError(f'unexpected source record: {key}')
        available = set(range(0, t, stride))
        selected = record['selected_frames']
        if len(selected) != 2 or len(set(selected)) != 2 or not set(selected) <= available:
            raise ValueError(f'invalid or future retrieval: {key}')
        if record['history_images'] != len(available):
            raise ValueError(f'history count changed: {key}')
        target = record['evaluator_target']
        visible = set(protocol['visual_annotation']['episodes'][str(ep)][target + '_frames'])
        if record['relevant_past_frames'] != sorted(visible & available):
            raise ValueError(f'source annotation mismatch: {key}')
        if record['historically_answerable'] != bool(visible & available):
            raise ValueError(f'answerability mismatch: {key}')
        groups[(ep, t)].append(record)
    if len(groups) != len(protocol['episodes']) * 3:
        raise ValueError('source probe count changed')
    for key, group in groups.items():
        if {r['method'] for r in group} != expected_methods:
            raise ValueError(f'incomplete methods: {key}')
        if len({(r['evaluator_subtask'], r['evaluator_target'], r['historically_answerable'])
                for r in group}) != 1:
            raise ValueError(f'inconsistent paired labels: {key}')

    # This deliberately weak, retrospective rule reads only the previously
    # recorded public gripper-derived query. It exposes a dataset shortcut.
    public = [r for r in records if r['method'] == 'siglip_public_gripper_query']
    crosstab = defaultdict(Counter)
    correct = 0
    for r in public:
        answerable = r['historically_answerable']
        crosstab[r['query_kind']]['present' if answerable else 'absent'] += 1
        correct += (r['query_kind'] == 'box') == answerable

    occupancy = []
    for r in records:
        if r['evaluator_subtask'] != annotation['predicate_probe_subtask']:
            continue
        ep, t = r['episode'], r['decision_frame']
        labels = annotation['episodes'][str(ep)]
        box = set(protocol['visual_annotation']['episodes'][str(ep)]['box_frames'])
        occupied = set(labels['deposited_can_frames'])
        empty = set(labels['empty_box_frames'])
        if occupied & empty or not (occupied | empty) <= box:
            raise ValueError(f'contradictory visual labels for episode {ep}')
        if not any(i < t for i in occupied):
            raise ValueError('occupancy probe has no positive evidence in history')
        categories = [
            'first_can_deposited' if i in occupied else
            'previously_empty' if i in empty else
            'target_box_not_visible' if i not in box else 'unadjudicated'
            for i in r['selected_frames']
        ]
        if 'unadjudicated' in categories:
            raise ValueError('selected target-box frame requires visual adjudication')
        occupancy.append(dict(
            episode=ep, decision_frame=t, method=r['method'],
            selected_frames=r['selected_frames'], frame_categories=categories,
            box_visible_at_2=any(i in box for i in r['selected_frames']),
            deposited_can_evidence_at_2='first_can_deposited' in categories,
            only_empty_box_evidence=(any(i in box for i in r['selected_frames']) and
                all(c in {'previously_empty', 'target_box_not_visible'} for c in categories)),
        ))
    summary = {}
    for method in protocol['methods']:
        group = [r for r in occupancy if r['method'] == method]
        summary[method] = dict(
            probes=len(group), box_visible_at_2=sum(r['box_visible_at_2'] for r in group),
            deposited_can_evidence_at_2=sum(r['deposited_can_evidence_at_2'] for r in group),
            only_empty_box_evidence=sum(r['only_empty_box_evidence'] for r in group),
        )
    return dict(
        status='complete_posthoc_development_audit',
        reused_episodes=len(protocol['episodes']), reused_correlated_probes=len(groups),
        new_episodes=0, new_model_forwards=0, new_rollouts=0, new_training_runs=0,
        public_query_answerability_crosstab={k:dict(v) for k,v in crosstab.items()},
        posthoc_query_only_shortcut=dict(correct=correct, total=len(public),
            rule='predict history evidence present iff public query_kind is box',
            held_out=False, learned=False),
        occupancy_records=occupancy, occupancy_summary=summary,
        annotation_limit=annotation['annotation_limit'],
        metric_note='Visual evidence for a coarse predicate, not action sufficiency or control success.',
        learned_gate_headroom_established=False,
        stale_memory_caused_control_failure_established=False,
        phase_confound_prevents_sufficiency_training=True,
        decision='no_training_admission; retain one conditional evidence-validity hypothesis only',
    )


def main():
    config_path = ROOT/'configs/evidence_claim_audit_v1.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    source_path, protocol_path = (ROOT/config[k] for k in ('source_report', 'source_protocol'))
    if sha(source_path) != config['source_report_sha256']:
        raise ValueError('source report changed; create a new audit version')
    source = json.loads(source_path.read_text(encoding='utf-8'))
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if source['protocol_sha256'] != sha(protocol_path):
        raise ValueError('source protocol changed')
    report = audit(source, protocol, config)
    report['inputs'] = {p.relative_to(ROOT).as_posix():sha(p) for p in
                        (config_path, source_path, protocol_path, Path(__file__))}
    report['contact_sheet_sha256'] = {
        str(ep):sha(ROOT/f'data/public_pilots/activearena_can_transfer_v1/inspection/episode_{ep}_stride16.png')
        for ep in protocol['episodes']}
    output = ROOT/'reports/evidence_claim_audit_20261019.json'
    output.write_bytes((json.dumps(report, indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'report':str(output), 'shortcut':report['posthoc_query_only_shortcut'],
                      'occupancy':report['occupancy_summary']}))


if __name__ == '__main__':
    main()
