"""Reanalyse existing artifacts on CPU; decide whether current task merits scaling.

No simulator/model imports, new rollouts, training, or historical-result mutation.
Run on Pro5000 so original image/action/branch artifacts are checked directly.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/research_headroom_gate_v1.json')
    args = p.parse_args()
    # This program has no GPU code; hiding devices also prevents accidental use.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    from PIL import Image
    from recoverability.fixed_skills import SharedRecoveryController
    from recoverability.research_gate import (
        enumerate_answer_ambiguity, match_public_action, rank_diagnostic, scaleup_decision,
    )
    root = ROOT.resolve()
    sources = {}

    def checked_path(value):
        path = Path(value)
        if not path.is_absolute():
            path = root/path
        path = path.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or out-of-project artifact: '+str(value))
        sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read_json(value):
        return json.loads(checked_path(value).read_text())

    protocol = read_json(args.protocol)
    collection = read_json(protocol['collection_report'])
    evaluation = read_json(protocol['evaluation_report'])
    if not (collection['status'] == evaluation['status'] == 'complete'
            and collection['all_pairs_matched'] and collection['all_repeated_R_matched']
            and evaluation['all_parent_states_matched']):
        raise ValueError('Incomplete or unmatched original experiment')
    # Fail closed if the code determining those observations has since changed.
    for report in (collection, evaluation):
        for name, expected in report['source_hashes'].items():
            path = checked_path(name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Original source changed; review snapshot explicitly: '+name)
    for name in ('scripts/audit_research_headroom.py', 'src/recoverability/research_gate.py',
                 'tests/test_research_gate.py'):
        checked_path(name)
    library_path = checked_path(protocol['library'])
    controller = SharedRecoveryController(library_path)
    for item in controller.manifest['skills']:
        checked_path(library_path.parent/item['file'])
    templates = {slot: skill.drives[np.linspace(0, len(skill.drives)-1, 16).astype(int)]
                 for slot, skill in controller.skills.items()}

    rows_by_group = defaultdict(list)
    for row in collection['labels']:
        rows_by_group[(row['family_id'], row['action_hash'])].append(row)
    expected_pairs = {(seed, action) for seed in collection['seeds'] for action in collection['actions']}
    observed_pairs = {(r['seed'], r['action_slot']) for r in collection['labels']}
    if (observed_pairs != expected_pairs or len(rows_by_group) != len(expected_pairs)
            or len(collection['labels']) != 4*len(expected_pairs)):
        raise ValueError('Missing or duplicate candidate/action groups')
    groups = []
    distinct_reads = {}
    for (family, action_hash), rows in sorted(rows_by_group.items()):
        row = rows[0]
        keep = read_json(row['keep_result'])
        if keep['status'] != 'complete' or keep['forbidden_calls']:
            raise ValueError('Invalid original keep branch')
        with np.load(checked_path(row['public_action']), allow_pickle=False) as data:
            rule_slot, distances = match_public_action(
                data['sampled_drives'], templates, protocol['rule_tolerance'])
        # action_slot is only used here as an evaluator check after public decoding.
        if rule_slot != row['action_slot']:
            raise ValueError('Public rule differs from historical closed-loop selector')
        for candidate in rows:
            if (candidate['keep_result'] != row['keep_result']
                    or candidate['public_action'] != row['public_action']):
                raise ValueError('Group uses inconsistent artifacts')
            slot = candidate['evidence_slot']
            recover = read_json(candidate['recover_result'])
            for branch in (keep, recover):
                if branch['status'] != 'complete' or branch['forbidden_calls']:
                    raise ValueError('Invalid original branch')
                for key in ('parent_hash', 'action_hash'):
                    if branch[key] != candidate[key]:
                        raise ValueError('Raw branch differs from label: '+key)
            k = next(o for o in keep['outcomes'] if o['slot'] == slot)
            r = next(o for o in recover['outcomes'] if o['slot'] == slot)
            if k != candidate['keep'] or r != candidate['recover']:
                raise ValueError('Raw outcomes differ from labels')
            path = checked_path(candidate['evidence_image'])
            rgb = np.asarray(Image.open(path).convert('RGB'))
            answer = controller.read(slot, rgb).answer
            key = (family, slot)
            item = dict(family=family, slot=slot, answer=answer, expected=k['answer'],
                        keep_success=k['success'], correct=bool(answer is not None and answer == k['answer'] and k['success']),
                        rgb_bytes=rgb.nbytes, image_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if key in distinct_reads and distinct_reads[key] != item:
                raise ValueError('Same parent evidence differs across action branches')
            distinct_reads[key] = item
        diagnostic = rank_diagnostic(rows, rule_slot)
        groups.append(dict(family=family, action_hash=action_hash,
                           public_program_max_error=distances[rule_slot], **diagnostic))

    totals = {}
    raw_closed_loop = []
    for episode in evaluation['episodes']:
        raw = read_json(episode['result_file'])
        if raw['status'] != 'complete' or raw['forbidden_calls']:
            raise ValueError('Invalid closed-loop evaluation')
        for field in ('correct', 'retained_slots', 'recovery_steps', 'image_payload_bytes', 'semantic_payload_bytes'):
            if raw[field] != episode[field]:
                raise ValueError('Evaluation summary differs from raw result: '+field)
        if raw['correct'] != sum(q['success'] for q in raw['queries']) or len(raw['queries']) != episode['queries']:
            raise ValueError('Raw queries disagree with reported success')
        if (raw['job']['seed'] != episode['seed'] or raw['job']['action_slot'] != episode['action_slot']
                or raw['job']['method'] != episode['method']):
            raise ValueError('Evaluation identity mismatch')
        method = episode['method']
        summary = totals.setdefault(method, dict(correct=0, queries=0, rollouts=0, recovery_steps=0,
                                                 max_image_payload_bytes=0, max_semantic_payload_bytes=0))
        for key in ('correct', 'queries', 'recovery_steps'):
            summary[key] += episode[key]
        summary['rollouts'] += 1
        for key in ('image_payload_bytes', 'semantic_payload_bytes'):
            summary['max_'+key] = max(summary['max_'+key], episode[key])
        raw_closed_loop.append(raw)
    for method in ('action_rule', 'structured', 'archive', 'fifo', 'learned_action'):
        if method not in totals or totals[method]['queries'] <= 0:
            raise ValueError('Missing baseline: '+method)
    raw_by_pair = defaultdict(dict)
    for raw in raw_closed_loop:
        key = (raw['job']['seed'], raw['job']['action_slot'])
        if raw['job']['method'] in raw_by_pair[key]:
            raise ValueError('Duplicate evaluation method')
        raw_by_pair[key][raw['job']['method']] = raw
    for pair in raw_by_pair.values():
        if set(pair) != set(totals) or len({r['parent_hash'] for r in pair.values()}) != 1:
            raise ValueError('Unpaired evaluation')
        if pair['learned_action']['retained_slots'] != pair['action_rule']['retained_slots']:
            raise ValueError('Learned/rule decisions differ; inspect before reporting equality')

    native_source = checked_path('third_party/EventVLA/RoboTwin-Mem/envs/cover_blocks_hard.py').read_text()
    independent_source = checked_path('src/recoverability/cue_probe_task.py').read_text()
    # Pin the generator assumptions behind the finite-state enumeration.
    if 'np.random.permutation(list(range(self.num_slots)))' not in native_source:
        raise ValueError('Native color generator changed')
    if 'integers(0, 4, size=4)' not in independent_source:
        raise ValueError('Mechanism color generator changed')
    semantic_sizes = []
    for family in sorted({key[0] for key in distinct_reads}):
        table = {slot: distinct_reads[(family, slot)]['answer'] for slot in range(4)}
        semantic_sizes.append(dict(family=family, payload_bytes=len(json.dumps(table).encode())))
    rule_matches = sum(g['public_rule_matches_best'] for g in groups)
    correct = sum(item['correct'] for item in distinct_reads.values())
    perfect = lambda method: totals[method]['correct'] == totals[method]['queries']
    decision = scaleup_decision(rule_matches=rule_matches, groups=len(groups), rgb_correct=correct,
                               rgb_total=len(distinct_reads), structured_closed_loop_perfect=perfect('structured'),
                               rule_closed_loop_perfect=perfect('action_rule'))
    timestamp = datetime.now(timezone.utc)
    run = root/'runs'/(timestamp.strftime('%Y%m%dT%H%M%SZ')+'_research_headroom_audit')
    run.mkdir(parents=True, exist_ok=False)
    report = dict(status='complete', created_utc=timestamp.isoformat(), run_dir=str(run), decision=decision,
                  fresh_gpu_rollouts=0, formal_training_started=False, new_independent_test_families=0,
                  retrospective_development_audit=True,
                  mechanism_families=len({row['family_id'] for row in collection['labels']}),
                  mechanism_label_rows=len(collection['labels']), action_groups=len(groups),
                  public_action_rule_best_priority_groups=rule_matches, priority_diagnostics=groups,
                  rgb_record_correct=correct, distinct_rgb_observations=len(distinct_reads),
                  rgb_records=list(distinct_reads.values()), semantic_payloads=semantic_sizes,
                  historical_closed_loop=totals,
                  historical_closed_loop_families=len({e['seed'] for e in evaluation['episodes']}),
                  native_unique_color_enumeration=enumerate_answer_ambiguity(False),
                  mechanism_independent_color_enumeration=enumerate_answer_ambiguity(True),
                  source_sha256=sources,
                  limitations=[
                      'Four development families at one geometry; only one family in historical closed-loop comparison.',
                      'Priority diagnostic uses single-deletion labels; no claim about joint-deletion cost or globally optimal recovery cost.',
                      'Public motor matcher was checked offline; it was not rerun in a fresh closed-loop trial.',
                      'Symbolic inference assumes correct remaining colors and known generator; it is not EventVLA ability.',
                      'Four-image archive has more image storage; no equal-storage superiority claim.',
                      'Semantic payload excludes shared reader/skill code and runtime; task-specific representation suffices here.',
                      'No-go applies to scaling this simple-color construction, not the general research idea.',
                  ])
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    (run/'report.json').write_text(encoded, encoding='utf-8')
    (root/'reports/latest_research_headroom.json').write_text(encoded, encoding='utf-8')
    lines = ['# 现有任务的学习空间审计（2026-10-19）', '',
             '结论：`'+decision+'`。这是回溯开发决策，不是新增泛化实验或统计显著性检验。', '',
             f'- 机制先导：{report["mechanism_families"]} 个已有场景族，{len(groups)} 个动作组，{len(collection["labels"])} 条单删除标签。',
             f'- 仅输入公开电机序列的模板规则：{rule_matches}/{len(groups)} 组选择了实测删除代价最大的证据；这只是单删除优先级上界。',
             f'- 用原 RGB 读色器重新提取：{correct}/{len(distinct_reads)} 条不同的场景/槽位证据正确；各场景 JSON 颜色表载荷 {min(s["payload_bytes"] for s in semantic_sizes)}～{max(s["payload_bytes"] for s in semantic_sizes)} 字节。',
             '- 原生不重复颜色设定：24 种排列 × 4 种单删除 = 96 个枚举情况，只要其余三条正确，全部可唯一推断；不是 96 次仿真。',
             '- 独立颜色机制：256 种赋值 × 4 种单删除 = 1024 个枚举情况，每个缺失值有四种可能，已排除上述捷径。', '',
             '## 已有真实闭环结果（未新增运行）', '',
             '| 方法 | 正确查询 | 回看物理步 | 最大图像载荷 | 最大语义载荷 |',
             '|---|---:|---:|---:|---:|']
    for method, total in totals.items():
        lines.append(f'| {method} | {total["correct"]}/{total["queries"]} | {total["recovery_steps"]} | {total["max_image_payload_bytes"]} | {total["max_semantic_payload_bytes"]} |')
    lines += ['', '## 决策边界', '',
              '停止当前简单颜色任务的扩量训练路线；20 父状态批次、24 固定槽位批次不自动续跑。保留环境、接口与历史结果作为调试资产。',
              '本次不支持把“还原真实 VLA 的接口敏感性”继续当成证明学习必要性的主任务。后续先筛选原生任务中的真实信息瓶颈；若强规则和紧凑语义仍足够，终止这一构造，不通过增大网络或更多同类种子补救。', '',
              '上述分析没有证明一般情况下学习无效，没有训练正式评分器，也没有执行新的 GPU 任务。', '',
              '输入文件与 SHA256、逐组结果和限制见同目录 report.json。']
    text = '\n'.join(lines)+'\n'
    (run/'report.md').write_text(text, encoding='utf-8')
    (root/'docs/research_headroom_decision_20261019.md').write_text(text, encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'run_dir', 'decision', 'mechanism_families',
                     'action_groups', 'public_action_rule_best_priority_groups', 'rgb_record_correct',
                     'distinct_rgb_observations', 'fresh_gpu_rollouts')}, indent=2))


if __name__ == '__main__':
    main()
