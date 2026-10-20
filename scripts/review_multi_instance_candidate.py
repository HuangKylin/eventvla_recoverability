"""One bounded candidate review: task logic and CPU archive cost, never GPU inference."""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/multi_instance_candidate_review_v1.json')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    from PIL import Image
    from recoverability.candidate_review import PublicVisitArchive, enumerate_candidate
    root = ROOT.resolve()
    manifest = {}

    def source(value):
        path = Path(value)
        path = (root/path if not path.is_absolute() else path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing/out-of-project input: '+str(path))
        manifest[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    protocol = json.loads(source(args.protocol).read_text())
    if protocol['gpu_jobs_max'] != 0 or protocol['new_simulation_rollouts_max'] != 0:
        raise ValueError('This review never starts simulation or GPU work')
    design = protocol['design_v0']
    if (design['evidence_count'], design['persistent_gpu_image_capacity'], design['query_count_per_episode']) != (8, 4, 1):
        raise ValueError('Finite model does not implement the changed candidate')
    if design['normalised_miss_cost'] != {'inaccessible': 1.0, 'inspectable': 0.1}:
        raise ValueError('Finite model costs changed')
    bounds = protocol['archive_cost_probe']
    if (not bounds['record_counts'] or not all(type(n) is int and 1 <= n <= 128 for n in bounds['record_counts'])
            or not 1 <= bounds['timed_repetitions'] <= 200 or not 0 <= bounds['warmups'] <= 10
            or not 1 <= bounds['descriptor_width'] <= 1024):
        raise ValueError('Probe exceeds the bounded CPU review budget')
    for name in ('scripts/review_multi_instance_candidate.py', 'src/recoverability/candidate_review.py',
                 'tests/test_candidate_review.py', 'third_party/EventVLA/RoboTwin-Mem/envs/observe_and_pickup.py'):
        source(name)
    assets = {}
    for category in protocol['asset_categories']:
        files = sorted((root/'assets/objects'/category).glob('model_data*.json'))
        if not files:
            raise ValueError('No instance assets for '+category)
        ids = []
        for path in files:
            source(path)
            json.loads(path.read_text())
            ids.append(int(path.stem.removeprefix('model_data')))
        assets[category] = dict(model_ids=ids, metadata_count=len(ids),
                               visual_distinctness_verified=False)

    prior = json.loads(source('reports/latest_action_pairs.json').read_text())
    if prior['status'] != 'complete':
        raise ValueError('Source imagery collection incomplete')
    image_paths = {}
    for row in prior['labels']:
        image_paths.setdefault((row['family_id'], row['evidence_slot']), source(row['evidence_image']))
    images = []
    for key, path in sorted(image_paths.items()):
        encoded = path.read_bytes()
        with Image.open(BytesIO(encoded)) as image:
            rgb = image.convert('RGB')
            images.append(dict(path=path, size=list(rgb.size), raw=rgb.tobytes(), encoded=encoded))
    if not images or len({len(image['raw']) for image in images}) != 1:
        raise ValueError('Need nonempty same-size real RGB workloads')
    probe = protocol['archive_cost_probe']
    warmups, repeats = probe['warmups'], probe['timed_repetitions']

    def measure(fn):
        for i in range(warmups):
            fn(i)
        elapsed = []
        for i in range(repeats):
            start = time.perf_counter_ns()
            fn(i)
            elapsed.append((time.perf_counter_ns()-start)/1e6)
        return dict(repetitions=repeats, p50_ms=float(np.percentile(elapsed, 50)),
                    p95_ms=float(np.percentile(elapsed, 95)), min_ms=min(elapsed), max_ms=max(elapsed))

    def png_decode(index, disk):
        item = images[index % len(images)]
        raw = item['path'].read_bytes() if disk else item['encoded']
        with Image.open(BytesIO(raw)) as image:
            return image.convert('RGB').tobytes()

    decode_memory = measure(lambda i: png_decode(i, False))
    decode_warm_disk = measure(lambda i: png_decode(i, True))
    for i, item in enumerate(images):
        if png_decode(i, True) != item['raw']:
            raise ValueError('Decode changed source pixels')
    rng = np.random.default_rng(20261019)
    workloads = []
    for count in probe['record_counts']:
        archive = PublicVisitArchive()
        start = time.perf_counter_ns()
        for visit in range(count):
            archive.append(visit, images[visit % len(images)]['raw'])
        build_ms = (time.perf_counter_ns()-start)/1e6
        verified = sum(archive.read(i) == images[i % len(images)]['raw'] for i in range(count))
        if verified != count:
            raise ValueError('Public-key retrieval changed evidence bytes')
        py_bytes = (sys.getsizeof(archive.records)
                    + sum(sys.getsizeof(k)+sys.getsizeof(v) for k, v in archive.records.items()))
        width = probe['descriptor_width']
        descriptors = rng.normal(size=(count, width)).astype(np.float32)
        descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-8)
        queries = rng.normal(size=(repeats+warmups, width)).astype(np.float32)
        queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-8)

        def search(i):
            # No BLAS thread pool or encoder; explicitly a cached-vector kernel.
            similarities = np.einsum('ij,j->i', descriptors, queries[i], optimize=False)
            return int(np.argmax(similarities))

        workloads.append(dict(record_count=count, independently_copied_records=True,
                              unique_source_rgb_frames=min(count, len(images)),
                              raw_payload_bytes=archive.payload_bytes,
                              python_dict_keys_and_bytes_size=py_bytes,
                              build_copy_ms=build_ms, exact_payload_roundtrips=verified,
                              indexed_rgb_copy=measure(lambda i: archive.read(i % count)),
                              cached_descriptor_array_bytes=descriptors.nbytes,
                              cached_descriptor_search_only=measure(search),
                              raw_archive_fits_payload_budgets_MiB={str(b): archive.payload_bytes <= b*1024*1024
                                                                 for b in probe['storage_budget_sensitivity_MiB']}))
        del archive, descriptors, queries
    symbolic = enumerate_candidate()
    # Logic can reject v0, but CPU search timing cannot certify v1 vision quality.
    v0 = ('no_go_rule_saturates_symbolic_priority_and_public_index_archive_preserves_evidence'
          if symbolic['rule_oracle_equal_conditions'] == symbolic['action_conditions']
          else 'inconclusive')
    if v0 == 'inconclusive':
        raise RuntimeError('Unexpected symbolic headroom; inspect before making a design decision')
    when = datetime.now(timezone.utc)
    run = root/'runs'/(when.strftime('%Y%m%dT%H%M%SZ')+'_multi_instance_candidate_review')
    run.mkdir(parents=True, exist_ok=False)
    report = dict(status='complete', created_utc=when.isoformat(), run_dir=str(run),
                  decision='candidate_not_admitted_after_one_design_revision',
                  original_goal_achieved=False,
                  design_v0_decision=v0,
                  revision_v1_decision='not_admitted_unestablished_advantage_over_full_archive_retrieval',
                  revision_v1_empirically_disproved=False,
                  revision_count=1, additional_revisions_scheduled=False,
                  gpu_jobs=0, new_simulation_rollouts=0, training_runs=0,
                  asset_inventory=assets, total_asset_metadata_records=sum(a['metadata_count'] for a in assets.values()),
                  exact_symbolic_model=symbolic,
                  archive_cost_probe=dict(image_width_height=images[0]['size'], bytes_per_rgb=len(images[0]['raw']),
                                          source_rgb_frames=len(images), decode_from_ram=decode_memory,
                                          decode_from_warm_filesystem=decode_warm_disk, workloads=workloads),
                  environment=dict(python=sys.version.split()[0], numpy=np.__version__,
                                   boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                                           if Path('/proc/sys/kernel/random/boot_id').exists() else None),
                  source_sha256=manifest,
                  limitations=[
                      'Finite geometry/access model is an explicit proposed task, not physical simulation or sampled generalization.',
                      'Archive information loss zero means the requested RGB is retrievable, not that perception/manipulation succeeds.',
                      'Cost workloads reuse old color-scene frames; they do not test new object recognition or retrieval accuracy.',
                      'Search uses synthetic cached vectors and excludes vision encoding, transfer, model inference, and index quality.',
                      'Filesystem benchmark uses warm page cache, not cold HDD latency or durable write cost.',
                      'Asset IDs are counted only for inventory; policies and retrieval receive no simulator identity.',
                      'V1 is unestablished rather than empirically refuted; the predetermined review cap ends automatic escalation.',
                  ])
    data = json.dumps(report, ensure_ascii=False, indent=2)
    (run/'report.json').write_text(data, encoding='utf-8')
    (root/'reports/latest_multi_instance_review.json').write_text(data, encoding='utf-8')
    text = render_report(report)
    (run/'report.md').write_text(text, encoding='utf-8')
    (root/'docs/multi_instance_candidate_decision_20261019.md').write_text(text, encoding='utf-8')
    print(json.dumps(dict(status='complete', run_dir=str(run), decision=report['decision'],
                         assets=report['total_asset_metadata_records'],
                         symbolic_conditions=symbolic['action_conditions'],
                         rule_optimal=symbolic['rule_oracle_equal_conditions'],
                         workloads=workloads, warm_file_decode_p95_ms=decode_warm_disk['p95_ms']), indent=2))


def render_report(report):
    symbolic = report['exact_symbolic_model']
    probe = report['archive_cost_probe']
    lines = ['# 多实例动作可恢复性候选：一次设计审查与一次修订', '',
             '结论：当前候选不准入 GPU 适配或评分器训练；结束本轮候选扩展。原版存在明确强规则/存档反例，修订版尚无优于公平存档检索的证据。修订版未被真实视觉实验否定，不将“未建立”写成“不可能”。', '',
             '## 原版：八条实例参考、四条工作记忆、左右挡板', '',
             '布局可观察，两个拟执行动作分别关闭左侧或右侧；后续只查询一个公开观察编号。动作前决定保留内容，遮挡侧在剩余时域内不能重访。损失为不可取得=1、可重访=0.1、已保留=0。', '',
             f'枚举 {symbolic["layout_partitions"]} 种左右分区 × 2 个动作 = {symbolic["action_conditions"]} 个条件，每条件枚举 70 种保留子集。',
             f'- 公开几何规则在 {symbolic["rule_oracle_equal_conditions"]}/{symbolic["action_conditions"]} 个条件达到该模型的最优值。',
             '- 动作改变保留项，同一动作在不同布局下也改变保留项；这两个条件成立仍不代表必须学习淘汰。',
             '- 单查询期望损失：FIFO 0.275、动作规则/枚举上界 0.05、精确索引存档的信息取得损失 0。它们是定义模型内的精确计算，不是机器人成功率。',
             '- 存档取回公开观察编号对应的原图，读取一张图即可；后续识别/控制是否正确仍未知，但拿到的证据可与保留该图时相同。实例变得更难识别，不会单独破坏这个反例。', '',
             '## 原始图像与 CPU 检索成本检查', '',
             f'检查现有五类实例资产，共 {report["total_asset_metadata_records"]} 份模型元数据；未验证它们的视觉可分性。',
             f'成本负载使用已有 {probe["source_rgb_frames"]} 条 RGB，大小 {probe["image_width_height"]}，每帧 {probe["bytes_per_rgb"]} 字节。扩大条目时复制这些旧图，不把重复条目算成新场景。', '',
             '| 存档条目 | 原始 RGB 载荷 MiB | 取回并复制一帧 p95 ms | 1024 维缓存向量搜索 p95 ms |',
             '|---|---:|---:|---:|']
    for row in probe['workloads']:
        lines.append(f'| {row["record_count"]} | {row["raw_payload_bytes"]/1024**2:.3f} | {row["indexed_rgb_copy"]["p95_ms"]:.4f} | {row["cached_descriptor_search_only"]["p95_ms"]:.4f} |')
    lines += ['', f'暖文件缓存下读 PNG 并解码一帧的 p95 为 {probe["decode_from_warm_filesystem"]["p95_ms"]:.4f} ms；不是冷盘结果。以上不含视觉编码、GPU 传输、模型前向和操作执行，不能当成完整检索延迟或正确率。', '',
              '## 唯一修订：延迟给出视觉实例查询，不提供精确观察编号', '',
              '该修订取消直接编号查表，但必须允许紧凑实例特征、缓存特征的全图存档、按查询检索和动作知情检索。CPU 核查说明小规模索引本身不能被预设成昂贵；它没有证明特征能识别这些对象。',
              '尚未建立跨视角检索误差、编码成本和动作条件的独立收益，也没有通过公共执行器验收。此时继续制造更难的物体与场景，会把主要未知量换成视觉检索和操作适配。按一次审查/一次修订的约定，结束该候选自动扩展。', '',
              '## 后续范围建议', '',
              '若继续同一大方向，建议另立可检验假设：允许所有方法保存完整历史，在相同读取 token/推理预算下，比较动作条件的证据选择与回看决策、查询检索及动作知情规则。学习收益应来自读取/决策质量或实测效率，不预设删历史必然优于保存历史。',
              '这是一项研究范围调整建议，尚未替换原目标、未开启新实验。没有证明效率收益，也没有完成原“超过强规则和完整存档”的目标。', '',
              '详细逐条件结果、预算敏感性、输入哈希和测量边界见 `reports/latest_multi_instance_review.json`。']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    main()
