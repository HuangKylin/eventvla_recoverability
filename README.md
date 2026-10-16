# EventVLA：证据可恢复性研究第一阶段

## GitHub 发布范围

本仓库保存第一阶段实现快照：研究代码、配置、测试、实验报告、已整理的机制先导 K/R 标签，以及两份轻量评分器权重。本次发布同步至 2026-10-16 的逐条开发先导、内容对照、新父状态验收及扩展批次断点。轻量评分器为此前机制原型，正式 EventVLA 评分器训练尚未开始。

```bash
git clone https://github.com/HuangKylin/eventvla_recoverability.git
cd eventvla_recoverability
```

仓库不包含原始 EventVLA / Qwen 主干权重、Python 环境、完整仿真资产、上游源码或完整运行目录。克隆后不能直接复现全部实验；环境、依赖版本和资产位置见 [实际路径与版本](configs/runtime_paths.json) 及 [实施记录](docs/implementation_status.md)。脚本中的服务器路径、CUDA 和环境约定需按实际机器配置。部分标签引用的原始图像、共享动作库和完整轨迹保存在上述服务器项目中，需补齐后才能重新训练或运行相应实验。

上游 EventVLA 使用 [InternRobotics/EventVLA](https://github.com/InternRobotics/EventVLA)，固定提交为 `a4e02f741f78ab01eb76f6a0bc6685f72f60ed24`；兼容性改动保存在 [补丁文件](reports/upstream_compatibility.patch)。上游代码、模型和资产分别遵循其原始许可。

`manifest.sha256` 记录本次发布文件的 SHA-256（不含自身及 Git 元数据），在 Linux 仓库根目录可执行 `sha256sum -c manifest.sha256` 校验。历史报告中的绝对路径和测试结果保留原始实验记录；本次上传未重新运行 GPU 实验。

发布快照时间：`2026-10-16T07:59:04.776649+00:00`。新父状态扩展批次状态为 `evaluating`，已落盘 2 条采集/续接记录（两类分别计数）；完整运行资产仍在 Pro5000。此处为推送时快照，后续进度以服务器状态为准。

**最新（2026-10-16）：逐条开发先导 18/18 完成；内容对照发现目标证据差异，也发现直接删条的数量混杂。正在扩充新父场景，正式评分器训练未开始。** 详见 [逐条续跑与样本扩展](docs/native_item_resume_20261016.md)。

**最新（2026-10-16）：六条真实 EventVLA 恢复对照及一条失败定位对照完成。保留原记忆正确揭盖 2/2，删除后回看 1/2，回看后删除新证据 0/2；第二场景未有效抬盖。共享控制器取回 4/4、3/4 条证据，部分恢复明确保留失败记录。仍在第二步，正式逐条 K/R 未开始。** 详见 [恢复闭环结果](docs/native_recovery_results_20261016.md)。

**最新（2026-10-15 19:09）：原生记忆闭环对照完成。保留事件首次选对目标 2/2，全删除 0/2，错配 0/2；错配两次均跟随供体红块位置。第二步的记忆依赖诊断通过，正式 K/R 尚未采集。** 详见 [本轮实施结果](docs/native_memory_results_20261015.md)。

远程项目：`SOTA-Pro5000:/home/user/workspace/HDD_8T/eventvla_recoverability_phase1_20261014`。

**2026-10-15 实施进度：已新增物理动作触发的 T3 任务，完成 48 个独立分支、32 个 K/R 标签和首版 60,321 参数评分器训练；接口测试 66 项通过。** 原任务两个 VLA 回合及上一轮 30 回合容量先导保留。新评分器是固定工位机制原型，正式 EventVLA 评分器训练与第一研究目标尚未完成。

## 文档与实际路径

- [实施记录、资产位置与运行命令](docs/implementation_status.md)
- [完整实施方案](docs/phase1_implementation_plan.md)
- [实施任务清单](docs/implementation_backlog.md)
- [实际路径与版本](configs/runtime_paths.json)
- [实验协议草案](configs/phase1_protocol.json)：尚不是训练入口。
- [严格加载与推理验证](reports/smoke_policy.json)
- [首轮闭环结果](reports/baseline_initial_results.json)
- [状态恢复与分支采集进展](docs/m1_progress.md)
- [最新：T3、真实 VLA 接线与首版评分器](docs/action_stage_progress.md)
- [上一轮：多证据容量与共享回看结果](docs/multicue_progress.md)
- [回看控制器的输入与边界](docs/recovery_controller_contract.md)
- [重建回放验证](reports/latest_rebuild_validation.json)
- [实际分支记录](reports/latest_cue_branches.json)
- [环境依赖检查](reports/dependency_checks.json)
- [原始研究总结](references/research_brief.md)

原研究目标保持为：在明确预算下检验动作改变证据可恢复性的作用，并比较学习方法、强规则和完整存档检索。当前先导结果没有证明学习方法优于强规则或完整存档。

## 运行入口

在远程项目根目录运行，使用前先用 nvidia-smi 核实 GPU 1 空闲。

```bash
envs/tools/bin/python scripts/run_baseline.py --task cover_blocks_hard --episodes 2 --gpu 1
PYTHONPATH="$PWD/src" envs/sim/bin/python -m pytest -q tests
```

评测入口另支持 `--history-capacity 4 --memory-policy fifo` 或 `uniform`，用于后续容量对照。诊断采集入口是 `scripts/compile_cue_probe.py` 和 `scripts/collect_cue_probe.py`；其输出明确放在 `data/diagnostics/`，不能混入训练或正式方法比较。新机制评分器权重位于 `checkpoints/scorers/20261015T020018Z_pilot_scorer/`，尚未接入原 EventVLA 的在线 `scored` 记忆接口。

多证据入口为 `scripts/run_multicue_eval.py`，共享动作库位于远程 `runs/20261014T111504Z_shared_skill_library/`。容量先导数据位于 `data/pilots/`；它使用固定工位的示范动作库，不是 EventVLA 的端到端操作评测。

T3 机制入口为 `scripts/collect_action_pairs.py`、`scripts/train_pilot_scorer.py`、`scripts/run_scored_action_eval.py`；真实 VLA 删除审计入口为 `scripts/run_vla_action_probe.py`。同一 GPU 上的 SAPIEN 仿真进程须串行运行。
