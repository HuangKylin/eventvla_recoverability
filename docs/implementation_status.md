# 实施记录：环境、资产与首轮闭环验证

**2026-10-20 09:43：有界诊断已结束，按预设条件停止当前两个任务的扩量。** 原生 12 条为 11 成功、1 失败；该失败的原样续接和两种固定历史图像替换均在第 532 步失败。配对及实际输入核验通过，但没有控制修复收益；本批次所有进程已退出，不进入正式评分器训练，原目标未完成。见 [最终结果与停止决定](evidence_failure_results_20261020.md)。以下为历史记录。

**2026-10-20：按用户继续指令恢复余下两条固定证据替换分支。** 已完成原样续接通过实际输入与动作复核并继承；新增续跑入口只运行未完成条目，旧暂停和中断记录保留。当前仍在第二步，不扩任务、不训练。见 [本次续跑记录](evidence_failure_resume_20261020.md)。以下为历史记录。

**2026-10-19 16:44：按用户要求暂停，已核验本批次所有进程退出。** 原生 12/12 完成（11 成功、1 失败）；唯一失败的原样续接已精确复现。证据替换分支在前缀回放时被中断，尚无替换结果，第三条未启动。正式 K/R 和训练均未开始；不得自动恢复。见 [完整暂停记录与恢复约束](evidence_failure_pause_20261019.md)。以下为历史记录。

**2026-10-19 13:20 快照：固定原生诊断已完成 1/12，首条成功，第二条运行中。** 首条 29 次请求、388 张输入图像通过真实来源核验；GPU 1 串行执行、短暂停冷却和 CPU 自动审计已运行。尚无控制失败证据，配对分支及正式训练均未开始，仍在第二步。动态进度以 `reports/latest_evidence_native_census.json` 和 `reports/latest_evidence_census_analysis.json` 为准；见 [本轮执行与恢复记录](evidence_native_execution_20261019.md)。以下为历史记录。

**2026-10-19 主张复核：发现现有 9 个公开查询的缺证据标签可被查询类别规则 9/9 命中；普通检索有一次返回旧空盒子状态，但原有夹爪查询在相同两图预算内已补到新状态。尚无新主张准入正式训练。** 只保留“动作所需证据是否仍有效”的待证假说，并固定最多 12 条原生回合及 12 条诊断分支的下一验证范围；该验证未运行，旧批次不恢复。本轮新增实验/训练为 0，原目标未完成，仍在第二步。见 [研究主张复核](research_claim_review_20261019.md)；以下均为历史记录。

**2026-10-19 最新：已完成一个公开任务的实际图文检索筛选。3 条训练演示、824 原始帧、52 张已交付观测、9 个相关决策点；6 个历史可答点中，普通任务文字检索的前两图覆盖 6/6，公开夹爪状态查询未提高覆盖率。另 3 个点的所需证据尚未出现在该历史中。结果不支持正式动作评分器训练；按预设约定结束本分支自动扩量和自动换任务。原目标未完成，仍在第二步。** 见 [公开任务筛选与停止决策](activearena_retrieval_decision_20261019.md)。下方为历史记录。

**2026-10-19 本轮更新：按用户继续指令，开始完整存档/相同读取预算问题的可行性核查。共享存档组件及真实处理器 CPU 审计已完成（6 项定向测试、39 项检查、18 组旧状态输入配置）。旧四事件任务中，动作提议后选两条需 1,664 视觉 token，一次读全四条仅需 1,024；未启动模型前向或训练。原目标未完成，仍在第二步。下一项只检查一个公开长历史任务的真实错例；没有证据则停止该路线。** 见 [本轮实现与决策](archive_readout_progress_20261019.md) 和 [查重边界](archive_readout_related_work_20261019.md)。下方为历史决策。

**当前状态（2026-10-19，候选审查后）：多实例候选的一次设计审查和一次修订审查已结束，未准入 GPU 适配或训练。原版存在几何规则/索引存档反例；视觉查询修订版收益未建立，未被真实视觉实验否定。当前构造路线不再自动扩展。** 已保存 [候选结论](multi_instance_candidate_decision_20261019.md) 和 [范围调整提案](research_scope_proposal_20261019.md)；原第一目标未完成，正式评分器训练未开始。

**当前状态（2026-10-19）：用户授权收敛后继续。本轮 CPU 学习空间审计完成，当前简单颜色构造不再扩量；原 20 父状态和 24 槽位 GPU 批次不续跑。下一优先级是一个候选任务的科学准入检查，尚无新任务通过。正式 EventVLA 评分器训练未开始。** 详见 [研究决策](research_headroom_decision_20261019.md) 与 [当前优先级及退出条件](research_priorities_20261019.md)。下方 2026-10-16 的暂停及“正在扩充”等文字为历史记录。

**当前状态（2026-10-16）：按用户要求暂停，无实验或排队进程。新父状态完成 3/20；其余因重启中断或未启动。固定槽位代码通过 CPU 输入审计，GPU 校准及正式评分器训练未开始。** 详见 [当前进度与暂停断点](native_slot_progress_20261016.md)。

以下为历史阶段记录。

**最新（2026-10-16）：逐条开发先导 18/18 完成；内容对照发现目标证据差异，也发现直接删条的数量混杂。正在扩充新父场景，正式评分器训练未开始。** 详见 [逐条续跑与样本扩展](native_item_resume_20261016.md)。

**最新（2026-10-16）：六条真实 EventVLA 恢复对照及一条失败定位对照完成。保留原记忆正确揭盖 2/2，删除后回看 1/2，回看后删除新证据 0/2；第二场景未有效抬盖。共享控制器取回 4/4、3/4 条证据，部分恢复明确保留失败记录。仍在第二步，正式逐条 K/R 未开始。** 详见 [恢复闭环结果](native_recovery_results_20261016.md)。

最新记忆闭环对照已完成：保留 2/2 首次选对目标、全删除 0/2、错配 0/2，两个错配分支均跟随供体红块位置。六条有效轨迹、14 次模型请求通过审计；一次运行停滞已保留记录并重试完成。目前仍在第二步，下一项是区分回看与提交并验证共享恢复分支，正式 K/R 和正式评分器训练尚未开始。详见 [记忆依赖性结果](native_memory_results_20261015.md)。

最新边界校准已完成：两个种子的父状态从 718/720 步移至 810/808 步；原生续接 2/2 正确揭盖，重复轨迹完全一致。不同位置短指令首次选择 0/2。首个模型输入与 12 组离线删除检查通过；正式 K/R 尚未采集。详见 [安全分支点与对照结果](native_boundary_results_20261015.md)。

更新：2026-10-15。服务器：`SOTA-Pro5000`，用户 `user`。

最新恢复进展：两种子原生历史对照的 10 项记录已完成，原生重复轨迹 2/2 精确一致；完整任务成功 1/2。独立动作回放确认父状态目标盖子已抬高 7.95 cm，导致“相对父状态再抬高 4 cm”指标漏判。当前仍在第二步的正式任务适配和评价校准，正式 EventVLA K/R 数据未采集。详见 [恢复结果与指标审计](native_control_results_20261015.md)。下文早期 2/2 为不同运行配置下的历史冒烟记录，不与本轮合并。

最新追加：物理动作触发的 T3 机制完成 48 个分支和 32 个 K/R 标签，首版 60,321 参数评分器已训练，接口测试 66 项通过。最新实际运行结果与正式训练边界见 `action_stage_progress.md`。上一轮八证据容量先导见 `multicue_progress.md`；下文保留环境及首轮诊断证据。

## 当前完成到哪里

已完成资源准备与首轮冒烟验证，以及部分后续研究接口。原任务 `cover_blocks_hard` 在 seed 100000、100001 上均成功，共 **2/2**，最大奖励均为 1.0。使用官方检查点和原任务成功条件，事件容量保持检查点中的 5 条。

两回合结果只证明链路能运行，不能用于估计正式成功率或宣称复现论文分数。随后完成了状态重建、K/D/R 采集、多证据容量任务及固定工位共享回看；见 `m1_progress.md`、`multicue_progress.md`。机制先导评分器已有权重，但更大规模原任务验收、三类正式配对任务、通用非特权控制器、正式 EventVLA 评分器训练及完整 VLA 方法比较仍未完成；第一阶段研究目标尚未达成。

## 实际资产位置

所有路径均位于远程根目录：

```text
/home/user/workspace/HDD_8T/eventvla_recoverability_phase1_20261014
```

该目录解析到 `/media/user/HDD_8T/eventvla_recoverability_phase1_20261014`。下表路径均相对于根目录；完整绝对路径见 `configs/runtime_paths.json`。

| 内容 | 实际位置 |
|---|---|
| EventVLA 与 RoboTwin-Mem 源码 | `third_party/EventVLA/` |
| cuRobo v0.7.8 源码及已编译扩展 | `third_party/curobo/` |
| 策略环境 | `envs/policy/bin/python` |
| 仿真环境 | `envs/sim/bin/python` |
| 下载/运行管理环境 | `envs/tools/bin/python` |
| Qwen3-VL-4B 主干 | `checkpoints/Qwen3-VL-4B-Instruct/` |
| EventVLA 原始权重 | `checkpoints/EventVLA/RoboTwin-MeM/final_model/pytorch_model.pt` |
| 本机可运行的检查点入口 | `checkpoints/runtime/EventVLA-RoboTwin-MeM/final_model/pytorch_model.pt` |
| 物体资产 | `assets/objects/` |
| ALOHA 机械臂资产 | `assets/embodiments/aloha-agilex/` |
| 研究接口与测试 | `src/recoverability/`、`tests/` |
| 新机制评分器权重 | `checkpoints/scorers/20261015T020018Z_pilot_scorer/` |
| 新动作配对先导标签 | `data/mechanism_pilot/20261015T013736Z_action_pairs/labels.jsonl` |
| 运行脚本与日志 | `scripts/`、`logs/`、`runs/` |
| 原任务结果与视频 | `third_party/EventVLA/RoboTwin-Mem/eval_result/` |

Qwen 主干约 8.89 GB，EventVLA 权重约 9.80 GB。仿真资产共 366 个文件、约 1.41 GB，全部经过大小及上游对象哈希核验。项目资源准备时约占 29 GiB，含环境和缓存。没有下载全部演示训练数据；新增数据位于 `data/diagnostics/` 和 `data/pilots/`，用于校准和先导评估，未生成正式训练集。

逐文件校验见 `reports/download_policy_manifest.json`、`reports/download_backbone_manifest.json`、`reports/download_assets_manifest.json`。两份规划配置由模板生成，修改记录在 `reports/generated_asset_configs.json`，原文件保存在远程 `references/original_asset_configs/`。运行检查点链接到原始权重，配套配置单独保存，没有改写原始权重。

## 验证证据

| 检查 | 结果 | 记录 |
|---|---|---|
| 设备与框架 | RTX PRO 5000 72GB，sm_120，PyTorch 2.7.1+cu128，BF16 运算通过 | `reports/smoke_policy.json` |
| 检查点匹配 | 737 项匹配；缺失、多余、形状不符均为 0 | `reports/smoke_policy.json` |
| 动作输出 | 0/5 张事件历史输入均产生有限的 `[1, 50, 14]` 动作数组 | `reports/smoke_policy.json` |
| 推理显存 | 两次测试的 PyTorch 峰值 allocated 约 9,269 / 9,508 MiB，不含完整仿真进程 | `reports/smoke_policy.json` |
| 仿真 | RGB、关节观察、规划与原任务专家轨迹通过 | `reports/smoke_sim_cover_blocks_hard.json` |
| VLA 闭环 | 两回合成功，退出码 0；两个回合均使用原始 FIFO | `reports/baseline_initial_results.json`、`reports/latest_baseline.json` |
| 研究接口 | 66 项测试通过，含上游接口、删除语义、容量审计、物理触发和运行时真值拒绝 | `reports/research_interface_tests.txt` |
| 多证据及回看 | 30 回合先导；240/240 初始读色、48/48 回看正确；固定工位 | `reports/multicue_validation.json` |
| 重建回放 | 两个场景各三次独立回放，父状态、接触续跑、RGB 误差均为 0 | `reports/latest_rebuild_validation.json` |
| 环境依赖 | policy、sim、tools 三个环境检查通过 | `reports/dependency_checks.json` |

推理冒烟使用真实图像的重复组合测试输入管线，不能替代任务验证。首次推理约 2.54 秒、第二次约 0.195 秒，含冷/热差异；不能据此比较不同记忆容量的速度。

## 本机兼容改动

完整源码改动在 `reports/upstream_compatibility.patch`，没有修改原任务成功条件。

- 使用 PyTorch 2.7.1+cu128 / torchvision 0.22.1 支持 Blackwell；cuRobo 按 sm_120 编译。
- 将 Qwen3 固定的 FlashAttention 改成可配置项，本次使用 SDPA。
- 原 SAPIEN 的 OIDN 和 OptiX 降噪器均报错，本次使用无降噪光线追踪、128 samples/pixel。官方原设置为 OIDN、32 samples/pixel；图像分布和渲染成本存在差异，正式对照必须统一设置。
- 增加可设置的评测回合数，修正 shell 等待循环提前退出的问题；策略服务默认绑定本机回环地址，固定指令采样种子。
- 记录检查点匹配并执行严格加载验证；新记忆选择器通过显式参数启用，默认保留原始 FIFO。

视频处理使用 PyAV/FFmpeg。移除了不参与当前流程、但含旧 cp36 ABI 元数据的 decord 包。未安装完整八卡训练依赖或可选点云 PyTorch3D；本次验证针对图像输入、冻结主干和轻量模块路线。

版本清单为 `reports/policy_installed.txt`、`reports/sim_installed.txt`、`reports/tools_installed.txt`。固定的 EventVLA commit：`a4e02f741f78ab01eb76f6a0bc6685f72f60ed24`；cuRobo commit：`d64c4b005459db10c5dd867d8b30a87d5bda9bdb`。

## 可复用命令

每次使用前确认物理 GPU 1 空闲。进程内通常映射成 cuda:0。

```bash
cd /home/user/workspace/HDD_8T/eventvla_recoverability_phase1_20261014
nvidia-smi

# 原检查点设置：5 条事件记忆、上游 FIFO。
envs/tools/bin/python scripts/run_baseline.py \
  --task cover_blocks_hard --episodes 2 --seed 0 --gpu 1

# 后续容量对照入口；这个组合尚未形成正式结果。
envs/tools/bin/python scripts/run_baseline.py \
  --task cover_blocks_hard --episodes 2 --seed 0 --gpu 1 \
  --history-capacity 4 --memory-policy fifo
# 均匀抽样：将最后的 fifo 改成 uniform。

PYTHONPATH="$PWD/src" envs/sim/bin/python -m pytest -q tests

# 专家诊断，不是 VLA 成功率。
EVENTVLA_RAY_DENOISER=none EVENTVLA_RT_SPP=128 \
  envs/sim/bin/python scripts/smoke_sim.py --expert

envs/policy/bin/python scripts/smoke_policy.py
```

闭环入口会启动本机策略服务并在结束时清理自己启动的进程。新的每次运行配置单独存入 `runs/`，容量同时调整模型与客户端。安装/下载脚本留作复现依据；现有运行资源齐全，推理不需要重新下载。

## 研究接口的边界与下一步

`memory.py` 实现 FIFO、固定随机优先级的流式均匀抽样及外部评分接口；没有训练好的评分器。`branch_protocol.py` 检查 K/R 分支匹配、恢复有效性、删除代价和分组；新增 `collect_cue_probe.py` 在真实仿真中采集分支，并以实际父状态、RGB、RNG 与重复续跑校验这些标记。

原地 `pack/unpack` 的接触续跑未通过容差，已拒绝用于标签。替代的新进程重建＋固定前缀回放通过测试，契约只覆盖固定关节驱动控制器；没有序列化完整 EventVLA 客户端/服务端状态。`revisit_or_erase_cue_probe_v0` 用于采集器校准，`multi_cue_versions_v1` 已验证多证据压力和共享固定工位回看。下一步需解决简单年龄规则即可完成任务的问题，并继续完整 VLA 的锚点/缓存审计、真实动作条件与新布局验证。

实际输入还包含 **4 个时间锚点 × 3 个视角**，并保有近期图像队列和首帧缓存。5 条事件记忆不是总共只读 5 张图片。主研究预算仍拟定为 4 条事件证据，正式评测将读取、存储和延迟分别计量。
