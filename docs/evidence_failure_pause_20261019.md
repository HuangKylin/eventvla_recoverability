# 唯一失败案例：定位、精确回放与用户暂停断点

记录时间：2026-10-19 16:44～16:48，北京时间。用户明确要求“停下，做好记录工作”。已发送 SIGTERM 至经过命令行、进程组、boot ID 核验的本批次 runner；16:44:44 核验 runner、策略服务、仿真 worker、冷却进程均已退出。禁止自动恢复、换任务或启动训练。

## 已完成的事实

原生清单 12/12 完成，`put_back_block_hard` 6/6 成功，`rearrange_blocks_hard` 5/6 成功、1/6 失败。273 次模型请求、3,591 张图像通过来源核验。

`put_back_block_hard` 的六条动作序列哈希完全相同：`f2b224a7d8e57ae350d1e61a6cb8a5fc59e5fe03d2e026ca96229129849034a3`。这六条不能当作六个独立场景。重排任务有六种动作序列；“不同序列”也不等于独立任务族。哈希分组存于 `reports/evidence_failure_checkpoint_20261019.json`。

唯一失败为 `rearrange_blocks_hard / 520001`，按原生任务代码和实际逐步记录定位为：

| 步数 | 实际事件 |
|---|---|
| 520 → 521 | 第二次按钮触发，计数变为 2，进入 `move_other_to_center` |
| 529 → 530 | 按钮释放，`press_flag` 清零；两块积木均在原位 |
| 530 | 自然动作块结束，保存父状态并发出第 12 次模型请求 |
| 532 | 另一块尚未移到中心，按钮计数已变为 3，原生判据触发失败 |

这是动作块衔接附近的再次按键计数。尚未区分模型意图、动作连续性和原生按钮检测机制各自的贡献，不能称为已证明的“记忆导致失败”。分支点只有 1 个事件槽，容量上限为 5，不存在已证实的容量压力。

## 三条预先固定的分支及暂停状态

清单：`configs/evidence_failure_selection_20261019.json`，运行前固定，未根据配对结果更换来源。

| 分支 | 内容 | 暂停时状态 |
|---|---|---|
| `unchanged` | 原样恢复第 530 步父状态并续接 | **完成**。2 个动作后再次在 532 步失败；父物理状态、观察、评估状态一致；续接动作、最终物理状态、最终评估状态均与原生轨迹精确一致。1 次实际请求、13 张图像审计通过。 |
| `second_press_witness` | 用已观察的第 521 步头部图像替换唯一事件槽 | **用户主动中断**于前缀回放，最后落盘回放进度为 326，日志最后打印到 336；未到 530 步，未应用图像替换，未得到配对控制结果。 |
| `public_recency` | 用最近严格历史帧 529 替换事件槽 | **未启动**。 |

runner 原始状态是 `failed`，错误为用户 SIGTERM 导致的 `KeyboardInterrupt: 15`；中断 worker 的最后状态文件仍写着 `running`。这些是停止前的程序记录，不代表进程仍在运行、服务器死机或策略失败。停止语义以 `reports/evidence_failure_user_stop_20261019.json` 及其中的进程退出核验为准，不覆盖原始日志。

目前只有原样对照得到结果，没有任何已完成的证据替换分支。不能据此报告“替换无效”“替换有效”或该研究方向的最终停止结论；正式逐条 K/R 和评分器训练均未开始，仍在第二步。

## 对照边界与验证

本轮代码使用内容替换诊断，固定事件槽、图像数量、原始时间戳 180、置信度、当前输入及首个请求的所有非目标图像字段。替换来源的真实时间 521/529 单独审计。它不是采用真实新时间戳的可部署检索方法；即使后续取得收益，也需额外验证真实元数据与强公开基线。

第 521 帧是一个评估器选择的阶段见证候选，不是遍历所有历史后的最优证据上界。最近帧是简单规则参考，不代表最强公开基线已被穷尽。

本地和服务器各通过 5 项首请求隔离测试，能拒绝未声明的锚点图像、随机种子及时间戳变化。已完成分支的实际像素与历史来源匹配，原样动作和最终状态精确复现。沿用 BF16、原始图像处理、仿真渲染及任务步数限制，单 GPU 串行、CPU 线程数 2、分支间休息 60 秒，启用墙钟冷却及既有资源停止保护。

## 资产位置与以后恢复

服务器项目根目录：`/home/user/workspace/HDD_8T/eventvla_recoverability_phase1_20261014`，真实路径为 `/media/user/HDD_8T/eventvla_recoverability_phase1_20261014`。

- 原生运行：`runs/20261019T050314Z_evidence_native_census/`。
- 唯一失败父状态：上述目录内 `rearrange_blocks_hard_520001/`，保留 `parent.json`、`prefix_actions.npz`、`client_state.pkl`、`rng.pkl`、原始图像与实际请求。
- 本轮分支运行：`runs/20261019T083752Z_evidence_failure_branches/`，保留源代码快照、固定清单、原样结果及审计、中断目录、所有资源和冷却日志、`user_stop.json`。
- 机器断点：`reports/evidence_failure_user_stop_20261019.json`、`reports/evidence_failure_checkpoint_20261019.json`、`reports/latest_evidence_failure_branches.json`。
- 新入口：`scripts/run_evidence_failure_branches.py`；输入审计：`scripts/audit_evidence_failure_branch.py`、`src/recoverability/evidence_branch_audit.py`。
- 最终汇总入口 `scripts/analyze_evidence_failure_branches.py` 已保存但**未执行**，因为它要求三条分支全部完成；未生成完整实验决策。

本地工作目录下 `outputs/eventvla_recoverability_phase1_20261014/` 保存代码、报告及选定结果副本；全量逐帧图像、权重和环境仍保留在服务器，未声称已全量下载。

收到用户新的继续指令后，先检查服务器、GPU 和已保存文件哈希。当前 runner 会拒绝同一父状态重复建立批次，**没有实现断点续跑参数，不要直接重跑旧命令**。如恢复，应新增明确的续跑入口：继承已通过审计的 `unchanged` 结果，保留中断目录，在新目录仅恢复余下两个原定分支，并保持最多 3 个完成分支的范围。中断在前缀回放中，不是可序列化的实时仿真状态，因此该分支需要从已保存动作前缀重新构建；不能从最后一条日志数字直接继续。
