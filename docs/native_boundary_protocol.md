# 原生动作块结束后的分支点与揭盖对照

日期：2026-10-15。此轮是任务适配诊断，不生成正式 K/R 标签或训练评分器。

## 固定范围

- 种子 100000、100001；复用上一轮各自原生 rollout 保存的真实动作前缀与客户端状态。
- 单 GPU 串行；每个种子先准备父状态，再运行 native、native_repeat、position_other。
- 准备阶段最多续接 220 个原生策略动作；每条控制分支最多 300 个策略动作。
- 保持原模型、三视角观察、原生事件记忆、逐请求重置服务端缓存及 `91000+timestep` 随机种子。
- 不清空客户端动作块、不调用专家、不修改物理场景或任务状态；在原动作块自然耗尽处改变语言。

## 父状态

旧父状态在进入 final 阶段时保存，但原任务只要求盖子离开 reveal 区域，不能保证其已放回。
新父状态同时要求：四次检查完成、final_pointer=0、无失败；四个盖子均与初始高度相差小于 1 cm，盖子与对应块水平距离小于 3 cm；双夹爪开度大于 0.8；盖好且松爪条件连续三个策略动作成立；`step == plan_start_step + len(raw_actions)`。

按首次满足条件的自然边界保存，不依后续成功与否选择父状态。没有合格父状态时保留失败准备记录并跳过该种子的续接。

物体姿态、颜色映射及原始 actor segmentation 只用于评价和公开位置指令的构造；模型输入仍限于 RGB、关节观测和语言。
保存三视角父状态图像及可读取的原始 actor ID 可见像素数。几何合格不自动意味着视觉证据完全不可见；分割不可用时明确记录，需人工图像检查。

## 对照与判据

native 与 native_repeat 使用原生整段指令。position_other 保留完全相同的父状态、历史和随机种子，要求 `Finally open the {位置} cover.`，位置固定为 `(原红块槽位+1)%4`，必定不同于原生目标。
该指令是公开位置诊断，不是等价的颜色记忆任务；原任务可能因打开不同盖子判 fail，需与遵循新指令分开报告。

1. 物理抬升：相对初始桌面静置高度超过 4 cm，连续三个物理子步确认。记录同时首次抬升的全部槽位；多槽同时抬升不算唯一正确目标。
2. 物理揭开：沿用原任务 `_is_slot_open_for_progress` 的位置判定，记录首次揭开的全部槽位。
3. 原任务进度：记录 final_pointer 和 fail_flag 变化；不得将单纯抬升自动等同于原任务揭盖完成。
4. 配对：新父物理状态、当前图像、全部客户端时间与事件历史必须与准备阶段一致；新快照使用 pickle protocol 5 保留数组标志。旧 protocol 4 快照只接受已验证的只读标志差异解释。
5. 重复：对比原生重复分支的动作哈希、最终物理哈希及首次模型图像载荷；不同位置条件仅允许语言目标不同。

运行器在首次揭开、原任务失败，或首次抬起错误目标时停止分支；因此错误抬盖分支不评价其后是否可能纠正。这是首次选择与揭盖诊断，不是任意重试后的最终完成率。

本轮没有删除证据、施加记忆容量压力或启动恢复控制，因此无论成功率如何都不是正式记忆价值标签。父状态隐藏目标后仍须审计近期帧、时间锚点等历史通道，排除删除干预旁路，再推进 K/R。

## 入口

```bash
cd /home/user/workspace/HDD_8T/eventvla_recoverability_phase1_20261014
PYTHONPATH="$PWD/src" envs/sim/bin/python -m pytest -q tests/test_native_boundary.py tests/test_native_control_validation.py tests/test_native_snapshot_audit.py
envs/tools/bin/python scripts/run_native_boundary_controls.py --gpu 1 --previous runs/20261015T072322Z_native_controls_resumed/status.json
```

运行前检查 `reports/latest_native_boundary_controls.json` 及现有进程，避免重复启动。运行器使用与前轮相同的资源阈值及文件锁，只清理自己启动的工作进程与模型服务。
