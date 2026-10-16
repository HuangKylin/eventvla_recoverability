# 2026-10-15 晚间暂停与明日恢复断点

用户明确要求暂停，明日继续。实验已停止，自己的 runner、worker、planner 子进程已清理。没有留下后台实验、策略服务或自动重启任务。

## 研究目标与阶段

第一目标仍是研究动作如何改变证据可恢复性，在有限记忆预算下训练轻量评分器，并与 FIFO、动作知情强规则、完整存档检索比较。四步中的第一步真实 EventVLA 链路已跑通，当前仍在第二步的正式任务与评价校准。固定工位机制先导已有评分器，但正式 EventVLA 的逐条 K/R 数据、评分器训练及第四步正式比较尚未开始。

上一轮记忆依赖诊断有效结果保持不变：保留事件 2/2 首次选对目标；全删除 0/2；错配 0/2 且均跟随供体红块位置。详见 `native_memory_results_20261015.md`。这只是两个选定父状态的机制诊断，不是普遍成功率。

## 本轮已保存的实现

- `src/recoverability/native_recovery.py`：显式 recovery / commit 协议。先声明回看模式，再执行共享动作；最终提交后恢复原生计分，不能倒退或清除失败。回看消耗全部计入物理 tick。
- 同文件的关节桥接与四槽固定顺序扫描：仅用机器人自身关节、RGB 和共享动作库，不在测试场景调用专家规划或物体真值。
- `scripts/native_recovery_worker.py`、`scripts/run_native_recovery_gate.py`：从原生前缀重建父状态，验证物理与观察哈希后，再验收共享回看、盖子归位、遮挡与原生任务状态保留。
- `src/recoverability/camera_capture.py`：仅新实验启用逐相机捕获与立即读回，保持捕获顺序、次数和渲染配置；这是待验收的运行绕行方案，并非已证实的底层故障修复。
- `tests/test_native_recovery.py`、`tests/test_camera_capture.py`：与既有 native_memory/native_boundary 相关测试一起，服务器共 27 项通过。
- `scripts/native_recovery_vla_worker.py`、`scripts/run_native_recovery_vla.py`：下一阶段草稿，语法检查通过，未运行。准备比较 keep / recover / recover_discard；两种恢复条件执行相同物理回看，区别仅为是否保留新证据。所有旧证据及待删除新证据均计划在实际模型请求中审计。

这属于修改后的恢复任务协议：回看与最终提交分离，不能当作未经修改的原生 benchmark 分数。尚未开始正式标签采集。

## 两次尝试与真实断点

1. `runs/20261015T121330Z_native_recovery_gate`：原生前缀第 447 步停滞。重复 faulthandler 栈位于左腕相机 `get_picture("Color")`，尚未进入恢复动作。无资源告警，主机可访问。已停止并保留 `interruption.json`、完整日志及资源记录。不能据此断言是服务器此前整机死机的同一原因。
2. `runs/20261015T122451Z_native_recovery_gate`：启用串行读回，日志已越过前次停滞位置，到 seed 100000 的第 556 步；周期性结果最后保存 501 步。此时用户要求暂停。状态明确为 `paused_by_user`，不是控制器失败样本；原退出栈另存 `termination_traceback`。

**关键：尚未到第 810 步父状态验收，尚未执行一次新的共享恢复。串行捕获的父状态图像等价性、恢复成功率和 EventVLA 恢复后的表现均未验证。seed 100001 尚未开始。**

## 明日接续顺序

先确认 GPU 与服务器可用、没有旧进程，再从原生前缀重新启动两个 controller gate。暂停点不具备可恢复的完整仿真检查点，不要从 556 步拼接，也不要重复启动今天已经停止的工作。

```bash
cd /home/user/workspace/HDD_8T/eventvla_recoverability_phase1_20261014
nvidia-smi
envs/tools/bin/python -u scripts/run_native_recovery_gate.py --gpu 1
```

必须先检查 `reports/latest_native_recovery_gate.json` 两个种子均 `gate_passed=true`。如父状态哈希不一致，不能放宽验收；如再次阻塞，先保存栈和资源记录再处理。恢复关节桥接、技能可达性与归位尚需实际验证。

两个 gate 通过后，审阅尚未运行的 VLA 草稿，尤其是恢复时间到模型 timestep 的映射、恢复后近期图像清空、R 与 R_discard 的物理状态及输入锚点配对，再运行：

```bash
envs/tools/bin/python -u scripts/run_native_recovery_vla.py --gpu 1
```

该 runner 会拒绝未通过两个 gate 的启动。预期六个闭环分支仍只是整组事件的恢复接线诊断；不能直接混入逐条正式 K/R 训练数据。正式 K/R 还需逐证据干预、恢复成本配对与动作条件设计。

## 保存位置

远程：`/home/user/workspace/HDD_8T/eventvla_recoverability_phase1_20261014`（实际解析到 `/media/user/HDD_8T/eventvla_recoverability_phase1_20261014`）。本机工作区 `outputs/eventvla_recoverability_phase1_20261014` 保存源码、测试、报告和本次两组日志副本。环境、权重、大型技能/原生父状态资产继续留在服务器。
