# T3 动作配对、真实 EventVLA 接线与首版评分器（2026-10-15）

已实现物理动作触发的 T3 机制任务，并完成首版轻量评分器训练和实际仿真比较。这些是固定工位机制先导，尚不是正式 EventVLA 评分器训练，也未证明论文创新。

## 四步进度

| 步骤 | 当前状态 |
|---|---|
| 1. EventVLA 基线 | 原任务 2/2 冒烟通过；正式基线统计仍未完成。 |
| 2. 三类配对任务 | T1 和 T3 机制先导已运行；T2 恢复成本任务、通用场景与正式 VLA 标签仍待完成。 |
| 3. 学习评分器 | 首版机制评分器已有真实权重及仿真结果；尚未接入 EventVLA 的在线记忆淘汰。 |
| 4. 对照与论文验证 | 完成小规模先导比较，不能外推正式任务或宣称优于强规则/存档。 |

## T3 采集协议与结果

同一个颜色/观察顺序场景内，保持父物理状态、当前 RGB/关节、历史图像与 RNG 相同，只改变拟执行的左侧或右侧机械臂动作。盖子实际抬升超过 4 cm 才触发指示器复位；环境读取物理位姿实施规则，控制器不读取。复位是本研究明确设计的模拟装置机制，不是原 RoboTwin 任务。

共 48 次独立进程运行，形成 32 个 K/R 标签，覆盖 4 个颜色/顺序场景、2 个动作。所有父状态、执行后状态及重复 R 检查通过。K/R 都可调用同一 RGB/关节回看控制器。

标签为 `J(R)-J(K)`，`J=失败指示+0.1*min(回看物理步数/10000,1)`。墙钟时间不进入标签。被该动作复位的证据删除代价约 1.033，其他证据约 0.033；同一证据的重要性随动作翻转。

训练/验证/测试按整个颜色与观察顺序场景分组：310000、310001 / 310002 / 310003。几何完全相同，不能声称跨布局泛化。过去 200000 系列评测场景未用于训练。

## 实际评分器与仿真结果

评分器使用 RGB ROI 统计、公开槽位/年龄以及拟执行电机目标的 16 个采样点。输入不包含动作槽位标签、真值颜色、失效标志、随机种子或未来查询。这是小型 MLP；没有训练 Qwen/EventVLA 主干，也没有提取其视觉隐层特征。

| 方法 | 正确查询 | 回看物理步数 | 最大图像载荷（字节） | 最大颜色表载荷（字节） |
|---|---:|---:|---:|---:|
| learned_action | 8/8 | 20051 | 230400 | 0 |
| learned_no_action | 7/8 | 20066 | 230400 | 0 |
| fifo | 6/8 | 19902 | 230400 | 0 |
| oldest | 7/8 | 20066 | 230400 | 0 |
| action_rule | 8/8 | 20051 | 230400 | 0 |
| archive | 8/8 | 0 | 921600 | 0 |
| structured | 8/8 | 0 | 0 | 54 |

表内结果来自实际仿真运行，不是训练脚本中的离线反事实拼接。每种方法仅 1 个留出颜色/顺序场景 × 2 个动作；不能估计可靠的泛化成功率。共享前缀暂存 4 张图，再于动作前一次性压缩至 1 张；这不是全程容量为 1 的在线记忆实验。存档保留全部图像，颜色表从 RGB 提取全部答案，并如实计入各自载荷。共享动作库与运行时开销另计。

## 真实 EventVLA 验证

执行 5 个真实 VLA 分支（含 1 次重复）、826 个实际 14 维动作；指定盖子在 200 步预算内被抬起 0/5 次。这是盖子操纵验证，不是完整取放任务成功率。

本轮使用“打开最左/最右盖子”的分段指令、允许颜色重复、打乱观察顺序，并人工注入统一示范前缀。这些设置与原检查点的原生整段任务不同。0/5 说明这一适配未过关，不代表原任务性能；各项分布变化的影响尚未分离。

实际发送的 17 个模型请求均通过删除审计，发现删除图像绕过输入的次数为 0。K 分支重复轨迹完全一致：True。

干预清理事件图像、对应三路相机历史、待提交事件及已计算动作缓存；初始帧必须无目标证据。服务端逐请求重置记忆和随机种子。审计验证已知删除帧及记录的清除，不等于证明任意语义信息都不可推断。前缀由统一示范动作库产生；干预后每个执行动作均来自原检查点。

并行运行两个 SAPIEN 仿真进程时两次在相机读取停滞，失败日志与源码快照保留。最终改为仿真串行执行；单个 VLA 仿真客户端与其策略服务共用 GPU 1。

## 下一步与边界

正式研究的下一关是先做原生指令、原生观察顺序的阳性对照，逐项增加任务改动，定位控制退化；必要时统一适配执行器。让 EventVLA 在配对任务上稳定执行并完成共享回看后，再从其真实闭环采集 K/R 标签。还需补恢复成本任务、全程容量约束、多个布局及新的动作。现有动作规则可直接解出复位机制；简单颜色表也保存了全部所需信息。这些结果验证了机制和训练链路，尚未构成可发表的方法优势。

## 资产与复现

- 机制采集：`/media/user/HDD_8T/eventvla_recoverability_phase1_20261014/runs/20261015T013736Z_action_pairs`
- K/R 数据：`/media/user/HDD_8T/eventvla_recoverability_phase1_20261014/data/mechanism_pilot/20261015T013736Z_action_pairs/labels.jsonl`
- 评分器训练：`/media/user/HDD_8T/eventvla_recoverability_phase1_20261014/runs/20261015T020018Z_pilot_scorer`
- 实际评分器评测：`/media/user/HDD_8T/eventvla_recoverability_phase1_20261014/runs/20261015T020607Z_scored_action_eval`
- 真实 VLA 验证：`/media/user/HDD_8T/eventvla_recoverability_phase1_20261014/runs/20261015T020021Z_vla_action_probe`
- learned_action 权重：`/media/user/HDD_8T/eventvla_recoverability_phase1_20261014/checkpoints/scorers/20261015T020018Z_pilot_scorer/learned_action.pt`；参数 60321；SHA256 `5a85fa181089e0af1c8547d6db01b850e9057536342e336f9a17a0fd156d99ed`。
- learned_no_action 权重：`/media/user/HDD_8T/eventvla_recoverability_phase1_20261014/checkpoints/scorers/20261015T020018Z_pilot_scorer/learned_no_action.pt`；参数 60321；SHA256 `961ca6fb7783f194e8225767c258126333e8fdf087bcb38b64b17af5448f1024`。

在远程项目根目录执行；同一 GPU 的仿真入口须串行：

```bash
envs/tools/bin/python scripts/collect_action_pairs.py --library runs/20261014T111504Z_shared_skill_library/library.json
envs/sim/bin/python scripts/train_pilot_scorer.py --collection <采集运行目录>/status.json
envs/tools/bin/python scripts/run_vla_action_probe.py --library runs/20261014T111504Z_shared_skill_library/library.json
envs/tools/bin/python scripts/run_scored_action_eval.py --training-report <训练运行目录>/report.json --library runs/20261014T111504Z_shared_skill_library/library.json
PYTHONPATH="$PWD/src" envs/sim/bin/python -m pytest -q tests
```
