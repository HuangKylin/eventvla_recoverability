# 首版机制评分器

这两份权重是固定工位、两个示范动作的机制验证原型，不是正式 EventVLA 记忆模块。

| 文件 | 用途 | SHA256 |
|---|---|---|
| `learned_action.pt` | RGB/公开位置与年龄 + 拟执行电机目标 | `5a85fa181089e0af1c8547d6db01b850e9057536342e336f9a17a0fd156d99ed` |
| `learned_no_action.pt` | 同样结构，去掉动作输入的消融 | `961ca6fb7783f194e8225767c258126333e8fdf087bcb38b64b17af5448f1024` |

每个模型 60,321 参数。训练/验证/测试分别为 2/1/1 个颜色与观察顺序场景，16/8/8 个 K/R 样本；所有场景使用相同几何和两个相同的电机程序。它不能说明跨动作或跨布局泛化。

输入构造与加载函数位于 `src/recoverability/scorer.py`。检查点包含网络权重，以及仅从训练划分估计的标准化均值和尺度。输入不含真值答案、失效标志、动作槽位标签、种子或未来查询。网络输出是未校准的回归评分，可能出现负值；不要将其当作概率。

EventVLA/Qwen 原始权重未更新。该评分器尚未挂接原 EventVLA 在线事件淘汰接口；它在自定义机制任务的动作前压缩环节使用。

训练详情：`reports/latest_pilot_scorer.json`。独立加载复核：`reports/scorer_reload_validation.json`。实际仿真比较：`reports/latest_scored_action_eval.json`。完整边界：`docs/action_stage_progress.md`。
