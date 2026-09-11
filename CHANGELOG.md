# 相对于初始版本 `SphericalRobot_LeggedGym-master` 的修改记录

## 1. 范围与基线

- 初始版本：`SphericalRobot_LeggedGym-master`
- 当前版本：`SphericalRobot_LeggedGym-master-new`
- 对比日期：2026-08-11
- 本文只记录会影响程序运行行为的源码修改。
- 深度相机和视觉导航实验已经移除。与相机耦合的 4 个共享文件已恢复为初始版本，视觉导航源码文件已删除。
- 训练 checkpoint、TensorBoard 事件文件、评估 JSON 和 Python 构建缓存属于运行产物，不列入源码修改清单。

## 2. 会影响运行的文件

共有 7 个会影响运行的文件：2 个新增文件和 5 个修改文件。

| 状态 | 文件 | 修改用途 | 对运行的影响 |
|---|---|---|---|
| 新增 | `legged_gym/envs/rotunbot/target_point/rotunbot_target_repro.py` | 实现论文复现版 Rotunbot 点到点任务环境。 | 新增 `rotunbot_target_repro` 任务行为，包括 19 维观测、目标点采样、成功距离课程、奖励、终止条件以及可恢复的课程状态。 |
| 新增 | `legged_gym/envs/rotunbot/target_point/rotunbot_target_repro_config.py` | 配置论文复现任务及其 PPO/DWL 训练参数。 | 设置平面环境、19 维 DWL-CNN 观测、控制限制、目标课程、评估阈值、`DWLOnPolicyRunner` 和 `rotunbot_target_repro` 实验参数。 |
| 修改 | `legged_gym/dwl/actor_critic_dwl.py` | 限制策略探索噪声。 | 增加 `min_noise_std=0.2` 和 `max_noise_std=1.5`，在采样前限制动作标准差，减少动作过大和控制器饱和。 |
| 修改 | `legged_gym/dwl/on_policy_runner_dwl.py` | 改进 DWL checkpoint 的保存与恢复。 | 保存正确的迭代次数和可选的环境状态；从 checkpoint 文件名恢复过期的迭代次数；支持恢复环境状态；允许不同 episode 的信息字段不完全一致。 |
| 修改 | `legged_gym/envs/__init__.py` | 注册新任务。 | 导入并注册 `rotunbot_target_repro`，因此可以通过 `--task rotunbot_target_repro` 选择该任务。 |
| 修改 | `legged_gym/envs/rotunbot/target_point/rotunbot_target_lh.py` | 修正动作速率限制逻辑。 | 修改判断条件，使 `set_a_rate_limit=True` 时真正启用两个 Rotunbot 动作的速率限制。论文复现任务继承 LH 目标环境，因此也会受到影响。 |
| 修改 | `legged_gym/scripts/play.py` | 增加论文复现 checkpoint 自动评估。 | 按每 50 次迭代评估一次，从 300 评估到 15000；使用连续 40 回合评估，并计算成功率、SPL、CLS、平衡性、路径长度等指标，可保存 JSON、CSV 和 NPZ 结果。 |

## 3. 训练和评估行为

### 训练

删除历史 `logs` 文件夹后，仍然可以从零开始训练。`train.py` 会创建环境和训练 runner，TensorBoard 日志目录会在训练开始时自动重新创建。

示例：

```powershell
python legged_gym/scripts/train.py --task rotunbot_target_repro
```

复现配置默认使用 `resume=False`，因此从零训练不需要旧 checkpoint。

### 评估

评估必须依赖训练得到的 checkpoint。删除 `logs` 不会破坏评估代码，但会删除评估所需的模型文件。重新训练并生成包含 `model_300.pt`、`model_350.pt` 等文件的运行目录后，才能使用 `play.py` 进行评估。

如果没有 checkpoint，`play.py` 会等待指定的 checkpoint，而不会真正执行策略评估。

## 4. 不影响运行的差异

以下文件虽然与初始版本内容不同，但不会改变程序行为：

- `legged_gym/envs/base/base_config.py`：只有注释和换行差异。
- `legged_gym/envs/rotunbot/target_point/rotunbot_target_lh_config.py`：只有注释中的示例路径和 checkpoint 编号变化。
- `legged_gym/utils/task_registry.py`：只有文件末尾换行差异；移除视觉 runner 后，其可执行逻辑与初始版本一致。

## 5. 已移除或恢复的相机相关内容

- 恢复为初始版本：`rotunbot_target.py`、`rotunbot_target_config.py`、`play_target.py`、`Ball_Display.py`。
- 已删除：视觉导航相关包和 `test_visual_encoder.py`。
- 当前版本源码中不再保留深度相机或视觉导航引用。
