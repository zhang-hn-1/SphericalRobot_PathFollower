# 完整汇总：行人跟随系统、公开 benchmark、官方基线与 IROS 差距

日期：2026-11-07
状态：仿真闭环完整；公开数据/复现包完整；学习型 2D crowd 策略为失败记录；尚不具备 IROS 主文水平。

## 0. 一句话结论

当前工作已经达到 **workshop / arXiv 系统+benchmark 报告** 水平；
如果要冲击 **IROS 主文**，还缺三件硬事：真机或完整视觉 crowd、官方基线在同一 2D 任务上的统一复现、以及一个能稳定优于 ORCA/CrowdNav 的学习方法。

## 1. 系统链路

```text
前双目 768×1024 @10Hz
  ↓ upright / self-occlusion
YOLO 检测（独立 Ultralytics 进程）
  ↓ bbox/conf/appearance
深度或框高 → 世界坐标 + 协方差
  ↓ range_bearing
TargetPerceptionLayer
  ├─ 多轨迹 + 不可变外观 anchor
  ├─ Hungarian 关联 / 歧义拒识 / 面积与深度一致性门限
  ├─ 匀速 Kalman [x,y,vx,vy]
  ├─ measured / predicted / lost 状态机
  └─ 1s 预测、5s 记忆、重识别
  ↓ history / velocity confidence / uncertainty
监督 GRU 未来位置预测器
  └─ 最近历史 → 0.5 / 1.0 / 2.0 s 位移
  ↓
决策层
  ├─ Teacher-student / DAgger（当前最佳）
  ├─ 规则 + 安全回退
  └─ 动作背离预测点时回退规则
  ↓
舞台 / 地图 / crowd 控制层
  ├─ ReplicaCAD 观察走廊跟随
  ├─ HouseExpo 迷宫式跟随
  └─ crowd benchmark 行人安全
```

## 2. 核心结果

### 2.1 ReplicaCAD 主 benchmark

| 方法 | frames | visible | hidden | 最长遮挡 | identity | 碰撞 | 最小间距 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Teacher-student / DAgger | 600 ×5 seeds | **563.2** | **36.4** | **1.72 s** | **99.97%** | 0 | 0.559 m |
| 规则 + 监督预测器 | 600 ×5 seeds | 437.4 | 162.2 | 11.02 s | 98.53% | 0 | 0.559 m |
| 直接 PPO | 600 ×3 seeds | 547 | 53 | 3.6 s | 100% | 0 | 0.559 m |

Teacher-student 的 5 个 seed：

| seed | visible | hidden | 最长遮挡 | identity |
|---:|---:|---:|---:|---:|
| 20261102 | 563 | 37 | 1.8 s | 100% |
| 7 | 571 | 28 | 1.2 s | 99.83% |
| 13 | 563 | 37 | 1.8 s | 100% |
| 21 | 553 | 46 | 2.0 s | 100% |
| 42 | 566 | 34 | 1.8 s | 100% |

### 2.2 多障碍 / 迷宫式场景

| 场景 | frames | 关键结果 |
|---|---:|---|
| ReplicaCAD 三障碍 | 600 | 583/599 visible，16 hidden，最长遮挡 0.6 s，identity 100%，零碰撞 |
| HouseExpo 宽走廊迷宫 | 443 | 目标路径 17.56 m，identity 100%，measured 90.1%，零碰撞，final progress 1.0 |
| 自建 4×4 窄迷宫 | 596 | P0 底盘贴墙/目标长时遮挡；作为失败记录保留 |

### 2.3 端到端延迟（ReplicaCAD 600 帧）

| 模块 | mean | p95 |
|---|---:|---:|
| Isaac Gym 渲染 | 7.62 ms | 8.53 ms |
| YOLO 检测进程（含 IPC） | **109.16 ms** | 127.87 ms |
| 目标层 update | 0.88 ms | 1.31 ms |
| 监督预测器 | 0.20 ms | 1.29 ms |
| 策略推理 | 0.15 ms | 0.17 ms |
| 总帧 | **123.12 ms** | 141.74 ms |

约 **8.1 Hz**，瓶颈是 YOLO。

### 2.4 多人 / 相机域偏移

| 条件 | visible / hidden | identity | measured |
|---|---:|---:|---:|
| 2 人正常图像 | 563 / 37 | 99.97% | 37% |
| 4 人正常图像 | 331 / 69 | 98.75% | 32% |
| 2 人，gain 0.7 + noise 0.02 | 260 / 24 | 100% | 24.6% |
| 4 人 + gain/noise | 72 / 412 | 97.7% | **0.6%** |

结论：检测端域偏移是当前最大风险。

### 2.5 1D crowd benchmark：3 动态行人

协议：随机路线、随机遮挡、3 动态行人、dropout 0.30、噪声 0.05 m、100 episodes。

| 方法 | success (95% CI) | 行人碰撞 | mean distance |
|---|---:|---:|---:|
| 规则 | 0.18 [0.104, 0.256] | 82 | 1.93 m |
| Teacher-student 15D | 0.21 [0.130, 0.290] | 79 | 2.28 m |
| **Attention + 0.35 s predictive safety** | **0.71 [0.621, 0.799]** | **17** | 3.07 m |

结论：预测式安全层把行人碰撞从 82 降到 17，success 从 0.18 提到 0.71；代价是平均跟距从 1.93 m 增到 3.07 m。

### 2.6 官方社交导航基线

| 基线 | 环境 | 20 test episodes |
|---|---|---:|
| ORCA（官方 Python-RVO2） | CrowdNav circle_crossing | success 0.40 / collision 0.60 / nav 10.94 s |
| CrowdNav（官方代码，SARL attention） | 同上；短训练 120 RL episodes | success 0.60 / collision 0.40 / nav 10.44 s |
| SARL（RelationalGraphLearning） | 源码已下载 | 未运行：缺 config.py 和明确 license |

**必须注意**：这些是 goal-reaching 社交导航结果，不是 1D target-following 结果，不能直接横比。

### 2.7 统一 2D crowd benchmark 与失败记录

新建 `perception/research/crowd_follow_2d_env.py`，包含 2D 动作、3 动态行人、随机静态遮挡、ORCA exact 基线。

| 方法 | 50 episodes | 行人碰撞 | 静态碰撞 | mean distance |
|---|---:|---:|---:|---:|
| ORCA exact | 0.10 | 335 | 4509 | 1.38 m |
| Attention 2D 首版 | 0.00 | 350 | 0 | 4.98 m |
| ORCA 行为克隆 + 距离特征 | 0.46 | 469 | 0 | 0.52 m |

结论：2D 学习策略失败，问题不在训练轮数，而在观测/动作/课程设计。已作为负结果记录，不冒充有效。

## 3. 公开数据 / replay / 统计

目录：

```text
artifacts/perception/target_perception_20261102/crowd_dynamic/
├── public_replay_25/
│   ├── scenarios.jsonl
│   ├── statistics.json
│   ├── replay_rule.jsonl
│   ├── replay_teacher.jsonl
│   └── replay_attention_safety.jsonl
└── public_replay_100/
    ├── scenarios.jsonl
    ├── statistics.json
    ├── replay_rule.jsonl
    ├── replay_teacher.jsonl
    └── replay_attention_safety.jsonl
```

25 episodes 摘要：

| 方法 | success (95% CI) | 行人碰撞 |
|---|---:|---:|
| 规则 | 0.28 [0.100, 0.460] | 18 |
| Teacher-student | 0.16 [0.013, 0.307] | 21 |
| Attention + safety | 0.72 [0.540, 0.900] | 4 |

100 episodes 摘要：

| 方法 | success (95% CI) | 行人碰撞 |
|---|---:|---:|
| 规则 | 0.18 [0.104, 0.256] | 82 |
| Teacher-student | 0.21 [0.130, 0.290] | 79 |
| Attention + safety | 0.71 [0.621, 0.799] | 17 |

## 4. 代码 / 数据 / 视频 / 报告索引

### 4.1 代码

| 文件 | 作用 |
|---|---|
| `perception/target_perception.py` | 世界坐标目标层 |
| `perception/range_bearing.py` | 双目/深度世界坐标 |
| `perception/learned_predictor.py` | 监督未来位置预测器 |
| `perception/research/train_trajectory_predictor.py` | 训练/验证预测器；`--no-absolute` 是 route-blind 消融 |
| `perception/research/eval_trajectory_predictor.py` | 同路线 / 未见路线评估 |
| `perception/research/train_teacher_student.py` | 特权 teacher + DAgger student |
| `perception/sim/stage_follow.py` | ReplicaCAD 闭环、多障碍、多视角视频 |
| `perception/sim/map_follow.py` | 地图 / 迷宫 / HouseExpo 闭环 |
| `perception/research/crowd_follow_env.py` | 1D 3 行人 crowd benchmark |
| `perception/research/train_crowd_attention.py` | CrowdNav-lite attention + predictive safety |
| `perception/research/export_crowd_benchmark.py` | 公开 scenario/replay/统计导出 |
| `perception/research/crowd_follow_2d_env.py` | 2D crowd benchmark + ORCA exact |
| `perception/research/train_crowd_attention_2d.py` | 2D attention / BC 尝试 |
| `perception/research/run_official_follow_benchmark.py` | 官方 ORCA/CrowdNav 统一 2D benchmark 尝试 |
| `perception/research/official_crowdnav/` | MIT 官方 CrowdNav 源码与 checkpoint |

### 4.2 视频

`artifacts/perception/report_videos/`

| 文件 | 内容 |
|---|---|
| `teacher_student_600_multiview.mp4` | 600 帧 teacher-student 最佳结果 |
| `teacher_student_multi_obstacle_600_multiview.mp4` | 三障碍 ReplicaCAD |
| `houseexpo_maze_450_multiview.mp4` | HouseExpo 迷宫式跟随 |
| `rule_predictor_400_multiview.mp4` | 规则基线 |
| `domain_shift_400_multiview.mp4` | 低光 + 多人失败案例 |

### 4.3 报告

- `artifacts/perception/REPORT_FOR_SENIOR_20261104_ZH.md`
- `artifacts/perception/TARGET_PERCEPTION_20261102.md`
- `artifacts/perception/IROS_READINESS_20261107_ZH.md`
- `artifacts/perception/target_perception_20261102/official_baselines/summary.json`

## 5. 发表现状判断

| 目标 | 现状 | 还缺 |
|---|---|---|
| arXiv | 可以 | 无硬性缺口 |
| ICRA / IROS workshop | 可以 | 把系统 + benchmark + 负结果组织成论文 |
| IROS 主文 | 不够 | 真机 / 完整视觉 crowd / 统一协议下击败官方基线 |
| CVPR 主文 | 不建议 | 缺少新的视觉方法 / 真实视觉 benchmark |
| CVPR workshop | 边缘 | VideoJev-lite 还太轻 |

## 6. 到 IROS 主文的补齐路线

按优先级：

1. **把 2D crowd benchmark 做成统一协议**
   - ORCA / CrowdNav / SARL / 我们的方法使用同一 2D 动作空间；
   - 同一批 scenario/replay；
   - 100+ episodes、多种子、95% CI。
2. **修复 2D 学习策略失败**
   - 观测加入目标轨迹 / 行人轨迹 / 地图结构；
   - 课程学习：无行人 → 低密度 → 高密度；
   - ORCA 行为克隆后再 PPO；
   - 或将 ORCA 作为安全层，学习方法只学残差。
3. **补 SARL 官方复现**
   - 找可运行 config 或根据论文重建；
   - 与 CrowdNav/ORCA 使用同一协议。
4. **加入完整视觉 / 身份 crowd 系统**
   - YOLO + 目标层 + 多行人；
   - 不再使用 2D 抽象观测。
5. **真机或硬件在环**
   - 即使小场景，也比纯仿真强很多。

## 7. 已知失败与诚实边界

- 监督预测器有当前路线先验；route-blind 在未见旋转路线 1 s MAE 从 7.1 cm 退化到 19.7 cm。
- 4 人 + 低光/噪声下 YOLO measured rate 从 37% 掉到 0.6%。
- 自建 4×4 窄迷宫在 P0 上失败，属底盘约束，不是感知失败。
- 2D attention 策略失败，不能作为有效方法。
- ORCA/CrowdNav 官方结果是 goal-reaching 社交导航，不是 target-following，不能直接横比。
- SARL 未运行，不算精确复现。

## 8. 一页汇报口径

> 我们完成了一个完整的行人跟随感知/预测/决策链路，并在 ReplicaCAD、HouseExpo、三障碍和多动态行人场景上验证。Teacher-student 在 600 帧 ×5 seeds 达到 563/600 可见帧、36 帧遮挡、最长 1.72 s、identity 99.97%、零碰撞。我们公开了 100 episode 的多动态行人 scenario/replay 数据和 95% CI 统计。预测式安全层把行人碰撞从 82 降到 17，success 从 0.18 提升到 0.71。我们也运行了官方 ORCA 和 CrowdNav 基线，并明确它们目前是 goal-reaching 社交导航，不能与我们的 target-following benchmark 直接横比。2D 学习策略的失败也作为负结果公开，因为它们说明当前观测/动作设计还不够，不能靠继续训练解决。
