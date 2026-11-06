# 行人跟随感知 / 预测 / 决策链路汇报

日期：2026-11-04
状态：Isaac Gym + ReplicaCAD 仿真闭环；没有真实机器人实验
机器人模型：沿用仓库里的旧球形机器人模型；舞台 demo 用 holonomic `SimpleBase`，P0 模型用于路径跟随

## 0. 一句话结论

链路已经完整跑通：
**前双目相机 → YOLO 检测 → 深度/框高世界坐标 → 世界坐标目标跟踪 → 监督未来位置预测 → teacher-student / 安全回退 → 舞台跟随控制**。
当前最佳配置在 ReplicaCAD 600 帧 ×5 seeds 上达到 **563/600 可见帧、36 帧遮挡、最长连续遮挡 1.72 s、identity 99.97%、零碰撞**。

## 1. 链路图

```text
前双目 768×1024 RGB @10Hz
        │
        ├─ upright / 自遮挡 mask
        ▼
YOLO 检测（独立 Ultralytics 进程）
        │  bbox + conf + HSV appearance
        ▼
深度/框高 → 世界坐标 + 协方差
        │  FrontPairCamera / range-bearing
        ▼
TargetPerceptionLayer
  ├─ 所有检测建轨迹
  ├─ 不可变外观 anchor
  ├─ 外观 + 运动双门限 Hungarian 关联
  ├─ 歧义拒识 / 框面积门限 / 深度一致性
  ├─ 匀速 Kalman [x,y,vx,vy]
  ├─ measured / predicted / lost 状态机
  └─ 1 s 预测、5 s 记忆、重识别
        │  position / velocity / covariance / history / confidence
        ▼
监督 GRU 预测器
  └─ 最近历史 → 0.5 / 1.0 / 2.0 s 未来位置
        │
        ▼
决策层
  ├─ teacher-student / DAgger（当前最佳）
  │    teacher：真值目标、无漏检（只训练用）
  │    student：测量历史 + 预测器（部署）
  ├─ PPO 直接策略（对照）
  └─ 安全回退：动作背离预测点 / 预测不可用 → 规则
        │
        ▼
舞台控制
  ├─ 观察走廊 y=-4 横移跟随
  ├─ 丢失时沿预测方向继续、速度衰减
  └─ 5 cm 舞台墙占据图 + 冰箱 OBB 间距检查
```

## 2. 各模块现在做了什么

### 2.1 相机与世界坐标

- 前双目 `768×1024`，检测前转 upright。
- 目标位置由深度或检测框高度得到 `base_link` 地面点，再旋转到世界坐标。
- 舞台初始朝向面向目标；当检测框被截断、高度先验把目标算远时，深度比高度先验近就采用深度。
- 修正后首帧目标位置误差约 **0.46 m**（修正前约 15.7 m）。

### 2.2 目标跟踪

- 目标由操作员第一帧点选。
- 所有检测都建轨迹，只有一条是目标。
- 外观 anchor 不可变；交叉/重叠时宁可暂时 coast，也不强行跟错人。
- 1 s 内无检测：状态 `predicted`，继续输出 Kalman 位置；超过后 `lost`。
- 轨迹保留 5 s，目标回来优先恢复同一 track；轨迹被清掉后按外观重识别。
- 输出：`position_xy`、`velocity_xy`、`covariance_xy`、`status`、`track_id`、轨迹历史、速度置信度、预测不确定度。

### 2.3 监督预测器

- GRU 输入最近 12 个测量点（相对位置、速度、时间间隔、绝对位置上下文）。
- 输出 0.5 / 1.0 / 2.0 s 位移。
- held-out 轨迹种子：1 s MAE **0.068 m**；平滑常数速度基线 0.173 m。
- 当前版本使用绝对世界位置作为路线先验；跨地图泛化尚未解决。

### 2.4 决策层

当前最佳是 teacher-student：

- teacher PPO 使用真值目标、无检测 dropout；
- student 只用可部署的测量历史 + 冻结预测器；
- 用 DAgger 在 student 自己的状态上让 teacher 重标注；
- student BC 后用 PPO 微调；
- 丢失阶段：如果 RL 动作方向明显背离监督预测点，或预测器没有可用历史，回退规则。

## 3. 结果

### 3.1 ReplicaCAD 600 帧 ×5 seeds

| 策略 | visible | hidden | 最长遮挡 | identity | 碰撞 | 最小间距 |
|---|---:|---:|---:|---:|---:|---:|
| Teacher-student（当前最佳） | **563.2** | **36.4** | **1.72 s** | **99.97%** | 0 | 0.559 m |
| Teacher-student + 三障碍（600 帧，seed 默认） | **583** | **16** | **0.6 s** | **100%** | 0 | 0.559 m |
| HouseExpo 迷宫式户型（450 帧） | — | — | — | **100%** | 0 | 0.545 m |
| 规则 + 监督预测器 | 437.4 | 162.2 | 11.02 s | 98.53% | 0 | 0.559 m |
| 直接 PPO（3 seeds） | 547 | 53 | 3.6 s | 100% | 0 | 0.559 m |

5 个种子分别：

| seed | visible | hidden | 最长遮挡 | identity |
|---|---:|---:|---:|---:|
| 20261102 | 563 | 37 | 1.8 s | 100% |
| 7 | 571 | 28 | 1.2 s | 99.83% |
| 13 | 563 | 37 | 1.8 s | 100% |
| 21 | 553 | 46 | 2.0 s | 100% |
| 42 | 566 | 34 | 1.8 s | 100% |

### 3.2 固定路线 vs 随机路线（50 episodes ×30 s）

随机路线包括随机矩形尺寸/朝向/中心、1–3 个随机遮挡物、随机目标速度、检测 dropout 0.30、噪声 0.05 m。

| 策略 | 固定 visible / success | 随机 visible / success |
|---|---:|---:|
| 规则 | 0.627 / 0.98 | **0.574 / 0.84** |
| PPO | 0.642 / 1.00 | 0.548 / 0.70 |
| Teacher-student | **0.644 / 1.00** | 0.565 / 0.78 |
| SAC | 0.572 / 0.94 | 0.508 / 0.60 |
| 残差 PPO | 0.629 / 0.98 | 0.575 / 0.84 |
| 随机化 teacher-student | — | **0.591 / 0.90** |

结论：固定路线上 teacher-student 最好；随机路线所有学习策略退化，随机化训练能补回一部分，但还没有解决跨场景泛化。

### 3.3 端到端延迟（ReplicaCAD 600 帧）

| 模块 | mean | p95 |
|---|---:|---:|
| Isaac Gym 渲染 | 7.62 ms | 8.53 ms |
| YOLO 检测进程（含 IPC） | **109.16 ms** | 127.87 ms |
| 目标层 update | 0.88 ms | 1.31 ms |
| 监督预测器 | 0.20 ms | 1.29 ms |
| 策略推理 | 0.15 ms | 0.17 ms |
| 总帧 | **123.12 ms** | 141.74 ms |

约 **8.1 Hz**。瓶颈是检测；跟踪/预测/策略合计不到 1.5 ms。真实机器人端应先优化检测 IPC、输入分辨率、TensorRT/异步检测。

### 3.4 多人 / 相机域偏移（初步）

| 条件 | visible / hidden | identity | measured rate |
|---|---:|---:|---:|
| 2 人，正常图像 | 563 / 37 | 99.97% | 37% |
| 4 人，正常图像 | 331 / 69 | 98.75% | 32% |
| 2 人，gain 0.7 + noise 0.02 | 260 / 24 | 100% | 24.6% |
| 4 人 + 域偏移 | 72 / 412 | 97.7% | **0.6%** |

低亮度/噪声下 YOLO 检测几乎失效；这是真实域迁移最大的风险。

## 4. 录制的视频

目录：`artifacts/perception/report_videos/`
视频均为四联屏：**front-left / front-right / top-down / telemetry**。

| 视频 | 内容 | 指标 |
|---|---|---|
| [teacher_student_600_multiview.mp4](report_videos/teacher_student_600_multiview.mp4) | 当前最佳 teacher-student，600 帧 | 585/599 visible，最长遮挡 0.5 s，identity 100%，零碰撞 |
| [teacher_student_multi_obstacle_600_multiview.mp4](report_videos/teacher_student_multi_obstacle_600_multiview.mp4) | 三障碍 ReplicaCAD，600 帧 | 583/599 visible，最长遮挡 0.6 s，identity 100%，零碰撞 |
| [houseexpo_maze_450_multiview.mp4](report_videos/houseexpo_maze_450_multiview.mp4) | HouseExpo 迷宫式户型，450 帧 | 路径 17.56 m，identity 100%，measured 90%，零碰撞 |
| [rule_predictor_400_multiview.mp4](report_videos/rule_predictor_400_multiview.mp4) | 规则 + 监督预测器基线，400 帧 | 220/399 visible，最长遮挡 15.2 s，identity 99.25% |
| [domain_shift_400_multiview.mp4](report_videos/domain_shift_400_multiview.mp4) | 4 人 + gain 0.7 + noise 0.02，400 帧 | 72/284 visible，measured 1.06%，长遮挡，展示检测失效 |

预览图：

- `report_videos/teacher_student_600_preview.png`
- `report_videos/teacher_student_multi_obstacle_600_preview.png`
- `report_videos/houseexpo_maze_450_preview.png`
- `report_videos/rule_predictor_400_preview.png`
- `report_videos/domain_shift_400_preview.png`

> MP4 被 `.gitignore` 忽略，GitHub 上只保留预览 PNG；本地汇报直接用上面路径。

## 5. 汇报建议顺序

1. 先放 **teacher_student_600_multiview.mp4**、**teacher_student_multi_obstacle_600_multiview.mp4** 和 **houseexpo_maze_450_multiview.mp4**：说明四联屏、单障碍、三障碍和迷宫式户型都能跑。
2. 放 **rule_predictor_400_multiview.mp4**：对比同一任务下规则的长遮挡。
3. 放 **domain_shift_400_multiview.mp4**：主动说明真实域瓶颈在检测，不在跟踪/策略。
4. 讲 5-seed 指标、固定 vs 随机路线、延迟表。
5. 最后讲边界：仿真、无真机、路线先验、YOLO 域偏移。

## 6. 当前边界

- 纯仿真（Isaac Gym + ReplicaCAD），没有真实机器人/真实相机。
- 监督预测器有当前路线先验；route-blind 版本在未见旋转路线 1 s 误差从 7.1 cm 退化到 19.7 cm。
- 三障碍结果目前是默认 seed 的一次 600 帧长跑，不是多种子统计。
- HouseExpo 迷宫式户型使用 P0 模型，442 帧内走完 17.56 m 路径；自建 4×4 窄迷宫因 P0 转弯半径和长时间遮挡未达到可用效果，已保留为失败分析，不作为成功演示。
- 低光/噪声下 YOLO 检测率从 37% 掉到 0.6%；真实部署前必须做检测域适应。
- 真实机器人端到端延迟未知；桌面 GPU 下 123 ms，主要耗在 YOLO。
- 已发表方法没有逐项复现；当前是规则/PPO/SAC/残差/teacher-student 强基线。

## 7. 关键文件

- 感知层：`perception/target_perception.py`
- 相机/世界坐标：`perception/range_bearing.py`
- 监督预测器：`perception/learned_predictor.py`
- stage 闭环 + 多视角录制：`perception/sim/stage_follow.py`
- 2D/随机 benchmark：`perception/research/benchmark_following.py`
- teacher-student：`perception/research/train_teacher_student.py`
- 详细报告：`artifacts/perception/TARGET_PERCEPTION_20261102.md`
