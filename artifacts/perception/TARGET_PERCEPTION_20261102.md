# 行人跟随感知层：目标锁定、干扰拒识与 1 秒预测

日期：2026-11-02（2026-11-03 增加监督预测 + RL 与 route-blind 消融）
状态：仿真中可运行；真实渲染 + YOLO + 闭环均已跑通  
机器人模型：沿用仓库里的旧球形机器人模型；感知层不依赖动量轮

> GitHub 预览：视频文件较大且被 `.gitignore` 忽略；仓库里同时放了
> `preview_*.png` 关键帧，分别在 `live_depth/`、`map_follow/`、`maps/`、
> `houseexpo/` 下。

## 0. 结论

这次补的是一层**目标人物感知**，不是做人脸识别。操作员在第一帧点选目标后，
感知层负责：

1. **只跟这一个目标**，不因旁边的人交叉、遮挡或外观相近而换人；
2. 目标消失 **1 秒左右** 时，用世界坐标下的匀速运动模型继续给出位置，
   状态标为 `predicted`，并给出协方差和未来 1 秒预测轨迹；
3. 超过 1 秒才报 `lost`；目标在记忆窗口内回来时优先恢复同一轨迹，
   轨迹被清掉后还能按外观重新识别；
4. 输出可以直接喂给闭环控制器：短丢失期间控制器继续使用预测位置，
   不需要每帧都看到检测框。

核心结果：

| 场景 | 指标 | 结果 |
|---|---|---|
| 合成交叉（10 seeds） | identity / 跟错人 | **100% / 0%** |
| 合成相似外观交叉（10 seeds） | identity / 跟错人 | **99.88% / 0.12%** |
| 合成 1 秒丢失（10 seeds） | 1.0 s 位置误差 | **3.8 cm** |
| 合成 1 秒丢失中目标转弯（10 seeds） | 1.0 s 位置误差 | **21.5 cm** |
| 真实渲染 + YOLO + 相似外观干扰人（2 episodes） | identity | **100%** |
| 真实渲染 1 秒消失（2 episodes，深度测距） | 消失期间 predicted 比例 | **100%** |
| 真实渲染 1 秒消失（深度测距） | 1.0 s 位置误差 | **0.155 m** |
| 真实渲染 1 秒消失（仅单目高度） | 1.0 s 位置误差 | **0.165 m** |
| 闭环跟随 300 帧（weave） | 正确跟踪覆盖率 / 跟错人 | **89.7% / 0** |
| 闭环跟随 300 帧（crossing） | 正确跟踪覆盖率 / 跟错人 | **84.0% / 0** |
| 复用场景（4 组交叉 + 24 人物 + 前双目） | identity | **98%–100%** |
| 地图场景（3 张 polygon 地图） | identity / 跟错人 | **100% / 0** |
| 地图闭环跟随（3 张图） | success / identity / 碰撞 | **3/3 / 100% / 0** |
| 开源 HouseExpo 室内户型（3 套） | success / identity / 碰撞 | **3/3 / 100% / 0** |
| ReplicaCAD 复杂环路 40 s（基线） | identity / 碰撞 / 最长遮挡 | **100% / 0 / 8.7 s** |
| 感知增强 + 预测性跟随（规则，旧舞台坐标） | 可见帧 / lost / 碰撞 | 325 / 58% / 0 |
| 舞台深度/初始朝向修正 + 规则预测 | 可见帧 / hidden / 最长遮挡 | **343 / 57 / 4.3 s** |
| 监督预测器 + RL + 安全回退（600 帧 3 seeds） | visible / hidden / 最长遮挡 | **547 / 53 / 3.6 s** |
| 消融：route-blind 预测器 + RL（600 帧 3 seeds） | visible / hidden / 最长遮挡 | 500 / 100 / 4.9 s |

这里的“1 秒位置误差”是预测点到目标真实世界坐标的距离。真实渲染中约
0.15 m 的误差里，大部分是**固定的单目/深度定位偏差**（看到的是人体前表面，
目标根节点在身体中心），不是预测漂移；在 1 秒内预测误差几乎不增长。

## 1. 新增文件

| 文件 | 作用 |
|---|---|
| `perception/target_perception.py` | 世界坐标多目标跟踪 + 不可变身份锚 + 歧义拒识 + 1 秒 Kalman 预测 |
| `perception/range_bearing.py` | 前双目几何：检测框 + 深度/身高先验 → `base_link` 地面点、协方差 |
| `perception/eval_target_perception.py` | 合成检测基准：交叉、相似外观、1 s/2 s 丢失、转弯、长丢失重识别 |
| `perception/test_target_perception.py` | 8 个 CPU 单元测试 |
| `perception/sim/occlusion_perception.py` | Isaac Gym + YOLO 的真实渲染消失实验，输出 JSON 和视频 |
| `perception/sim/follow_p0.py` | 新增 `--target-layer` / `--use-depth`，把世界坐标层接进现有闭环跟随 |
| `perception/research/predictive_follow_env.py` | 15D 观测 / 1D 横移动作的快速 2D 预测跟随环境（目标环路 + 冰箱遮挡） |
| `perception/research/train_predictive_follow.py` | 规则行为克隆 + PPO，输出可加载的 actor |
| `perception/learned_predictor.py` | 监督式未来位置预测器：历史 -> 0.5/1/2 s 位移 |
| `perception/research/train_trajectory_predictor.py` | 生成轨迹窗口并训练/验证预测器；`--no-absolute` 做 route-blind 消融 |
| `perception/research/eval_trajectory_predictor.py` | 在训练路线和未见路线上评估预测器 |
| `perception/stage_map.py` | 舞台墙占据图 + 2D 线段可见性（无 Isaac Gym 依赖） |
| `perception/sim/stage_follow.py` | 可选 `--policy/--predictor`，并把监督+RL 接入舞台闭环 |

设计上参考了你给的源码包里的 `recovery_tracker` / `assisted_tracker` 思路：
**不可变 anchor、外观与运动双门限、歧义时宁可 abstain、memory 窗口内重捕获**。
实现没有直接搬它的 ROS / Isaac Sim 依赖，而是在本仓库的相机与仿真接口下重写成
纯 NumPy 的世界坐标层，方便单测和替换。

## 2. 感知层怎么工作

### 2.1 所有人都建轨迹，只有一个是目标

跟踪器不是“目标模板匹配”，而是对每帧所有检测都维护轨迹，再给其中一条打上
`target_id`。这样目标与干扰人重叠时，干扰人仍然有自己的轨迹，不会把目标的
轨迹挤掉。

每条轨迹有：

- **不可变身份锚** `anchor`：第一次关联时的外观向量，之后永不更新；
- **短期外观** `feature`：只在“外观相似度高 + 检测置信度高 + 时间连续”时缓慢更新；
- 世界坐标匀速 Kalman 状态 `[x, y, vx, vy]` 和协方差；
- `last_measure_t` / `last_seen_t` / `hits` / `confirmed`。

目标的身份锚不随匹配帧数漂移，这是“不慢慢跟错人”的关键。

### 2.2 关联：外观 + 运动 + 歧义拒识

关联代价是外观余弦距离加运动距离，用匈牙利算法分配。目标轨迹额外有：

- 更紧的位置门限（默认 0.9 m + 速度项）；
- 面积比门限：拒绝“目标 + 遮挡者”的合并框和遮挡边缘的细条框；
- 检测置信度门限；
- **歧义拒识**：如果目标轨迹的最佳检测和次佳检测代价差小于 `ambiguity_margin`，
  这一帧不更新目标，让它按预测走；下一帧再判断。对导航来说，
  “短暂预测”比“赌一个身份”安全。

### 2.3 丢失后的状态机

| 状态 | 条件 | 输出 |
|---|---|---|
| `measured` | 本帧有检测关联到目标 | 滤波后的世界位置 |
| `predicted` | 无检测，但丢失时间 ≤ `max_prediction_s`（默认 1.0 s） | Kalman 预测位置、协方差、未来 1 s 轨迹 |
| `lost` | 丢失时间 > 1.0 s | 仍给最后预测位置用于诊断，但 `lost=True`，控制器应停止跟随 |

轨迹本身会保留 `max_memory_s`（默认 5 s）。目标在 5 s 内回来，优先关联回原轨迹；
超过 5 s 轨迹被清掉，下一帧按不可变 anchor 在现存轨迹里重新识别。

### 2.4 像素 → 世界坐标

`range_bearing.py` 处理 20 机器人前双目：

- 方位角由检测框中心相对光心得到；
- 距离优先用框下部深度，深度与“身高 × 焦距 / 框高”单目估计做一致性检查；
- 两者偏差超过阈值时标 `consistent=False`，上层**不拿这个位置更新滤波器**，
  只保留外观匹配，让运动模型继续预测。这解决了“部分遮挡时深度打到遮挡物、
  单目高度又变小，导致位置跳变”的问题；
- 输出带协方差的 `base_link` 地面点，再按机器人位姿转到世界坐标。

### 2.5 输入/输出接口

输入每帧一个 `list[PersonDetection]`：

```python
PersonDetection(
    position_xy=np.array([x, y]),   # 世界坐标，可为 None
    bbox=np.array([x1, y1, x2, y2]),# 图像框，仅用于选择和诊断
    appearance=np.ndarray,          # HSV / ReID embedding
    score=0.9,
    covariance_xy=np.diag([...]),
)
```

输出 `TargetBelief`：

```python
position_xy        # 世界坐标
velocity_xy        # m/s
covariance_xy      # 2x2
status             # measured / predicted / lost
lost               # 控制器用的布尔量
track_id
measurement_age_s
identity_score     # 与不可变 anchor 的余弦
predicted_trajectory  # 未来 1 s，每 0.1 s 一个点
```

## 3. 结果

### 3.1 合成检测基准（10 seeds）

检测位置加 6 cm 高斯噪声，外观向量加小噪声；GT 只用于打分。

| 场景 | identity | 跟错人 | lost | 1.0 s 预测误差 | 轨迹恢复（同轨迹 / 重识别） |
|---|---:|---:|---:|---:|---:|
| clean | 100% | 0% | 0% | — | — |
| crossing（不同外观） | 100% | 0% | 0% | — | — |
| similar_crossing（外观余弦 ~0.98） | 99.88% | 0.12% | 0% | — | — |
| dropout_1s（匀速） | 100% | 0% | 0% | **3.8 cm** | 100% 同轨迹 |
| dropout_turn（丢失中 90° 转弯） | 100% | 0% | 0% | **21.5 cm** | 100% 同轨迹 |
| dropout_2s | 100% | 0% | 12.4% | 6.3 cm | 100% 同轨迹 |
| long_loss_reid（丢失 5 s，轨迹已清） | 100% | 0% | 39.6% | 6.3 cm | 0% 同轨迹 / 100% 重识别 |
| reid_after_memory（轨迹已清，按外观重识别） | 100% | 0% | 24.7% | 5.8 cm | 0% 同轨迹 / 100% 重识别 |

要点：匀速丢失预测非常准；丢失中转弯会有约 0.2 m 的不可避免误差，
但身份不丢、跟错人为 0。超过 1 s 后 `lost` 比例上升是设计行为，不是故障。

复现：`artifacts/perception/target_perception_20261102/synthetic_benchmark.json`

### 3.2 真实渲染 + YOLO：1 秒消失

场景：Isaac Gym 前双目 768×1024，Rocketbox 人物，目标与干扰人都用
`Business_Male_01/02`（外观刻意相近）。目标从 `(4, -1)` 以 0.6 m/s 向 +y 走；
第 2.0–2.9 s 把目标从场景移走（受控消失，评分仍按虚拟轨迹）；
干扰人站在 `(5, 2.5)` 全程可见。检测器为 `l1_front_pair_v1/weights/best.pt`，
置信度阈值 0.25，10 Hz。

| 测距方式 | identity | 消失期间 predicted | 0.5 s 误差 | 1.0 s 误差 | 重捕获 | 同一轨迹 |
|---|---:|---:|---:|---:|---:|---:|
| 深度 + 单目融合（深度经验证的帧） | 100% | 100% | 0.158 m | **0.155 m** | ≤0.3 s | 是 |
| 仅单目框高 | 100% | 100% | 0.178 m | **0.165 m** | ≤0.3 s | 是 |

两个 episode 的 visible RMSE 分别为 0.169 / 0.186 m（深度）和 0.291 / 0.242 m（单目）。
可见帧误差和消失帧误差几乎一样，说明误差主要是**固定定位偏差**，
不是预测发散。干扰人全程在画面里，`wrong_frames=0`。

视频：`artifacts/perception/target_perception_20261102/live_depth/dropout_episode00.mp4`

**重要说明**：这是一个“受控消失”，不是物理墙遮挡。真实物理遮挡实验也试过：
薄墙/宽墙会造成部分框、深度打到墙、单目高度变小，位置估计会跳；
当前层用“深度/单目一致性检查 → 不一致就不更新位置”来兜底，
所以真实遮挡下会表现为短暂 `predicted` 或 `lost`，而不是跟错人。
要把物理遮挡也做成漂亮的连续曲线，下一步应该让检测器同时输出实例 mask，
用 mask 内像素做深度和框高估计。

### 3.3 闭环跟随（旧机器人模型 + P0 模型）

`follow_p0.py --target-layer --use-depth` 把世界坐标层接到现有 P0 闭环：
预测状态 `predicted` 时控制器继续跟随预测位置；超过 1 s 才进入 SEARCH。

| 场景（300 帧，30 s） | C（正确跟踪） | W（跟错人） | A（无输出） | 覆盖率 | 最小距离 |
|---|---:|---:|---:|---:|---:|
| weave | 269 | **0** | 31 | 89.7% | 2.55 m |
| crossing | 252 | **0** | 48 | 84.0% | 2.56 m |

`lost_events=0`、`false_relocks=0`。crossing 的覆盖率低一些，主要是目标转向/检测
漏帧造成的 A 帧，不是跟错人；300 帧里没有一帧 W。

### 3.4 复用现成场景套件（先跑这批）

新加的 `perception/eval_reusable_scenarios.py` 不重新渲染，直接吃仓库里已有的数据：

| 数据 | 场景/特点 | 规模 | 输入口径 |
|---|---|---|---|
| `l1_crossing_v2/gap035,gap060,gap100` | 两人对向交叉，深度差 0.35/0.6/1.0 m | 每组 8 episode × 150 帧 | 缓存 YOLO 检测框 + 单目框高测距 |
| `l1_crossing_v2/biz035` | 交叉 + 两人都是相似 Business 头像 | 8 episode × 150 帧 | 缓存 YOLO 检测框 + 单目框高测距 |
| `sim_v4_avatar_dev/test` | 24 个 Rocketbox 人物、6 个环境、随机墙/柱 | 各 36 组 × 40 帧 | GT 框 + GT 世界位置（只评跟踪器） |
| `l1_front_pair_v1` | 前双目 768×1024，4 人 + 杂乱场景 | 24 组 × 50 帧 | GT 框 + GT 世界位置 |
| `l1_appearance_v1` | 外观/人物划分，4 人 | 16 组 × 40 帧 | GT 框 + GT 世界位置 |

结果（`artifacts/perception/target_perception_20261102/reusable_scenarios.json`）：

| 数据集 | 有效组 | 帧数 | identity | 跟错人 | lost | 1.0 s 预测误差 |
|---|---:|---:|---:|---:|---:|---:|
| crossing/gap035 | 8 | 1200 | **100%** | 0% | 1.8% | — |
| crossing/gap060 | 8 | 1200 | **100%** | 0% | 0.9% | — |
| crossing/gap100 | 8 | 1200 | **98.0%** | 2.0% | 1.1% | — |
| crossing/biz035（相似外观） | 8 | 1198 | **100%** | 0% | 0% | — |
| avatars/dev | 10 | 390 | **98.5%** | 1.5% | 16.9% | 0.007 m |
| avatars/test | 11 | 367 | **100%** | 0% | 10.6% | 0.017 m |
| front_pair/clutter | 23 | 1088 | **100%** | 0% | 3.5% | 0.024 m |
| front_pair/appearance | 10 | 396 | **100%** | 0% | 14.4% | 0.013 m |

几个要点：

- **交叉场景**：3/4 组 identity 100%，相似外观 `biz035` 也是 100%。唯一掉到
  98% 的 `gap100` 是两人深度差 1 m 的交叉：缓存检测框在交叉遮挡时变矮，
  单目框高把距离估远，位置被拉向后面的人。用 `--crossing-position gt_match`
  把同一批检测框换成匹配到的 GT 位置后，`gap100` identity 回到 **100%**
  （`reusable_crossing_gtpos.json`），说明 2% 是测距偏差，不是身份关联失败。
  真实部署应优先用双目深度/实例 mask，而不是交叉遮挡时的单目框高。
- **24 人物/多环境**：`sim_v4_avatar_*` 里目标经常大面积不可见（test 只有 28%
  的帧可见），有效初始化的组只有 10/11；目标消失期间靠 `predicted`/`lost`
  过渡，identity 98.5%/100%。26/25 组被跳过是因为目标全程没有可用框，
  不是跟踪失败。
- **前双目杂乱场景**：`l1_front_pair_v1` 的 23 组有效数据 identity 100%，
  包含近距/远距、墙体/柱子、4 人同框；这是目前最接近“多人杂乱”的一批。
- **预测误差**：avatar/front-pair 场景用 GT 位置，1 s 预测误差 0.007–0.024 m；
  这些误差主要来自 Kalman 过程和初始位置偏差，不是测距。
- 这批数据已经够做第一轮回归：建议固定 `crossing/gap035`、`crossing/biz035`、
  `avatars/test`、`front_pair/clutter` 四组作为日常 smoke，`gap100` 留给
  测距/mask 改进时对比。

### 3.5 地图场景（polygon maps）

地图数据直接复用你给的源码包里的 `rl_validation.json` / `rl_test.json`：
30×30 m、polygon 障碍、带 start/goal，family 包括 `irregular_clutter`、
`forced_hairpin`、`u_goal_inside`。已复制到 `perception/assets/maps/`。

`perception/sim/map_perception.py` 做了三件事：

1. 把机器人放在地图的 `start`，朝向 `goal`，把障碍多边形转到机器人坐标系；
2. 每个 polygon 用 **OBB（有向包围盒）** 加一个 1.5 m 高的墙体；
3. 目标沿一条人工挑好的、不穿墙的走廊走，墙体自然造成“可见 → 墙后消失 → 再出现”。

| 地图场景 | family | 帧数 | 可见/遮挡 | identity | 跟错人 | 遮挡期 predicted/lost | 重捕获 | 同轨迹 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| irregular_clutter_pass | map_013_irregular_clutter | 70 | 58 / 12 | **100%** | 0% | 0% / 100%* | 1.2 s | 是 |
| forced_hairpin_uturn | map_004_forced_hairpin | 70 | 53 / 17 | **100%** | 0% | 35% / 65% | 0.4 s | 是 |
| u_goal_uturn | map_012_u_goal_inside | 70 | 67 / 3 | **100%** | 0% | 100% / 0% | 0.4 s | 是 |

*irregular_clutter 的目标在墙边先进入检测缺口：层在缺口里预测了约 1 秒，
之后 `lost`；GT 可见性稍后才掉到 30 px 以下，所以“遮挡期 predicted”这个
按 GT 可见性统计的口径是 0。按“检测缺口”统计则是 14 帧缺口、1 帧 predicted、
13 帧 lost；身份始终没丢。

关键结果：

- 三张地图 **identity 100%、跟错人 0 帧**，包括目标从墙后回来之后。
- `forced_hairpin` 是最完整的地图 1 秒预测案例：目标墙后 1.7 s，前 1 s 由
  Kalman 预测接管，之后 `lost`；回来时 0.4 s 内重捕获，`same_track=True`。
- `u_goal` 只有 0.3 s 的短遮挡，遮挡期 100% 是 `predicted`。
- 地图场景的定位 RMSE 约 0.46–0.60 m，主要来自单目框高在墙体/部分框上的偏差，
  不是身份错误；下一步加实例 mask 或双目深度会直接改善这一项。

运行：

```bash
/home/jason/legged_gym/.venv/bin/python perception/sim/map_perception.py \
    --out /home/jason/perception_data/runs/map_perception_v3 \
    --scenario all --frames 70 --video
```

视频和逐帧结果在 `artifacts/perception/target_perception_20261102/maps/`。

### 3.6 地图闭环跟随：目标和球一起穿过障碍

`perception/sim/map_follow.py` 把地图场景接成闭环：

- 目标沿地图 `start -> goal` 的 A* 路线走（目标不知道机器人，只按路线走）；
- 机器人**只从感知层拿目标位置/速度**，不知道目标的真实路线；
- 机器人用地图 + 感知目标位置，每 0.8 s 重规划一条到“目标身后 1.5 m”的 A* 路径；
- 纯跟踪 + 候选曲率搜索选择 P0 可执行的 `(a1, a2)`，同时检查 1 s 弧线最小墙距；
- 目标丢失 ≤1 s 用预测位置继续；超过 1 s 停车，等目标重新出现。

三张地图都跑通了（`artifacts/perception/target_perception_20261102/map_follow/`）：

| 地图 | 帧数 | 目标路线长度 | identity | 平均跟距 | 最小墙距 | 碰撞帧 | measured/predicted/lost | 终点进度 | success |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| irregular_clutter | 320 | 26.9 m | **100%** | 3.11 m | **0.63 m** | **0** | 65.0% / 25.9% / 9.1% | 68.7% | ✅ |
| forced_hairpin | 180 | 7.1 m | **100%** | 2.52 m | **0.69 m** | **0** | 81.1% / 6.1% / 12.8% | 100% | ✅ |
| u_goal | 220 | 10.2 m | **100%** | 2.81 m | **0.88 m** | **0** | 93.6% / 6.4% / 0% | 100% | ✅ |

要点：

- **目标和球都穿过了障碍**：`irregular_clutter` 的目标路线 26.9 m，机器人跟了 320 帧，
  最小墙距 0.63 m，几何上零碰撞；`forced_hairpin` 和 `u_goal` 都到达终点。
- **目标不是喂给控制器的**：控制器只拿 `TargetBelief.position_xy / velocity_xy`；
  目标在墙后时 predicted 比例 6–26%，说明感知预测确实在闭环里起作用。
- **没有跟错人**：三张图 identity 都是 100%，干扰人一直在场景里。
- `forced_hairpin` 后段 lost 12.8%：目标到终点后停在墙后，预测超时后机器人停车；
  目标再次可见时身份仍然正确，属于预期的“到点等待”。
- 定位/跟距不是重点：平均跟距 2.5–3.1 m，最小 1.19 m；最小墙距 0.63 m，
  机器人半径约 0.4 m，仍有余量。

视频：

- `artifacts/perception/target_perception_20261102/map_follow/irregular_clutter.mp4`
- `.../forced_hairpin.mp4`
- `.../u_goal.mp4`

运行：

```bash
/home/jason/legged_gym/.venv/bin/python perception/sim/map_follow.py \
    --out /home/jason/perception_data/runs/map_follow_v1 \
    --map irregular_clutter --frames 320 --video
```

限制：地图是已知的（相当于机器人有地图）；避障用的是地图 OBB + 几何弧线检查，
不是激光/深度动态避障；P0 是识别模型，没有物理碰撞响应；目标路线是 A* 的静态路线，
没有加入行人动态避让。下一步可以接 lidar/depth、动态障碍和多目标。

### 3.7 开源场景：HouseExpo 室内户型

除了你给的 map JSON，这次还接了一个**许可清晰的开源场景项目**：

- 数据集：**HouseExpo — A Large-scale 2D Indoor Layout Dataset**
- 仓库：`https://github.com/TeaganLi/HouseExpo`
- 许可：**MIT**
- 本地内容：12 套选中户型（`perception/assets/maps/houseexpo/`），转换脚本
  `perception/tools/houseexpo_to_maps.py`

转换方式：

- HouseExpo 的 `verts` 是墙线 polyline，逐段变成 0.10 m 厚、1.5 m 高的墙；
  门洞保持开口。
- start/goal 取两个不同房间的中心；地图 JSON 的 schema 与现有地图一致。
- 户型整体放大 **1.35×**：HouseExpo 门宽常约 0.8 m，球形机器人直径也约 0.8 m，
  不缩放会贴边甚至几何重叠；缩放是明确的“机器人尺寸适配”，不是原始数据比例。

闭环跟随（目标沿 A* 路线穿过室内门洞，球跟着）：

| 户型 | 帧数 | 路线长度 | identity | 平均跟距 | 最小墙距 | 几何碰撞帧 | measured | 进度 | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| houseexpo_a64a310dfa11 | 220 | 17.6 m | **100%** | 3.14 m | 0.55 m | **0** | 100% | 70.4% | ✅ |
| houseexpo_ca5e42107f1d | 220 | 17.5 m | **100%** | 3.11 m | 3.08 m | **0** | 100% | 70.6% | ✅ |
| houseexpo_4769e72cf173 | 220 | 17.5 m | **100%** | 3.14 m | 0.49 m | **0** | 100% | 70.5% | ✅ |

结果说明：

- 三套户型都是 identity 100%、几何碰撞 0；最小墙距 0.49–3.08 m。
- 这三条路线大多保持在视线内，所以 measured 100%；室内遮挡/丢失场景由
  3.5/3.6 的 polyline 地图场景补足。
- HouseExpo 是 2D 墙线，不是带家具/贴图的 3D 场景；它补的是**真实户型拓扑**
  （多个房间、门洞、走廊），不是视觉真实感。
- 本地只保留 12 套选中户型；完整 HouseExpo 约 25 MB，网络下载不全时用选中的
  子集即可。`SOURCE.md` 记录了来源、MIT 许可和转换/缩放说明。

视频：

- `artifacts/perception/target_perception_20261102/houseexpo/houseexpo_a64a310dfa11.mp4`
- `.../houseexpo_ca5e42107f1d.mp4`
- `.../houseexpo_4769e72cf173.mp4`

运行：

```bash
/home/jason/legged_gym/.venv/bin/python perception/sim/map_follow.py \
    --out /home/jason/perception_data/runs/houseexpo_follow_v1 \
    --map-json perception/assets/maps/houseexpo/houseexpo_a64a310dfa11.json \
    --scene-id houseexpo_a64a310dfa11 \
    --plan-inflation 0.42 --replan-inflation 0.40 \
    --frames 220 --video
```

### 3.8 真实 ReplicaCAD 公寓：40 s 复杂环路

前面 3.7 是 2D 户型；这一节用真正的 3D 场景，并把路线加长、加复杂：

- 视觉场景：ReplicaCAD **frl_apartment_stage**（Meta，CC BY-NC 4.0），
  木地板、墙、门洞、楼梯；Blender 转 OBJ/URDF 后由 Isaac Gym 加载。
- 遮挡物：ReplicaCAD **refrigerator**，放在客厅中。
- 目标路线：客厅里的矩形闭环
  `(-1.2,-1.1) -> (1.2,-1.1) -> (1.2,-3.2) -> (-1.2,-3.2) -> ...`，
  一圈约 9 m，速度 0.4 m/s。400 帧 = 40 s，约 1.7 圈。
- 机器人：在观察走廊 `y=-4.0` 上横向跟随目标的 x，相机独立对准目标/预测；
  目标进入遮挡时停车，不向冰箱或墙移动。控制器的间距检查同时使用
  **整张舞台 OBJ 的 5 cm 占据图 + 冰箱 OBB**。

| 指标 | 结果 |
|---|---:|
| 帧数 / 时长 | **400 / 40 s** |
| 目标圈数 | 约 1.7 圈 |
| identity | **100%** |
| 几何碰撞帧 | **0** |
| 最小墙/冰箱距离 | **0.707 m** |
| 平均跟距 | 2.37 m |
| 遮挡事件数 | 7 |
| 最长连续遮挡 | 8.7 s |
| 同轨迹恢复 | `same_track=True` |
| success | ✅ |

说明：

- **不再撞墙**：之前只检查冰箱，机器人最后会穿进布景墙；现在整张 `stage.obj`
  的竖直三角面被栅格化成 5 cm 占据图，墙、门、楼梯都参与间距检查。最小墙距
  0.707 m，几何碰撞帧 0。
- 复杂环路里目标多次被冰箱/布景遮挡，出现 7 段遮挡，最长 8.7 s；机器人保持
  观察走廊，不追进遮挡区，因此不会撞墙。代价是：这一段路线下机器人比较保守，
  遮挡期间以保持位置/跟踪相机为主，不是激进的全自主搜索。
- 目标身份始终 100%，最后仍是同一条轨迹；`success` 由“identity + 无碰撞 +
  至少 1 s 遮挡”判定。

视频/关键帧：

- `artifacts/perception/target_perception_20261102/stage_follow/stage_follow.mp4`
- `.../preview_start.png`
- `.../preview_hidden.png`
- `.../preview_reacq.png`

运行：

```bash
source local_env.sh
/home/jason/legged_gym/.venv/bin/python perception/sim/stage_follow.py \
    --out /home/jason/perception_data/runs/stage_follow_complex_v8 \
    --frames 400 --video
```

### 3.9 感知层增强 + 预测性丢失跟随（规则 baseline）

3.8 的机器人遮挡时基本停车，没有用上“消失前速度”。这一轮补了两层：

**感知层新增输出**

- `velocity_confidence`：用最近 3 s 轨迹做匀速拟合，综合时间跨度、拟合残差、
  相邻速度稳定性和丢失时间；丢失后按 `exp(-age/2)` 衰减。
- `trajectory_history` / `history_span_s`：最近 3 s 的观测轨迹，供预测和控制使用。
- `prediction_uncertainty_m`：当前位置协方差的最大特征值平方根，随丢失时间增长。
- `predicted_trajectory[*].confidence`：每个未来点的置信度，随预测时域衰减。

**控制规则（先做可解释 baseline，不急着上 RL）**

- 目标 `lost` 但 `age ≤ 2 s`、`velocity_confidence ≥ 0.35`、速度 > 0.05 m/s：
  - 用 `last_position + last_velocity * age` 预测位置；
  - 沿安全观察走廊继续横移，速度按 `0.55 * (1 - age/2)^1.5` 衰减；
  - 每一步仍检查舞台墙 + 冰箱间距，间距不足立即停车。
- `predicted` 状态：正常跟随，但速度乘 0.70。
- 预测不可信或超时：停车扫相机，等目标重新出现。

**复杂环路对比（400 帧 / 40 s，7 段遮挡）**

| 指标 | 3.8 基线（丢失即停） | 3.9 预测性跟随 |
|---|---:|---:|
| visible frames | 213 | **325** |
| hidden frames | 187 | **75** |
| measured rate | 10.3% | **17.0%** |
| predicted rate | 12.5% | **24.8%** |
| lost rate | 77.3% | **58.3%** |
| predictive-follow frames | 0 | **30** |
| 最长连续遮挡 | 8.7 s | **3.2 s** |
| identity | 100% | **100%** |
| 几何碰撞帧 | 0 | **0** |
| 最小墙/冰箱距离 | 0.707 m | 0.559 m（仍安全） |
| 同轨迹恢复 | 是 | 是 |

结论：

- “消失前向哪走、下一步就向哪走但减速”这个规则在仿真里是有效的：可见帧从 213
  提升到 325，lost 从 77% 降到 58%，最长遮挡从 8.7 s 降到 3.2 s，仍然零碰撞。
- 这是一版**规则 baseline**。`velocity_confidence`、预测窗口、衰减曲线、间距阈值
  都已经显式暴露，随后在 3.10 中用它们作为 RL 的观测/动作参数做了对比。
- 最小间距从 0.707 m 降到 0.559 m，机器人半径约 0.4 m，仍有约 0.16 m 余量；
  如果要更保守，可以把 `--predict-speed-scale` 从 0.55 降到 0.4。

运行：

```bash
/home/jason/legged_gym/.venv/bin/python perception/sim/stage_follow.py \
    --out /home/jason/perception_data/runs/stage_predictive_v1 \
    --frames 400 --video
```


### 3.10 监督预测器 + RL：舞台校准、训练与闭环对比

这一轮把“预测”拆成两层，而不是端到端 RL：

1. **监督模型**学习目标动力学：用目标历史预测未来 0.5/1/2 秒位置；
2. **RL** 只学习决策：给定预测器输出和信念特征，输出横移速度；
   丢失阶段再加一个方向安全检查，动作若明显背离监督预测点就回退到规则。

**先修了一个会污染所有对比的问题**

舞台 demo 原来的初始朝向是 `+y`，目标在视场边缘；检测框高度被截断，
单目高度先验把 3.6 m 的人当成 10+ m，`belief.position_xy` 误差约 15 m。
旧报告里的 3.8/3.9 指标是在这个配置下测得的。修正后：

- 初始朝向改为朝向目标；
- 舞台帧启用 Isaac Gym 深度；当“深度比高度先验近”时接受深度（这正是
  上半身框被截断的情形），深度明显比高度先验远时仍拒识；
- 修正后首帧目标位置误差约 0.46 m（此前 15.7 m），舞台 `hidden_rmse`
  从 ~13 m 降到 ~2.7 m。

**监督预测器**

`perception/learned_predictor.py` + `research/train_trajectory_predictor.py`：

- 输入：最近 12 个测量点（相对位置、速度、时间间隔、绝对位置上下文）；
- 输出：0.5/1/2 s 位移；
- 数据：同一矩形路线，随机方向/速度/相位/测量噪声，300 train / 60 val
  条轨迹，30k/6k 窗口；
- 结果（held-out 轨迹种子）：

| horizon | 监督模型 MAE | 平滑常数速度 | 原始最后两帧常数速度 |
|---:|---:|---:|---:|
| 0.5 s | **0.049 m** | 0.097 m | 0.418 m |
| 1.0 s | **0.068 m** | 0.173 m | 0.801 m |
| 2.0 s | **0.109 m** | 0.395 m | 1.584 m |

注意：这里故意把绝对位置作为上下文，因此模型学到了当前舞台路线的转弯
先验；换一张全新地图不保证保持这个误差。

**RL 环境**

`research/predictive_follow_env.py` 加入两项舞台校准：

- `stage_visibility=True`：用 `perception/stage_map.py` 的舞台墙占据图 +
  冰箱 OBB 做线段可见性；
- `detection_dropout=0.30`、`measurement_noise=0.05`：即使几何可见，
  检测也可能缺失/带噪，观测与 reward 都只看“测量到”的目标。

策略是 15D 观测 -> 1D 横移速度，BC 40 episodes + PPO 120 iterations
（8 episodes × 30 s）。训练时已去掉两类仿真真值泄漏：可见时不再给
GT 目标位置/速度，路线进度改用预测点计算；舞台侧策略也只用
`TargetBelief`，GT mask 只用于打分和初始选择。

**丢失阶段的安全回退**

RL 动作不是无条件执行：

- 若监督预测器没有可用历史：回退规则；
- 若 `|predicted_x - robot_x| > 0.20 m` 且 RL 动作方向与之相反：
  回退规则。

这是“监督预测 + RL + 安全 shield”，不是纯端到端 RL；它的作用是避免
策略在长遮挡里把球带离预测方向。

**ReplicaCAD 闭环对比**

600 帧（约 2.6 圈）在 3 个随机种子上：

| 指标 | 规则 + 监督预测器 | 监督预测器 + RL + shield |
|---|---:|---:|
| visible frames (mean) | 481.3 | **546.7** |
| hidden frames (mean) | 118.3 | **53.0** |
| 最长连续遮挡 (mean) | 9.1 s | **3.6 s** |
| identity (mean) | 99.0% | **100%** |
| 几何碰撞 | 0 | 0 |
| 最小墙/冰箱距离 | 0.559 m | 0.559 m |
| 同轨迹恢复 | 是 | 是 |

逐种子 600 帧结果：

| seed | 规则 + 预测器 | 监督 + RL + shield |
|---|---|---|
| 20261102 | 535 visible / 65 hidden / 4.3 s | 538 / 62 / 4.7 s |
| 7 | 471 / 128 / 7.7 s | **569 / 30 / 1.1 s** |
| 13 | 438 / 162 / 15.3 s | **533 / 67 / 4.9 s** |

400 帧单种子上两者接近（343/57 vs 343/57），说明 400 帧不足以稳定区分；
600 帧、多种子后 RL 的收益才清楚。

**2D stage-like 环境**

| 策略 | visible rate | hidden frames | collisions | reacquired |
|---|---:|---:|---:|---:|
| 规则 + 监督预测器 | 63.02% | 2219 | 0 | 20/20 |
| 监督预测器 + BC/PPO | **64.68%** | **2119** | 0 | 20/20 |

结论：

- “有监督模型 + RL”在当前设置下是有效的：监督模型负责目标动力学，
  RL 负责动作选择，安全回退保证不退化到旧版规则的坏分支。
- 证据是多 seed、长时长的舞台闭环：可见帧 +13.6%，hidden 帧 -55%，
  最长遮挡从 9.1 s 降到 3.6 s，且 identity 从 99.0% 提升到 100%。
- 当前模型的路线先验较强；下一步应做路线/遮挡随机化和跨地图验证，
  否则它更像“这张舞台的跟随专家”，不是通用行人跟随策略。

### 3.11 消融：绝对位置是“路线记忆”还是动力学？

为了验证 3.10 中监督预测器的 6.8 cm 是否来自当前舞台路线，做了一个干净的
route-blind 消融：**保持数据、标签、网络和训练超参不变，只删掉输入中的
绝对世界坐标和绝对锚点上下文**，只保留相对历史、速度、时间间隔和有效位。

**离线误差（1 s，held-out 轨迹种子）**

| 路线 | route-aware | route-blind | 平滑常数速度 |
|---|---:|---:|---:|
| 训练同路线 | **0.071 m** | 0.143 m | 0.181 m |
| 未见：旋转矩形 | 0.277 m | **0.197 m** | 0.168 m |
| 未见：窄矩形 | **0.155 m** | 0.159 m | 0.204 m |

对应 0.5 / 2 s 的趋势相同：route-aware 在同路线最好，但旋转路线退化到
0.163/0.449 m；route-blind 是 0.122/0.399 m，退化明显更小。结论很直接：

- 绝对位置让模型记住了“这张图的转角”，同路线 7.1 cm 不是通用动力学；
- 去掉绝对位置后，跨路线退化从 +20.6 cm 降到 +5.4 cm（1 s，旋转路线），
  但**仍然没有在所有未见路线上超过平滑常数速度**；
- 所以它学到的是一部分历史动力学 + 较弱的路网先验，不是“通用行人模型”。

**接回舞台闭环（600 帧）**

| 配置 | visible | hidden | 最长遮挡 | identity |
|---|---:|---:|---:|---:|
| 规则 + route-aware 预测器 | 535 | 65 | 4.3 s | 99.7% |
| 规则 + route-blind 预测器 | 340 | 260 | 16.9 s | 97.8% |
| RL + route-aware 预测器（3 seeds 均值） | **547** | **53** | **3.6 s** | **100%** |
| RL + route-blind 预测器（3 seeds 均值） | 500 | 100 | 4.9 s | 100% |

Route-blind RL 的逐种子里程碑波动很大：默认种子 580/20/1.0 s，seed 7
390/209/8.9 s，seed 13 529/71/4.8 s。它说明两件事：

- 去掉路线记忆后，策略有时能靠 RL 和相对历史补回来，甚至单次跑得更好；
- 但稳定性不够，不能用“单次 580 帧”替代多种子均值。

**结论**

- 之前 6.8 cm 的提升确实**部分来自路线记忆**；用户的怀疑是合理的。
- 这不是数据泄漏：推理只用在线估计位置，没有用 GT 或未来；问题在于
  **评估只在同一路线**，以及模型把绝对位置当成了捷径。
- 当前最诚实的定位：route-aware 版本是“已知地图/固定路线下的专用预测器”；
  route-blind 版本更泛化，但还没有稳定超过平滑常数速度或路线专用版本。
- 下一步必须做**路线/遮挡/速度随机化训练**，并同时报告 route-aware 和
  route-blind 的同路线、跨路线结果；否则不能声称通用行人跟随。

## 4. 与“最新机器人模型 / 更真实场景”的关系

- **旧模型可用**：当前仿真和闭环用的是仓库里的旧球形机器人模型，感知层不依赖
  动量轮。新模型主要增加动量轮，若相机安装位置/朝向不变，感知层和跟踪器无需改；
  若相机外参变了，只需更新 `range_bearing.FrontPairCamera` 的几何参数并重跑标定，
  不需要重训跟踪器。
- **场景**：现在用 Rocketbox 人物 + 简单几何场景；后续换更真实场景时，
  检测器需要新的仿真/真实数据微调，但感知层的输入契约
  （框 + 外观 + 世界坐标 + 协方差）不变。
- **物理遮挡**：如 3.2 所述，下一步优先给检测器加 mask 输出，用实例 mask
  计算框高和深度，才能把“被墙挡住半边身体”也稳定地连续跟踪。

## 5. 复现

```bash
cd /media/jason/Jason‘s\ u/SphericalRobot_PathFollower_20260911

# 1. CPU 单元测试（NumPy only）
python3 -m unittest perception.test_target_perception -v

# 2. 合成基准（10 seeds）
python3 perception/eval_target_perception.py \
    --out artifacts/perception/target_perception_20261102/synthetic_benchmark.json \
    --seeds 10

# 3. 复用现成数据集的场景套件（Ultralytics venv，NumPy 2 读取 det_cache）
/home/jason/.venvs/ultralytics-rice/bin/python perception/eval_reusable_scenarios.py \
    --out artifacts/perception/target_perception_20261102/reusable_scenarios.json

# 诊断：crossing 用 GT 位置隔离单目测距偏差
/home/jason/.venvs/ultralytics-rice/bin/python perception/eval_reusable_scenarios.py \
    --groups crossing --crossing-position gt_match \
    --out artifacts/perception/target_perception_20261102/reusable_crossing_gtpos.json

# 4. 真实渲染 + YOLO 的一秒消失实验（Isaac Gym venv）
source local_env.sh
/home/jason/legged_gym/.venv/bin/python perception/sim/occlusion_perception.py \
    --out /home/jason/perception_data/runs/occlusion_perception_depth_v3 \
    --episodes 2 --frames 40 --use-depth --video

# 只用单目框高
/home/jason/legged_gym/.venv/bin/python perception/sim/occlusion_perception.py \
    --out /home/jason/perception_data/runs/occlusion_perception_height_v1 \
    --episodes 2 --frames 40 --video

# 5. 地图场景：polygon 墙体 + 墙后消失/重现
/home/jason/legged_gym/.venv/bin/python perception/sim/map_perception.py \
    --out /home/jason/perception_data/runs/map_perception_v3 \
    --scenario all --frames 70 --video

# 6. 地图闭环跟随：目标走 A*，球跟着穿障碍
/home/jason/legged_gym/.venv/bin/python perception/sim/map_follow.py \
    --out /home/jason/perception_data/runs/map_follow_v1 \
    --map irregular_clutter --frames 320 --video

# 7. 开源 HouseExpo 室内户型闭环
/home/jason/legged_gym/.venv/bin/python perception/sim/map_follow.py \
    --out /home/jason/perception_data/runs/houseexpo_follow_v1 \
    --map-json perception/assets/maps/houseexpo/houseexpo_a64a310dfa11.json \
    --scene-id houseexpo_a64a310dfa11 \
    --plan-inflation 0.42 --replan-inflation 0.40 \
    --frames 220 --video

# 8. ReplicaCAD 真实公寓：40 s 复杂环路 + 预测性跟随
/home/jason/legged_gym/.venv/bin/python perception/sim/stage_follow.py \
    --out /home/jason/perception_data/runs/stage_predictive_v1 \
    --frames 400 --video

# 9. 闭环跟随：世界坐标层 + 1 秒预测
/home/jason/legged_gym/.venv/bin/python perception/sim/follow_p0.py \
    --out /home/jason/perception_data/runs/follow_p0_target_layer_crossing300 \
    --episodes 1 --frames 300 --target-layer --use-depth --scenario crossing
```


# 10. PPO 预测跟随：规则 BC 初始化 + PPO（2D 环境）
/home/jason/.venvs/ultralytics-rice/bin/python \
    perception/research/train_predictive_follow.py \
    --out /home/jason/perception_data/runs/predictive_follow_ppo_bc_v2 \
    --iters 60 --episodes 4 --episode-s 20 \
    --bc-episodes 40 --bc-steps 1500

# 11. 把 PPO actor 接进 ReplicaCAD（零样本，实验性）
/home/jason/legged_gym/.venv/bin/python perception/sim/stage_follow.py \
    --out /home/jason/perception_data/runs/stage_ppo_bc_v3 \
    --frames 400 --video \
    --policy /home/jason/perception_data/runs/predictive_follow_ppo_bc_v2/actor.pt


# 12. 监督式未来位置预测器（held-out 轨迹验证）
/home/jason/.venvs/ultralytics-rice/bin/python \
    perception/research/train_trajectory_predictor.py \
    --out /home/jason/perception_data/runs/trajectory_predictor_v2 \
    --epochs 30 --train-trajectories 300 --val-trajectories 60 \
    --samples-per-trajectory 100 --duration 20

# 13. 舞台式 RL：舞台可见性 + 检测丢失 + 监督预测器
/home/jason/.venvs/ultralytics-rice/bin/python \
    perception/research/train_predictive_follow.py \
    --out /home/jason/perception_data/runs/predictive_follow_ppo_stage_v9 \
    --iters 120 --episodes 8 --episode-s 30 --seed 7 \
    --bc-episodes 40 --bc-steps 1500 \
    --predictor /home/jason/perception_data/runs/trajectory_predictor_v2/predictor.pt \
    --stage-visibility --detection-dropout 0.30 --measurement-noise 0.05

# 14. 舞台闭环：规则 + 预测器 vs 监督 + RL + shield
/home/jason/legged_gym/.venv/bin/python perception/sim/stage_follow.py \
    --out /home/jason/perception_data/runs/stage_rule_predictor_v2_600 \
    --frames 600 --video \
    --predictor /home/jason/perception_data/runs/trajectory_predictor_v2/predictor.pt

/home/jason/legged_gym/.venv/bin/python perception/sim/stage_follow.py \
    --out /home/jason/perception_data/runs/stage_ppo_stage_v9guard_600 \
    --frames 600 --video \
    --predictor /home/jason/perception_data/runs/trajectory_predictor_v2/predictor.pt \
    --policy /home/jason/perception_data/runs/predictive_follow_ppo_stage_v9/actor.pt


# 15. 消融：route-blind 监督预测器（去掉绝对位置）
/home/jason/.venvs/ultralytics-rice/bin/python \
    perception/research/train_trajectory_predictor.py \
    --out /home/jason/perception_data/runs/trajectory_predictor_relative_v1 \
    --epochs 30 --train-trajectories 300 --val-trajectories 60 \
    --samples-per-trajectory 100 --duration 20 --no-absolute

# 16. 同路线 / 未见路线评估（route-aware vs route-blind）
/home/jason/.venvs/ultralytics-rice/bin/python \
    perception/research/eval_trajectory_predictor.py \
    --checkpoint /home/jason/perception_data/runs/trajectory_predictor_v2/predictor.pt \
    --out /home/jason/perception_data/runs/eval_predictor_absolute_v2.json

/home/jason/.venvs/ultralytics-rice/bin/python \
    perception/research/eval_trajectory_predictor.py \
    --checkpoint /home/jason/perception_data/runs/trajectory_predictor_relative_v1/predictor.pt \
    --out /home/jason/perception_data/runs/eval_predictor_relative_v1.json

# 17. route-blind 预测器接回舞台
/home/jason/legged_gym/.venv/bin/python perception/sim/stage_follow.py \
    --out /home/jason/perception_data/runs/stage_relative_policy_600 \
    --frames 600 --video \
    --predictor /home/jason/perception_data/runs/trajectory_predictor_relative_v1/predictor.pt \
    --policy /home/jason/perception_data/runs/predictive_follow_ppo_stage_relative_v1/actor.pt

## 6. 结果文件

| 路径 | 内容 |
|---|---|
| `artifacts/perception/target_perception_20261102/synthetic_benchmark.json` | 合成基准全部 episode 指标 |
| `.../predictive_follow/best.json` | 2D 规则 vs BC+PPO 的 20-episode 对比 |
| `.../predictive_follow/actor.pt` | BC+PPO actor（15D→1D） |
| `.../supervised_predictor/predictor.pt` | 监督未来位置预测器（0.5/1/2 s） |
| `.../supervised_rl/actor.pt` | 监督预测器输入的 BC+PPO actor |
| `.../stage_supervised_rl/summary.json` | 600 帧 3 seeds 规则 vs 监督+RL 汇总 |
| `.../supervised_predictor_relative/eval_seen_unseen.json` | route-blind 的同路线/未见路线误差 |
| `.../stage_supervised_rl_relative/summary.json` | route-blind 预测器接回舞台的 600 帧结果 |
| `.../stage_ppo/summary.json` | 早期 PPO 零样本实验（已被 3.10 取代） |
| `.../live_depth/summary.json` | 真实渲染 1 s 消失，深度融合 |
| `.../live_depth/dropout_episode00.mp4` | 真实渲染消失视频 |
| `.../live_height/summary.json` | 真实渲染 1 s 消失，仅单目 |
| `.../reusable_scenarios.json` | 4 组交叉 + 24 人物/多环境 + 前双目场景 |
| `.../reusable_crossing_gtpos.json` | crossing 用 GT 位置的诊断结果 |
| `.../maps/summary.json` | 3 张 polygon 地图的感知结果 |
| `.../maps/*.mp4` | 地图场景视频 |
| `.../map_follow/summary.json` | 3 张地图的闭环跟随汇总 |
| `.../map_follow/*.mp4` | 目标和球一起穿障碍的视频 |
| `.../map_follow/preview_irregular_clutter.png` | 地图闭环关键帧 |
| `.../houseexpo/preview_a64a310dfa11.png` | HouseExpo 室内跟随关键帧 |
| `.../houseexpo/summary.json` | HouseExpo 三套户型闭环汇总 |
| `.../houseexpo/*.mp4` | HouseExpo 室内跟随视频 |
| `.../stage_follow/summary.json` | ReplicaCAD 40 s 复杂环路结果 |
| `.../stage_follow/stage_follow.mp4` | 真实公寓复杂环路视频 |
| `.../closed_loop/weave300_summary.json` | 闭环 weave 300 帧 |
| `.../closed_loop/crossing300_summary.json` | 闭环 crossing 300 帧 |
