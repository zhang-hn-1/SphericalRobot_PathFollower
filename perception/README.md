# Perception layer: person detection

Status: detector/tracking scaffolding plus a **2026-11-02 world-frame target
layer** (identity lock, distractor rejection, 1 s prediction; see below). The
original design and evidence are in
[`../artifacts/perception/PERCEPTION_PLAN_ZH.md`](../artifacts/perception/PERCEPTION_PLAN_ZH.md).

## What is here

| File | Purpose |
|---|---|
| `camera_rig.py` | frozen rig geometry: offsets, optics, frame mapping, projection |
| `rig_upright.py` | rotate rig frames so people stand up; box/mask mapping |
| `preprocess.py` | self-occlusion mask, IMU roll correction, low-light enhancement |
| `person_perception.py` | run a COCO person detector offline, emit JSON/JSONL, benchmark latency |
| `sim/people_scene.py` | Isaac Gym multi-person scene: stereo RGB, depth, instance seg, GT boxes/IDs/target |
| `detect_sim_dataset.py` | run a trained detector over a generated dataset |
| `detect_rig.py` | detect on rig frames (rotate upright first, map boxes back) |
| `compare_upright.py` | measure the effect of the upright rotation on recall |
| `target_tracker.py` | single-target tracking on multi-person detections (image space) |
| `eval_target_tracking.py` | target-tracking metrics on generated sequences |
| `target_perception.py` | world-frame target layer: identity anchor, distractor rejection, 1 s prediction |
| `range_bearing.py` | upright front-pair box + depth/height -> metric ground point |
| `eval_target_perception.py` | synthetic crossing/dropout/re-ID benchmark |
| `eval_reusable_scenarios.py` | run the layer over existing crossing / avatar / front-pair datasets |
| `sim/occlusion_perception.py` | Isaac Gym + YOLO rendered 1 s disappearance experiment |
| `sim/map_perception.py` | polygon map scenes: wall occlusion, U-turn, re-acquisition |
| `sim/map_follow.py` | closed-loop map following: target A* route + robot A* stand-off |
| `sim/stage_follow.py` | ReplicaCAD apartment + refrigerator: 40 s complex loop, multiple occlusions, wall clearance; optional `--policy` |
| `research/predictive_follow_env.py` | fast predictive-follow environment: stage visibility, detection dropout, 15D obs / 1D action |
| `research/train_predictive_follow.py` | rule behavior cloning + PPO; `--residual` is the residual-RL variant |
| `research/train_predictive_follow_sac.py` | SAC baseline on the same env/predictor/eval protocol |
| `research/train_teacher_student.py` | privileged teacher PPO + DAgger student distillation |
| `research/crowd_follow_env.py` | multi-dynamic-pedestrian 2D benchmark |
| `research/train_crowd_attention.py` | CrowdNav-lite attention + predictive safety benchmark |
| `research/export_crowd_benchmark.py` | export deterministic scenarios, replays and 95% CI statistics |
| `research/calibrate_conformal.py` | split-conformal prediction-radius calibration |
| `research/train_multimodal_predictor.py` | GRU + K-mode MDN (Trajectron++-lite) |
| `candidate_scorer.py` | 8-feature linear candidate scorer (VideoJev-lite) |
| `research/train_candidate_scorer.py` | train the candidate scorer |
| `research/benchmark_following.py` | fixed/random-route multi-policy benchmark with standard metrics and latency |
| `research/eval_candidate_scorer.py` | with/without scorer stress comparison |
| `learned_predictor.py` | supervised future-position predictor (0.5/1/2 s displacement) |
| `research/train_trajectory_predictor.py` | train/validate the trajectory predictor; `--no-absolute` is the route-blind ablation |
| `research/eval_trajectory_predictor.py` | score a predictor on the training route and unseen routes |
| `stage_map.py` | ReplicaCAD wall occupancy + 2D line-of-sight without Isaac Gym |
| `tools/houseexpo_to_maps.py` | convert MIT HouseExpo floorplans into local map JSON |
| `test_target_perception.py` | CPU unit tests for the world-frame target layer and geometry |
| `build_crops.py` | person-crop dataset for appearance models |
| `train_appearance.py` | ResNet-18/34 identity classifier (embedding for association) |
| `appearance.py` | HSV vs ResNet appearance descriptors |
| `eval_appearance.py` | appearance retrieval metrics (target vs distractors) |
| `make_self_occlusion_mask.py` | fit the 46% self-occlusion mask for each eye |
| `person_det.yaml`, `person_seg.yaml` | Ultralytics dataset templates |
| `test_*.py` | CPU-only checks (preprocess, camera rig, rig transform) |

## Target layer (2026-11-02): identity lock + 1 s prediction

`target_perception.TargetPerceptionLayer` keeps all people as tracks, anchors the
operator-selected target with an immutable appearance vector, rejects ambiguous
crossings, and runs a constant-velocity Kalman filter in world coordinates.
A missing target is reported as `predicted` for up to 1.0 s (position,
covariance and a 1 s trajectory), then as `lost`; the track is retained for
re-acquisition/re-identification.

```bash
# CPU unit tests
python3 -m unittest perception.test_target_perception -v

# synthetic crossing / dropout / re-ID benchmark
python3 perception/eval_target_perception.py \
    --out artifacts/perception/target_perception_20261102/synthetic_benchmark.json \
    --seeds 10

# reusable existing datasets: 4 crossing variants, 24 avatars, front-pair clutter
/home/jason/.venvs/ultralytics-rice/bin/python perception/eval_reusable_scenarios.py \
    --out artifacts/perception/target_perception_20261102/reusable_scenarios.json

# polygon map scenes (walls, wall-hidden target, U-turn)
/home/jason/legged_gym/.venv/bin/python perception/sim/map_perception.py \
    --out /home/jason/perception_data/runs/map_perception_v3 \
    --scenario all --frames 70 --video

# closed loop: target walks an A* route through the map; ball follows it
/home/jason/legged_gym/.venv/bin/python perception/sim/map_follow.py \
    --out /home/jason/perception_data/runs/map_follow_v1 \
    --map irregular_clutter --frames 320 --video

# realistic ReplicaCAD apartment: 40 s complex loop + predictive lost-follow
/home/jason/legged_gym/.venv/bin/python perception/sim/stage_follow.py \
    --out /home/jason/perception_data/runs/stage_predictive_v1 \
    --frames 400 --video

# four-view recording: front-left / front-right / top-down / telemetry
/home/jason/legged_gym/.venv/bin/python perception/sim/stage_follow.py \
    --out /home/jason/perception_data/runs/video_teacher_student_600 \
    --frames 600 --video --multiview \
    --policy /home/jason/perception_data/runs/teacher_student_ppo_v1/actor.pt \
    --predictor /home/jason/perception_data/runs/trajectory_predictor_v2/predictor.pt

# supervised future-position predictor (held-out route seeds)
/home/jason/.venvs/ultralytics-rice/bin/python \
    perception/research/train_trajectory_predictor.py \
    --out /home/jason/perception_data/runs/trajectory_predictor_v2 \
    --epochs 30 --train-trajectories 300 --val-trajectories 60 \
    --samples-per-trajectory 100 --duration 20

# stage-like BC + PPO actor on top of the frozen predictor
/home/jason/.venvs/ultralytics-rice/bin/python \
    perception/research/train_predictive_follow.py \
    --out /home/jason/perception_data/runs/predictive_follow_ppo_stage_v9 \
    --iters 120 --episodes 8 --episode-s 30 --seed 7 \
    --bc-episodes 40 --bc-steps 1500 \
    --predictor /home/jason/perception_data/runs/trajectory_predictor_v2/predictor.pt \
    --stage-visibility --detection-dropout 0.30 --measurement-noise 0.05

# rule baseline with the predictor, and supervised+RL with the safety shield
/home/jason/legged_gym/.venv/bin/python perception/sim/stage_follow.py \
    --out /home/jason/perception_data/runs/stage_rule_predictor_v2_600 \
    --frames 600 --video \
    --predictor /home/jason/perception_data/runs/trajectory_predictor_v2/predictor.pt

/home/jason/legged_gym/.venv/bin/python perception/sim/stage_follow.py \
    --out /home/jason/perception_data/runs/stage_ppo_stage_v9guard_600 \
    --frames 600 --video \
    --predictor /home/jason/perception_data/runs/trajectory_predictor_v2/predictor.pt \
    --policy /home/jason/perception_data/runs/predictive_follow_ppo_stage_v9/actor.pt

# route-blind ablation: train, evaluate on seen/unseen routes, run in the stage
/home/jason/.venvs/ultralytics-rice/bin/python \
    perception/research/train_trajectory_predictor.py \
    --out /home/jason/perception_data/runs/trajectory_predictor_relative_v1 \
    --epochs 30 --train-trajectories 300 --val-trajectories 60 \
    --samples-per-trajectory 100 --duration 20 --no-absolute

/home/jason/.venvs/ultralytics-rice/bin/python \
    perception/research/eval_trajectory_predictor.py \
    --checkpoint /home/jason/perception_data/runs/trajectory_predictor_relative_v1/predictor.pt \
    --out /home/jason/perception_data/runs/eval_predictor_relative_v1.json

/home/jason/legged_gym/.venv/bin/python perception/sim/stage_follow.py \
    --out /home/jason/perception_data/runs/stage_relative_policy_600 \
    --frames 600 --video \
    --predictor /home/jason/perception_data/runs/trajectory_predictor_relative_v1/predictor.pt \
    --policy /home/jason/perception_data/runs/predictive_follow_ppo_stage_relative_v1/actor.pt

# alternate RL baselines (SAC and residual PPO; not selected)
/home/jason/.venvs/ultralytics-rice/bin/python \
    perception/research/train_predictive_follow_sac.py \
    --out /home/jason/perception_data/runs/predictive_follow_sac_v2 \
    --steps 30000 --warmup 10000 --batch 256 --lr 1e-4 \
    --predictor /home/jason/perception_data/runs/trajectory_predictor_v2/predictor.pt \
    --stage-visibility --detection-dropout 0.30 --measurement-noise 0.05

/home/jason/legged_gym/.venv/bin/python perception/sim/stage_follow.py \
    --out /home/jason/perception_data/runs/stage_residual_abs_seed7_600 \
    --frames 600 --seed 7 --video \
    --predictor /home/jason/perception_data/runs/trajectory_predictor_v2/predictor.pt \
    --policy /home/jason/perception_data/runs/predictive_follow_ppo_residual_v1/actor.pt \
    --residual --residual-scale 0.4

# teacher-student: privileged teacher, DAgger student, PPO fine-tune
/home/jason/.venvs/ultralytics-rice/bin/python \
    perception/research/train_predictive_follow.py \
    --out /home/jason/perception_data/runs/teacher_privileged_ppo_v1 \
    --iters 120 --episodes 8 --episode-s 30 --seed 7 --privileged

/home/jason/.venvs/ultralytics-rice/bin/python \
    perception/research/train_teacher_student.py \
    --teacher /home/jason/perception_data/runs/teacher_privileged_ppo_v1/actor.pt \
    --predictor /home/jason/perception_data/runs/trajectory_predictor_v2/predictor.pt \
    --out /home/jason/perception_data/runs/teacher_student_v1 \
    --dagger-iters 2 --episodes 10 --bc-steps 1500 \
    --detection-dropout 0.30 --measurement-noise 0.05

/home/jason/.venvs/ultralytics-rice/bin/python \
    perception/research/train_predictive_follow.py \
    --out /home/jason/perception_data/runs/teacher_student_ppo_v1 \
    --iters 120 --episodes 8 --episode-s 30 --seed 7 \
    --init-actor /home/jason/perception_data/runs/teacher_student_v1/student.pt \
    --predictor /home/jason/perception_data/runs/trajectory_predictor_v2/predictor.pt \
    --stage-visibility --detection-dropout 0.30 --measurement-noise 0.05

# multi-route benchmark: fixed vs randomized routes/occluders, standard metrics
/home/jason/.venvs/ultralytics-rice/bin/python \
    perception/research/benchmark_following.py \
    --out /home/jason/perception_data/runs/benchmark_random_50_v2.json \
    --randomize --episodes 50 --episode-s 30 --seed 1000 \
    --absolute-predictor /home/jason/perception_data/runs/trajectory_predictor_v2/predictor.pt \
    --relative-predictor /home/jason/perception_data/runs/trajectory_predictor_relative_v1/predictor.pt \
    --ppo /home/jason/perception_data/runs/predictive_follow_ppo_stage_v9/actor.pt \
    --teacher /home/jason/perception_data/runs/teacher_student_ppo_v1/actor.pt \
    --sac /home/jason/perception_data/runs/predictive_follow_sac_v2/actor.pt \
    --residual /home/jason/perception_data/runs/predictive_follow_ppo_residual_v1/actor.pt \
    --relative-ppo /home/jason/perception_data/runs/predictive_follow_ppo_stage_relative_v1/actor.pt

# maze-like HouseExpo floorplan: target walks an A* route, P0 follows
/home/jason/legged_gym/.venv/bin/python perception/sim/map_follow.py \
    --out /home/jason/perception_data/runs/maze_houseexpo_450 \
    --map-json perception/assets/maps/houseexpo/houseexpo_a64a310dfa11.json \
    --scene-id houseexpo_a64a310dfa11 --map-name houseexpo_maze \
    --frames 450 --plan-inflation 0.42 --replan-inflation 0.40 --video

# three-obstacle ReplicaCAD scene (teacher-student, 600 frames)
/home/jason/legged_gym/.venv/bin/python perception/sim/stage_follow.py \
    --out /home/jason/perception_data/runs/video_multi_obstacle_600 \
    --frames 600 --multi-obstacle --video --multiview \
    --policy /home/jason/perception_data/runs/teacher_student_ppo_v1/actor.pt \
    --predictor /home/jason/perception_data/runs/trajectory_predictor_v2/predictor.pt

# open-source HouseExpo indoor floorplans (MIT, 1.35x scaled for the ball)
/home/jason/legged_gym/.venv/bin/python perception/sim/map_follow.py \
    --out /home/jason/perception_data/runs/houseexpo_follow_v1 \
    --map-json perception/assets/maps/houseexpo/houseexpo_a64a310dfa11.json \
    --scene-id houseexpo_a64a310dfa11 \
    --plan-inflation 0.42 --replan-inflation 0.40 \
    --frames 220 --video

# rendered Isaac Gym + YOLO check: target disappears for 1 s
/home/jason/legged_gym/.venv/bin/python perception/sim/occlusion_perception.py \
    --out /home/jason/perception_data/runs/occlusion_perception_depth_v3 \
    --episodes 2 --frames 40 --use-depth --video

# closed loop through the existing P0 follow controller
/home/jason/legged_gym/.venv/bin/python perception/sim/follow_p0.py \
    --out /home/jason/perception_data/runs/follow_p0_target_layer_crossing300 \
    --episodes 1 --frames 300 --target-layer --use-depth --scenario crossing
```

The reusable suite uses `l1_crossing_v2` (including the similar-appearance
variant and three depth gaps), `sim_v4_avatar_dev/test` (24 Rocketbox avatars,
6 environments) and `l1_front_pair_v1` / `l1_appearance_v1` (four-person
clutter). Crossing variants use cached YOLO detections; the other datasets use
GT boxes + GT metric positions to isolate tracker behaviour.

Measured: identity 100% through a similar-appearance distractor in the rendered
test, 100% of the 1 s disappearance frames stayed in `predicted`, and the 1 s
position error was 0.155 m (depth) / 0.165 m (monocular). Reusable-scene
identity is 98-100% (only `crossing/gap100` drops to 98% from monocular-range
bias; its GT-position diagnostic is 100%). Three polygon map scenes
(`perception/assets/maps/rl_validation.json`, `rl_test.json`) also keep identity
at 100% while the target walks behind walls and returns. The closed-loop map
follower (`sim/map_follow.py`) follows the perceived target through the same
maps at 2.5-3.1 m distance with 0.63 m minimum wall clearance and zero
geometric collisions. Open-source **HouseExpo** floorplans (MIT, 12 selected
apartments, 1.35x scaled for the 0.8 m ball) are in
`assets/maps/houseexpo/`; three closed-loop runs also keep identity at 100%
with zero geometric collisions. A real ReplicaCAD apartment + refrigerator
scene (`assets/scenes/`, CC BY-NC 4.0) runs a 40 s, ~9 m rectangular loop.
With velocity confidence and a predictive lost-follow rule, visible frames rise
from 213 to 325 and the longest occlusion drops from 8.7 s to 3.2 s while
identity stays 100% and geometric collisions stay at zero (minimum clearance
0.559 m).

The follow layer is now **supervised prediction + RL with a safety shield**:
a GRU predictor maps the target history to 0.5/1/2 s displacement (1 s MAE
0.068 m vs 0.173 m for smoothed constant velocity on held-out route seeds), and
a BC+PPO actor chooses the lateral speed from that prediction. The current best
variant is **teacher-student / DAgger**: a privileged teacher PPO sees the clean
target state, and the student sees only measured history plus the frozen
predictor. On the ReplicaCAD stage, 600-frame runs across five seeds give
**563 vs 437 visible frames, 36 vs 162 hidden frames, and 1.7 s vs 11.0 s
longest occlusion** (mean; identity 99.97%, zero collisions, minimum clearance
0.559 m). A 50-episode randomized-route benchmark shows the learned policies
degrade out of distribution (teacher 0.565 vs rule 0.574 visible; PPO 0.548),
so cross-route generalization is still open. A randomized-trained
teacher-student recovers to **0.591 visible / 0.90 success** vs rule
0.574/0.84, but is not yet the fixed-route best. A preliminary camera
domain-shift test (4 people, gain 0.7, noise 0.02) collapses YOLO measured rate
to 0.6%; detection-side domain randomization is the next bottleneck. End-to-end latency is about
123 ms per frame, dominated by the 109 ms YOLO pass; tracking, predictor and
policy together are under 1.5 ms. The predictor intentionally uses absolute world position as a route
prior, so cross-map generalization is not claimed.

Ablation (same data/architecture, absolute position removed): the route-aware
model's 1 s error rises from 7.1 cm on the training route to **27.7 cm** on an
unseen rotated route; the route-blind model goes from 14.3 cm to **19.7 cm**
(smoothed constant velocity: 18.1 / 16.8 cm). So the 7.1 cm result is partly
route memory, not a general dynamics model. In the stage, route-blind + RL is
highly seed-dependent (580/20/1.0 s, 390/209/8.9 s, 529/71/4.8 s visible /
hidden / longest), so it does not replace the route-aware policy yet.

Two further RL baselines were tested and are **not selected**: residual PPO
(rule action + learned correction) reached 63.2% vs 63.0% in the 2D env but
degraded in the stage (393/121/335 visible across three seeds, identity
94–99%); SAC was unstable without tuning and reached 58.1% visible in 2D.
The teacher-student follow-up is implemented (report section 3.13); the
conformal speed-cap and 3-mode MDN prototypes are documented in section 3.14.
The conformal cap was too conservative (503/97 vs 563/37) and the MDN's 1 s
minADE was 0.372 m vs 0.071 m for the single-mode predictor, so neither is
selected. A VideoJev-lite candidate scorer (8 features, logistic regression,
96.4% candidate accuracy) gives only +0.025 pp identity on the hardest noisy
similar-crossing stress case and no change elsewhere; it is an optional
`candidate_scorer=` hook, not the default. The remaining recommendations are
TD3+BC/AWAC offline-to-online and CrowdNav-style attention / model-based RL. See
[`../artifacts/perception/TARGET_PERCEPTION_20261102.md`](../artifacts/perception/TARGET_PERCEPTION_20261102.md).

## Unified 2D crowd benchmark

`research/crowd_follow_2d_env.py` adds a 2D-action benchmark with 3 moving
pedestrians and random static occluders. ORCA is an exact Python-RVO2 baseline;
the first 2D attention policy is not yet competitive (0.00 success vs ORCA 0.10
over 50 episodes), so it is documented as a failed training run, not a result.

## Official social-navigation baselines

On the native CrowdNav circle-crossing task, the official ORCA/Python-RVO2
baseline reached 0.40 success / 0.60 collision over 20 test episodes; the
official CrowdNav code (SARL attention, short 120-episode training) reached
0.60 / 0.40. These are goal-reaching results, not the 1D target-following
benchmark, and must not be compared directly. SARL (RelationalGraphLearning)
source was downloaded but lacks a runnable config/license, so it is not claimed
as reproduced. See report section 3.21.

## Dynamic crowd benchmark

`research/crowd_follow_env.py` adds 3 moving pedestrians, random routes and
static occluders. Over 50 episodes, CrowdNav-lite attention improved success
from 0.28 (rule) to 0.54 and reduced pedestrian collisions from 36 to 19.
Adding a 0.35 s predictive safety filter reached **0.80 success / 5
pedestrian collisions** (50 episodes), at the cost of a larger 3.1 m
following distance. A 100-episode deterministic replay set and 95% CI
statistics are in
`artifacts/perception/target_perception_20261102/crowd_dynamic/public_replay_100/`
(attention-safety 0.71 [0.62,0.80], 17 pedestrian collisions vs rule 0.21 [0.13,0.29], 66).
This is a 2D benchmark, not a full visual/identity crowd system. The attention
policy is a CrowdNav-lite reimplementation, not the official CrowdNav code;
ORCA/SARL exact baselines are still blocked by missing packages/network.

## Latest maze result

`map_follow` with the HouseExpo `houseexpo_a64a310dfa11` floorplan follows a
17.56 m target route with **identity 100%, measured 90.1%, zero collisions,
final progress 1.0** in 443 frames. A self-built 4x4 narrow maze is kept as a
failure analysis: the P0 base cannot clear the tight corners; the perception
layer is not the limiting factor.

## Rocketbox avatars (optional, replaces the coloured primitives)

`perception/assets/rocketbox/` holds 115 MIT-licensed, textured, human-proportioned
avatars baked from Microsoft Rocketbox into one OBJ + URDF each. Enable them with:

```bash
python -m perception.sim.people_scene --out /tmp/people_avatars \
    --avatar-dir perception/assets/rocketbox --avatars 12 --save-upright
```

The scene loads `--avatars` of them (seeded, `0` = all 115), gives every person a
random one per episode, and keeps their real height (1.43-1.87 m) in
`Person.height` and in `gt/tracks.jsonl`. Data prep and the Blender pipeline live
in `sim/avatar/`; `check_avatars.py` and `test_people_scene_avatars.py` run
without Isaac Gym.

## The rig lies down: rotate before detecting

The frozen rig's optical axis points forward, but its image axes are rotated 90
degrees relative to the body, so a standing person appears **lying down**.
That breaks detectors pretrained on upright people.

Measured on COCO samples with `yolo26n-seg`:

| frame | persons |
|---|---|
| upright original | 4 (0.91 / 0.87 / 0.87 / 0.41) |
| rotated so the person lies | 1 (0.31) |
| lying -> rotated upright | 4 (identical confidences) |

The rotation is lossless. Use it **for both training and inference**; a model
trained in one space collapses in the other (measured: 3.1% / 21.4%).

```bash
# generate upright training data
/home/jason/legged_gym/.venv/bin/python -m perception.sim.people_scene \
    --out /home/jason/perception_data/sim_up --save-upright \
    --occlusion-mask /home/jason/perception_data/masks/rig.png \
    --num-envs 8 --people 5 --frames 40 --episodes 6

# detect on raw rig frames (rotates internally, boxes come back in rig coords)
/home/jason/.venvs/ultralytics-rice/bin/python perception/detect_rig.py \
    --weights /path/to/best.pt \
    --images /home/jason/perception_data/sim_rig_test/all.txt \
    --out /home/jason/perception_data/sim_rig_test/det.jsonl
```

## Quick start: the sim perception loop

```bash
# 1. generate a multi-person dataset (Isaac Gym venv)
source local_env.sh
/home/jason/legged_gym/.venv/bin/python -m perception.sim.people_scene \
    --out /home/jason/perception_data/sim_v2 \
    --num-envs 8 --people 5 --frames 40 --episodes 6

# 2. train the detector from scratch (Ultralytics venv, no internet needed)
/home/jason/.venvs/ultralytics-rice/bin/yolo detect train \
    model=yolov8n.yaml data=/home/jason/perception_data/sim_v2/data.yaml \
    epochs=40 imgsz=640 batch=16 workers=0 device=0 \
    project=/home/jason/perception_data/runs name=proxy_v1 exist_ok=True

# 3. export detections
/home/jason/.venvs/ultralytics-rice/bin/python perception/detect_sim_dataset.py \
    --weights /home/jason/perception_data/runs/proxy_v1/weights/best.pt \
    --images /home/jason/perception_data/sim_v2/val.txt \
    --out /home/jason/perception_data/sim_v2/detections_val.jsonl --conf 0.05

# 4. evaluate target tracking
python3 perception/eval_target_tracking.py \
    --dataset /home/jason/perception_data/sim_v2 --split val \
    --detections /home/jason/perception_data/sim_v2/detections_val.jsonl \
    --out /home/jason/perception_data/sim_v2/track_det.json
```

The first result is in
[`../artifacts/perception/SIM_TARGET_TRACKING_20261020.md`](../artifacts/perception/SIM_TARGET_TRACKING_20261020.md).

## Current best (2026-10-20)

- Detector: **YOLOv8n, 150 epochs** on `sim_v2` (mAP50 0.952, mAP50-95 0.883).
- Appearance: **ResNet-18** identity classifier embedding (64x128 crops).
- Tracker: default gates (area 0.40–2.2, centre 1.20).
- Independent test set (`seed=100`, 48 episodes, 1920 frames):
  **identity 89.4%, centre hit 90.7%, visible-IoU success 88.4%**.
- Remaining bottleneck: targets farther than 6 m (identity 84% at 10 m+ vs 100% at 2–6 m).

YOLOv8s has higher detection mAP but did not beat YOLOv8n on tracking; see the
results document for the full comparison.

## Why there is no *real-data* training yet

The repository has no images of people, and the existing camera frames are not
usable as training input: 89.9% of pixels are below 30/255 and 46.2% of each
eye is self-occluded. Real-data training still requires fixing the imaging path
and collecting labelled frames (see the plan, sections 3.4 and 5).

What *is* available offline is the sim loop above. It trains on **proxy
actors** (coloured geometric humanoids), so it validates the pipeline and the
target-tracking logic, not real-person recognition. Swapping in human meshes is
the next step; the only function that needs to change is
`PeopleScene._build_assets`.

The machine also has cached COCO segmentation weights, so the zero-shot
baseline in `person_perception.py` works without network access.

## Run the baseline

Use the `ultralytics-rice` venv (the main `legged_gym` venv has torch 1.10 and
cannot import ultralytics):

```bash
/home/jason/.venvs/ultralytics-rice/bin/python perception/person_perception.py \
    --source /path/to/frames \
    --out artifacts/perception/baseline.jsonl \
    --imgsz 640 --conf 0.35 --device 0 --benchmark
```

With the Rotunbot preprocessing chain:

```bash
/home/jason/.venvs/ultralytics-rice/bin/python perception/person_perception.py \
    --source /path/to/frames \
    --mask /path/to/self_occlusion_mask.png \
    --roll 12.5 --gamma 0.7 --clahe 3.0 \
    --out artifacts/perception/baseline.jsonl
```

`--roll` is the camera roll in degrees, normally supplied by the IMU. The mask
convention is **non-zero = visible**, zero = self-occluded.

## Output

Each detection follows the shape already consumed by
`home_robot/.../person_follow_node.py` (`class`, `confidence`, `bbox`) and adds
`bbox_raw`, `height_px`, `range_m` (only with `--focal-px`) and `has_mask`.
The wrapper `summary` records model, image size, preprocessing and frame counts
so two runs can be compared directly.

## Unit checks

```bash
python3 perception/test_preprocess.py
```

## Baseline numbers

See [`../artifacts/perception/BASELINE_20261020.md`](../artifacts/perception/BASELINE_20261020.md).
Short version: `yolo26n-seg` runs at 6.2 ms on the RTX 4070 and detects 4/4 people
in the bundled sample, but the ball-camera frames contain no people, so **no
accuracy claim can be made yet**.

## Fine-tuning (once labelled data exists)

Detection weights:

```bash
/home/jason/.venvs/ultralytics-rice/bin/yolo detect train \
    model=/path/to/yolo26n.pt data=perception/person_det.yaml \
    imgsz=640 epochs=100 batch=16 device=0
```

The cached weights on this machine are **segmentation** models, which expect
polygon labels; use `perception/person_seg.yaml` with them.  Instance masks are
also what makes the stereo range estimate robust (background inside the box is
excluded), so annotating masks is not wasted work.

Do not train until the imaging path is fixed: 89.9% of pixels below 30/255 and
46.2% self-occlusion are properties of the input, not of the model.

