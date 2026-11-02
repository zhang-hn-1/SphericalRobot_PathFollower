#!/usr/bin/env python3
"""Closed-loop person following under the September physical base (P0).

The robot pose is advanced by the identified P0 model at 50 Hz — command rate
limits (0.08 / 0.02 per 20 ms), low-pass (0.50 / 0.40), first-order speed lag
(tau 1.08 s fitted from the spin-up trace), coast deceleration 0.30*v, and the
identified curvature grid (cannot turn in place; steering reaches full deflection
in ~0.52 s).  At 10 Hz the front-left camera renders, the detection worker
separates perception, the B0 tracker keeps identity, and a follow controller
commands (drive, steer) strictly inside the identified envelope.

This is an engineering loop: the pose model is the *identified steady-state*
P0, not the full PhysX environment (that gate is G0).  Claims here are about
whether the P0 constraints still allow following, not about final numbers.

    /home/jason/legged_gym/.venv/bin/python perception/sim/follow_p0.py \
        --episodes 2 --frames 450 --kp 0.6 --kd 0.8 \
        --out /home/jason/perception_data/runs/follow_p0_v1
"""

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))

def iou(a, b):
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


from l1_front_pair import (FRONT_CAM_FORWARD, FRONT_HALF_BASELINE, HEIGHT,
                           HORIZONTAL_FOV_DEG, TOE_OUT_DEG, WIDTH,
                           camera_basis, front_pair_config,
                           person_full_bbox)  # noqa: E402
from people_scene import PeopleScene, _bbox_from_mask  # noqa: E402
from range_bearing import FrontPairCamera  # noqa: E402
from target_perception import PersonDetection, TargetPerceptionLayer  # noqa: E402
from target_tracker import (Detection, TargetState, TargetTracker,
                            hsv_histogram)  # noqa: E402

FX = (WIDTH / 2.0) / math.tan(math.radians(HORIZONTAL_FOV_DEG) / 2.0)
CX, CY = WIDTH / 2.0 - 0.5, HEIGHT / 2.0 - 0.5


def _load_policy(ckpt):
    global torch
    import torch
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(48, 64), torch.nn.ELU(),
        torch.nn.Linear(64, 64), torch.nn.ELU(),
        torch.nn.Linear(64, 2), torch.nn.Tanh())
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict({k.replace("net.", "", 1): v
                           for k, v in state.items()})
    model.eval()
    return model


def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# --------------------------------------------------------------- P0 model

def load_curvature_grid():
    path = REPO / "artifacts/path_follower/system_identification/open_loop_grid/raw.csv"
    cells = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            key = (float(row["action_joint1"]), float(row["action_joint2"]))
            cells.setdefault(key, []).append(
                float(row["trajectory_curvature_inv_m"]))
    a1s = sorted({k[0] for k in cells})
    a2s = sorted({k[1] for k in cells})
    grid = np.array([[np.mean(cells[(a1, a2)]) for a2 in a2s] for a1 in a1s])
    return a1s, a2s, grid


A1S, A2S, GRID = load_curvature_grid()


def curvature(a1, a2):
    """Bilinear lookup; |a1| < 0.4 is unidentified, assume linear to 0."""
    a1c = float(np.clip(a1, A1S[0], A1S[-1]))
    a2c = float(np.clip(a2, A2S[0], A2S[-1]))
    if abs(a1) < A1S[2]:  # below the smallest identified drive magnitude
        scale = abs(a1) / A1S[2]
    else:
        i1 = np.searchsorted(A1S, a1c) if a1c > 0 else np.searchsorted(A1S, a1c)
        # piecewise-linear in a1 on the identified side
        lo = max([x for x in A1S if x <= a1c]) if a1c > 0 else min(
            [x for x in A1S if x >= a1c])
        hi = min([x for x in A1S if x >= a1c]) if a1c > 0 else max(
            [x for x in A1S if x <= a1c])
        if hi == lo:
            scale = 1.0
            row = GRID[A1S.index(lo)]
            return float(np.interp(a2c, A2S, row)) * (scale if abs(a1) < A1S[2] else 1.0)
        r_lo, r_hi = GRID[A1S.index(lo)], GRID[A1S.index(hi)]
        w = (a1c - lo) / (hi - lo)
        row = r_lo * (1 - w) + r_hi * w
        return float(np.interp(a2c, A2S, row))
    i1 = min(range(len(A1S)), key=lambda i: abs(A1S[i] - A1S[2]))
    row = GRID[i1]
    return float(np.interp(a2c, A2S, row)) * scale


def max_curvature(a1):
    return max(abs(curvature(a1, a2)) for a2 in A2S)


def invert_steer(a1, kappa_des):
    """Find a2 with curvature(a1, a2) = kappa_des (profile is monotone in a2)."""
    row = [curvature(a1, a2) for a2 in A2S]
    return float(np.interp(kappa_des, row[::-1], A2S[::-1]))


class P0Robot:
    """50 Hz identified-model robot: rate limit, low-pass, speed lag, curvature."""

    DT = 0.02
    RATE1, RATE2 = 0.08, 0.02
    ALPHA1, ALPHA2 = 0.50, 0.40
    KV = 0.397          # (m/s) per rad/s of wheel speed; wheel = 3 * a1
    TAU_UP = 1.08       # fitted from the full-drive spin-up trace
    COAST_K = 0.30      # coast deceleration ~ 0.30 * v

    def __init__(self, xy=(0.0, 0.0), yaw=0.0):
        self.x, self.y, self.yaw = xy[0], xy[1], yaw
        self.v = 0.0
        self.a1_f = self.a2_f = 0.0

    def tick(self, a1_cmd, a2_cmd):
        a1_cmd = float(np.clip(a1_cmd, -1.0, 1.0))
        a2_cmd = float(np.clip(a2_cmd, -0.9, 0.9))
        a1_rl = np.clip(a1_cmd, self.a1_f - self.RATE1, self.a1_f + self.RATE1)
        a2_rl = np.clip(a2_cmd, self.a2_f - self.RATE2, self.a2_f + self.RATE2)
        self.a1_f += self.ALPHA1 * (a1_rl - self.a1_f)
        self.a2_f += self.ALPHA2 * (a2_rl - self.a2_f)
        v_target = self.KV * 3.0 * self.a1_f
        if v_target > self.v:
            self.v += (v_target - self.v) * (self.DT / self.TAU_UP)
        else:
            self.v = max(v_target, self.v - self.COAST_K * self.v * self.DT)
        kappa = curvature(self.a1_f, self.a2_f)
        self.yaw += self.v * kappa * self.DT
        self.x += self.v * math.cos(self.yaw) * self.DT
        self.y += self.v * math.sin(self.yaw) * self.DT


# ------------------------------------------------------------- perception

class DetectionWorker:
    """YOLO in the ultralytics venv, one JSON line per frame over stdin."""

    def __init__(self, weights, conf=0.05, imgsz=960, boxmot="",
                 embedder=""):
        self.tmp = Path(tempfile.mkdtemp(prefix="follow_p0_"))
        self.proc = subprocess.Popen(
            ["/home/jason/.venvs/ultralytics-rice/bin/python",
             str(REPO / "perception/sop/detection_worker.py"),
             "--weights", str(weights), "--conf", str(conf),
             "--imgsz", str(imgsz),
             "--boxmot", str(boxmot), "--embedder", str(embedder)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        assert self.proc.stdout.readline().strip() == json.dumps(dict(ready=True))
        self.frame_id = 0

    def detect(self, image):
        path = self.tmp / f"f{self.frame_id:06d}.png"
        self.frame_id += 1
        Image.fromarray(np.asarray(image)).save(path)
        self.proc.stdin.write(json.dumps(dict(image=str(path))) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        payload = json.loads(line)
        if "tracks" in payload:
            return payload["tracks"]
        return payload.get("detections", [])

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=30)


# ----------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights",
                        default="/home/jason/perception_data/runs/"
                                "l1_front_pair_v1/weights/best.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--frames", type=int, default=450)
    parser.add_argument("--kp", type=float, default=0.6)
    parser.add_argument("--kd", type=float, default=0.8)
    parser.add_argument("--cruise", type=float, default=0.65,
                        help="drive action while following (0.65 = 0.77 m/s)")
    parser.add_argument("--far-drive", type=float, default=0.9)
    parser.add_argument("--target-speed", type=float, default=0.4)
    parser.add_argument("--roll-amp", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--avatars", type=int, default=8)
    parser.add_argument("--num-walls", type=int, default=0,
                        help="0 keeps the spawn corridor clear; walls were "
                             "found to occlude the target at spawn otherwise")
    parser.add_argument("--num-pillars", type=int, default=0)
    parser.add_argument("--scenario", choices=["weave", "stop", "fov_exit",
                                                "crossing"], default="weave")
    parser.add_argument("--stop-t", type=float, default=12.0,
                        help="stop scenario: the target halts at this time")
    parser.add_argument("--search", choices=["none", "last_bearing", "blind",
                                             "predictive", "reason_cond"], default="none",
                        help="lost-target search policy (Tier-2 baseline set)")
    parser.add_argument("--search-budget", type=float, default=25.0,
                        help="seconds of searching before giving up (STOP)")
    parser.add_argument("--target-layer", action="store_true",
                        help="use the world-frame target perception layer "
                             "(identity lock + 1 s motion prediction) instead "
                             "of the image-space B0 tracker")
    parser.add_argument("--use-depth", action="store_true",
                        help="with --target-layer, use camera depth in "
                             "addition to the monocular box-height prior")
    parser.add_argument("--tracker", default="",
                        help="worker-side boxmot tracker (e.g. occluboost): "
                             "ID-lock protocol instead of the B0 tracker")
    parser.add_argument("--relock", action="store_true",
                        help="hybrid: when the locked track dies, re-lock onto "
                             "the active track whose HSV best matches the "
                             "target template (hysteresis 1 s)")
    parser.add_argument("--policy-ckpt", default="",
                        help="T-A behavioural-cloning policy: replaces the "
                             "analytic controller in-loop")
    parser.add_argument("--confirm-tau", type=float, default=0.0,
                        help="seconds of continuous acceptance required before "
                             "committing FOLLOW after a re-acquisition (delta-tau)")
    parser.add_argument("--exit-side", choices=["left", "right"], default="left",
                        help="fov_exit: which side the target leaves through")
    parser.add_argument("--save-crops", default="",
                        help="save every detection crop during SEARCH/CONFIRM "
                             "(+1 s) into this directory with a manifest, for "
                             "training the identity verifier")
    parser.add_argument("--similar-distractor", action="store_true",
                        help="force target and distractor to be Business "
                             "avatars (similar appearance) in fov_exit")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dt = 0.1
    ticks_per_frame = 5  # 50 Hz model, 10 Hz perception

    cfg = front_pair_config(num_envs=1, num_people=2, frames=args.frames,
                            episodes=args.episodes, seed=args.seed,
                            avatar_dir=str(REPO / "perception/assets/rocketbox"),
                            avatar_count=args.avatars, save_seg_every=1,
                            avatar_list=("Business_Female_01,Business_Female_02,"
                                         "Business_Female_03,Business_Male_01,"
                                         "Business_Male_02,Business_Male_03,"
                                         "Business_Male_04,Business_Female_04")
                            if args.similar_distractor else "",
                            keep_in_fan=False,
                            num_walls=args.num_walls,
                            num_pillars=args.num_pillars)
    scene = PeopleScene(cfg)
    worker = DetectionWorker(args.weights, boxmot=args.tracker)
    crop_manifest = None
    if args.save_crops:
        crop_dir = Path(args.save_crops)
        (crop_dir / "crops").mkdir(parents=True, exist_ok=True)
        crop_manifest = (crop_dir / "manifest.jsonl").open("w")
    results = []
    try:
        for episode in range(args.episodes):
            scene.reset(0, episode_seed=args.seed + episode)
            people = sorted(scene.people[0], key=lambda p: p.person_id)
            target, distractor = people[0], people[1]
            # Target starts 5 m ahead and walks away at 0.4 m/s with a slow
            # weave; the distractor stands 8 m out to the side.
            target.pos = np.array([5.0, 0.0])
            side = 1.0 if args.exit_side == "left" else -1.0
            distractor.pos = np.array([7.5, side * 4.0])
            distractor.vel = np.zeros(2)

            robot = P0Robot()
            first = None
            # Render frame 0 and initialise the tracker on the GT box
            # (stand-in for the operator's click).
            roll = 0.0
            scene.set_cameras(0, roll_deg=roll, base_yaw=robot.yaw,
                              base_xy=(robot.x, robot.y))
            scene._apply_people(0)
            views = scene.render(0, save_seg=True)
            seg0 = np.asarray(views[0]["seg"])
            gt_target = next(p for p in _gt_people(seg0, people, 0.0)
                             if p["person_id"] == 1)
            image = np.asarray(views[0]["color"])
            seg = np.asarray(views[0]["seg"])
            box, _ = _bbox_from_mask(seg == 1, 30)
            tracker = TargetTracker(init_box=np.asarray(box, dtype=float),
                                    init_appearance=hsv_histogram(image, box),
                                    continuous_reid=True,
                                    coast_frames=5,
                                    init_fallback_frames=10,
                                    init_location_priority=True)

            labels, rows = [], []
            beta_prev, beta_rate_prev = 0.0, 0.0
            feature_window = deque(maxlen=8)
            target_track_id = None
            if args.policy_ckpt:
                policy = _load_policy(args.policy_ckpt)
            lost_since = None
            search_t0 = None
            sweep_dir = 1.0
            last_range = None
            tgt_world_est = None        # (pos, vel) world-frame belief
            reacquired_t = None
            phase = "FOLLOW"            # FOLLOW / SEARCH / CONFIRM / STOP
            target_track_id = None
            tgt_template = None         # HSV of the target at init
            miss_count = 0
            relocks = 0
            world_layer = None
            world_belief = None
            world_camera = FrontPairCamera()
            confirm_since = None
            ever_searched = False
            last_seen_t = 0.0
            search_policy = None
            for frame in range(args.frames):
                if frame:
                    scene._apply_people(0)
                # --- perception at 10 Hz ---
                t = frame * dt
                roll = (args.roll_amp * math.sin(2 * math.pi * t / 1.7)
                        if args.roll_amp else 0.0)
                scene.set_cameras(0, roll_deg=roll, base_yaw=robot.yaw,
                                  base_xy=(robot.x, robot.y))
                scene._apply_people(0)
                views = scene.render(0, save_seg=True)
                left = views[0]
                image = np.asarray(left["color"])
                seg = np.asarray(left["seg"])
                gt = _gt_people(seg, people, roll)
                dets = worker.detect(image)
                if args.tracker:
                    # ID-lock protocol: worker-side boxmot keeps tracks; the
                    # target is locked once by overlap with the GT init box.
                    if target_track_id is None:
                        gt_t = next(p for p in gt if p["person_id"] == 1)
                        if gt_t["visible_box"] is not None:
                            best_id, best_iou = None, 0.3
                            for tr in dets:
                                value = iou(tr["bbox"], gt_t["visible_box"])
                                if value > best_iou:
                                    best_id, best_iou = tr["track_id"], value
                            if best_id is not None:
                                target_track_id = best_id
                                tgt_template = hsv_histogram(
                                    image, gt_t["visible_box"])
                                miss_count = 0
                    tbox = next((np.asarray(t["bbox"], dtype=float)
                                 for t in dets
                                 if t.get("track_id") == target_track_id), None)
                    if tbox is None:
                        miss_count += 1
                        # Hybrid: after 1 s without the locked track, try a
                        # B0-style HSV re-lock onto the active tracks.  This
                        # keeps OccluBoost's association while retaining a way
                        # back after the track dies (strict ID lock cannot).
                        if (args.relock and tgt_template is not None
                                and miss_count >= 10):
                            best_id, best_sim = None, 0.55
                            for tr in dets:
                                hist = hsv_histogram(image, tr["bbox"])
                                if hist is None or tgt_template is None:
                                    continue
                                denom = float(np.linalg.norm(hist)
                                              * np.linalg.norm(tgt_template))
                                sim = (float(np.dot(hist, tgt_template) / denom)
                                       if denom > 0 else 0.0)
                                if sim > best_sim:
                                    best_id, best_sim = tr["track_id"], sim
                            if best_id is not None:
                                target_track_id = best_id
                                miss_count = 0
                                relocks += 1
                                tbox = next(
                                    (np.asarray(t["bbox"], dtype=float)
                                     for t in dets
                                     if t.get("track_id") == target_track_id),
                                    None)
                    else:
                        miss_count = 0
                    state = TargetState(bbox=tbox, track_id=target_track_id,
                                        lost=tbox is None)
                    detections = []
                elif args.target_layer:
                    detections = []
                    for det in dets:
                        bbox = np.asarray(det["bbox"], dtype=float)
                        world = world_camera.detection_to_world(
                            bbox,
                            robot_xy=(robot.x, robot.y),
                            robot_yaw=robot.yaw,
                            depth=left.get("depth") if args.use_depth else None,
                            side="left")
                        if world is None:
                            continue
                        detections.append(PersonDetection(
                            position_xy=(world["position_world_xy"]
                                         if world.get("consistent", True)
                                         else None),
                            bbox=bbox,
                            appearance=hsv_histogram(image, bbox),
                            score=float(det.get("confidence", 1.0)),
                            covariance_xy=world["covariance_xy"],
                            meta=dict(range_source=world["range_source"]),
                        ))
                    if world_layer is None:
                        gt_t = next(p for p in gt if p["person_id"] == 1)
                        if gt_t["visible_box"] is not None:
                            world_layer = TargetPerceptionLayer(
                                init_bbox=np.asarray(gt_t["visible_box"],
                                                     dtype=float),
                                init_appearance=hsv_histogram(
                                    image, gt_t["visible_box"]),
                                max_prediction_s=1.0,
                                max_memory_s=5.0,
                                birth_confidence_min=0.25,
                                init_location_priority=True,
                            )
                    if world_layer is not None:
                        world_belief = world_layer.update(detections, t)
                        state = TargetState(
                            bbox=world_belief.bbox,
                            track_id=world_belief.track_id,
                            score=world_belief.identity_score,
                            lost=world_belief.lost,
                        )
                    else:
                        state = TargetState(bbox=None, track_id=None, lost=True)
                else:
                    detections = []
                    for det in dets:
                        x1, y1, x2, y2 = det["bbox"]
                        detections.append(Detection(
                            bbox=np.asarray([x1, y1, x2, y2], dtype=float),
                            score=float(det.get("confidence", 1.0)),
                            appearance=hsv_histogram(image,
                                                     [x1, y1, x2, y2])))
                    state = tracker.update(detections)

                # --- score against GT ---
                label = "A"
                if state is not None and state.bbox is not None and not state.lost:
                    best_id, best_iou = None, 0.0
                    for person in gt:
                        value = iou(state.bbox, person["visible_box"])
                        if value > best_iou:
                            best_id, best_iou = person["person_id"], value
                    if best_iou >= 0.3:
                        label = "C" if best_id == 1 else "W"
                labels.append(label)

                # --- controller (envelope-constrained) ---
                target_gt = next(p for p in gt if p["person_id"] == 1)
                # The controller may only use tracker output: an accepted box
                # (C or W) drives the follow/confirm logic; anything else is a
                # loss.  GT labels only score, never steer.
                control_on = (state is not None and state.bbox is not None
                              and not state.lost)
                if args.target_layer:
                    control_on = bool(
                        world_belief is not None
                        and world_belief.position_xy is not None
                        and not world_belief.lost)
                if control_on:
                    if args.target_layer:
                        # The world-layer output is already the metric target
                        # position; use it directly.  During a short dropout
                        # this is the Kalman prediction, which is exactly what
                        # lets the controller keep following for up to 1 s.
                        dx_world = (world_belief.position_xy[0] - robot.x)
                        dy_world = (world_belief.position_xy[1] - robot.y)
                        last_range = float(math.hypot(dx_world, dy_world))
                        beta = wrap_angle(
                            math.atan2(dy_world, dx_world) - robot.yaw)
                        beta_rate = float(np.clip((beta - beta_prev) / dt,
                                                  -1.0, 1.0))
                        beta_rate_prev = beta_rate
                        beta_prev = beta
                        range_est = last_range
                        last_seen_t = t
                    else:
                        last_range = 1.7 * FX / max(
                            state.bbox[3] - state.bbox[1], 1.0)
                        u = (state.bbox[0] + state.bbox[2]) / 2.0
                        h = state.bbox[3] - state.bbox[1]
                        beta = math.atan2(CX - u, FX) + math.radians(TOE_OUT_DEG)
                        beta_rate = float(np.clip((beta - beta_prev) / dt,
                                                  -1.0, 1.0))
                        beta_rate_prev = beta_rate
                        beta_prev = beta
                        range_est = 1.7 * FX / max(h, 1.0)
                        last_seen_t = t
                stale_s = t - last_seen_t

                if control_on:
                    if phase in ("SEARCH", "STOP", "CONFIRM"):
                        # candidate back: require confirm_tau of continuous
                        # acceptance before committing FOLLOW (delta-tau)
                        if confirm_since is None:
                            confirm_since = t
                        phase = ("FOLLOW" if t - confirm_since >= args.confirm_tau
                                 else "CONFIRM")
                    else:
                        phase = "FOLLOW"
                    if phase == "FOLLOW":
                        confirm_since = None
                        if reacquired_t is None and ever_searched:
                            reacquired_t = t
                        if range_est < 1.5:
                            a1 = -0.3      # emergency braking (P0 cannot stop fast)
                        elif range_est < 2.5:
                            a1 = 0.0       # coast to a halt
                        elif range_est < 4.0:
                            a1 = 0.4
                        elif range_est > 8.0:
                            a1 = args.far_drive
                        else:
                            a1 = args.cruise
                        kappa_des = ((args.kp * beta + args.kd * beta_rate)
                                     / max(robot.v, 0.3)) if robot.v > 0.05 else 0.0
                    else:  # CONFIRM: close gently on the candidate
                        a1 = 0.4
                        kappa_des = float(np.clip(
                            0.8 * beta / max(robot.v, 0.3), -0.3, 0.3))
                else:
                    confirm_since = None
                    if lost_since is None:
                        lost_since = t
                    if args.search == "none":
                        beta, beta_rate = beta_prev, 0.0
                        a1 = 0.0 if stale_s > 2.0 else 0.3
                        kappa_des = 0.0
                        if stale_s > 2.0:
                            phase = "STOP"
                    elif stale_s >= 0.5 and phase in ("FOLLOW", "CONFIRM"):
                        beta, beta_rate = beta_prev, 0.0
                        phase = "SEARCH"
                        search_t0 = t
                        ever_searched = True
                        # reason-conditional choice, GT-free inputs only
                        moving = (abs(beta_rate_prev) * (last_range or 0.0)
                                  > 0.1)
                        if args.search == "reason_cond":
                            search_policy = ("predictive" if moving
                                             and last_range is not None
                                             else "sweep")
                        elif args.search == "predictive":
                            search_policy = "predictive"
                        else:
                            search_policy = "sweep"
                        sweep_dir = 1.0 if beta >= 0 else -1.0
                        if args.search == "blind":
                            sweep_dir = 1.0   # blind: fixed direction
                        if last_range is not None:
                            wx = robot.x + last_range * math.cos(
                                robot.yaw + beta)
                            wy = robot.y + last_range * math.sin(
                                robot.yaw + beta)
                            vt = beta_rate_prev * last_range
                            tgt_world_est = (
                                np.array([wx, wy]),
                                np.array([-math.sin(robot.yaw + beta),
                                          math.cos(robot.yaw + beta)]) * vt)
                        else:
                            tgt_world_est = None
                    elif phase == "SEARCH":
                        pass
                    elif stale_s > 2.0:
                        phase = "STOP" if phase != "SEARCH" else phase

                    if phase == "SEARCH":
                        if (search_policy == "predictive"
                                and (stale_s > 8.0 or tgt_world_est is None)):
                            search_policy = "sweep"   # belief stale -> cover
                        if t - search_t0 > args.search_budget:
                            phase = "STOP"
                    if phase == "SEARCH":
                        if search_policy == "predictive":
                            age = stale_s
                            p_hat = tgt_world_est[0] + tgt_world_est[1] * age
                            err = wrap_angle(
                                math.atan2(p_hat[1] - robot.y,
                                           p_hat[0] - robot.x) - robot.yaw)
                            a1 = 0.77 if age < 6.0 else 0.4
                            kappa_des = float(np.clip(
                                1.2 * err / max(robot.v, 0.3),
                                -max_curvature(a1), max_curvature(a1)))
                        else:  # sweep / last_bearing: single-direction arc
                            a1 = 0.4
                            kappa_des = 0.43 * sweep_dir
                    else:  # STOP
                        beta, beta_rate = beta_prev, 0.0
                        a1, kappa_des = 0.0, 0.0
                kappa_des = float(np.clip(kappa_des, -max_curvature(robot.a1_f),
                                          max_curvature(robot.a1_f)))
                a2 = invert_steer(robot.a1_f, kappa_des)
                if args.policy_ckpt:
                    feature_window.append([
                        min(stale_s, 10.0) / 10.0, beta_prev,
                        (last_range or 10.0) / 10.0, robot.v / 1.2,
                        1.0 if phase == "SEARCH" else 0.0,
                        1.0 if target_gt["in_fov"] else 0.0])
                    window = np.zeros((8, 6), dtype=np.float32)
                    vals = list(feature_window)
                    window[8 - len(vals):] = vals
                    with torch.no_grad():
                        pred = policy(torch.tensor(
                            [window.tolist()], dtype=torch.float32))[0]
                    a1 = float(pred[0])
                    a2 = float(pred[1]) * 0.9
                for _ in range(ticks_per_frame):
                    robot.tick(a1, a2)

                # --- crop collection for the identity verifier (T-B) ---
                if args.save_crops and (phase in ("SEARCH", "CONFIRM")
                                        or stale_s < 1.0):
                    crop_dir = Path(args.save_crops)
                    for det_idx, det in enumerate(dets):
                        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                        crop = image[max(0, y1):y2, max(0, x1):x2]
                        if crop.size == 0:
                            continue
                        crop = cv2.resize(crop, (64, 128))
                        name = (f"e{episode:02d}_f{frame:04d}_{det_idx}.jpg")
                        cv2.imwrite(str(crop_dir / "crops" / name), crop)
                        # GT identity by IoU against the scored persons
                        gt_id, gt_iou = None, 0.3
                        for person in gt:
                            value = iou([x1, y1, x2, y2], person["visible_box"])
                            if value > gt_iou:
                                gt_id, gt_iou = person["person_id"], value
                        crop_manifest.write(json.dumps(dict(
                            crop=name, episode=episode, frame=frame,
                            t=round(t, 3), det_idx=det_idx,
                            box=[x1, y1, x2, y2],
                            gt_person_id=gt_id,
                            phase=phase)) + "\n")


                rows.append(dict(
                    frame=frame, t=round(t, 3), label=label,
                    phase=phase,
                    stale_s=round(stale_s, 2),
                    beta=round(beta_prev, 4),
                    range_est=round(range_est, 2) if control_on else None,
                    search=search_t0 is not None,
                    robot_xy=[round(robot.x, 3), round(robot.y, 3)],
                    robot_yaw=round(robot.yaw, 4), v=round(robot.v, 3),
                    a1=round(robot.a1_f, 3), a2=round(robot.a2_f, 3),
                    target_pos=[round(float(target_gt["world"][0]), 3),
                                round(float(target_gt["world"][1]), 3)],
                    in_fov=target_gt["in_fov"],
                    range_m=round(float(np.hypot(
                        target_gt["world"][0] - robot.x,
                        target_gt["world"][1] - robot.y)), 3),
                ))
                if first is None:
                    first = True
                if frame % 100 == 0:
                    print(f"ep{episode} frame {frame}: v={robot.v:.2f} "
                          f"label={label}", flush=True)
                # --- scenario scripting ---
                if args.scenario == "stop":
                    if t < args.stop_t:
                        target.vel = np.array([
                            args.target_speed * math.cos(0.3 * math.sin(0.15 * t)),
                            args.target_speed * math.sin(0.3 * math.sin(0.15 * t))])
                    else:
                        target.vel = np.zeros(2)   # abrupt halt
                elif args.scenario == "fov_exit":
                    side = 1.0 if args.exit_side == "left" else -1.0
                    if t < 15.0:
                        target.vel = np.array([0.0, side * args.target_speed])
                    elif t < 27.0:
                        target.vel = np.zeros(2)   # fully outside the FOV
                    else:
                        target.vel = np.array([0.0, -side * args.target_speed])
                elif args.scenario == "crossing":
                    target.vel = np.array([
                        args.target_speed * math.cos(0.3 * math.sin(0.15 * t)),
                        args.target_speed * math.sin(0.3 * math.sin(0.15 * t))])
                    # The distractor walks across the TARGET at nearly the
                    # same depth, so both merge in the image around t=15 s
                    # while the ball is moving and swaying.
                    if t < 10.0:
                        distractor.pos = target.pos + np.array([0.35, -2.5])
                        distractor.vel = np.zeros(2)
                    else:
                        distractor.vel = np.array([0.0, 0.5])
                else:  # weave
                    target.vel = np.array([
                        args.target_speed * math.cos(0.3 * math.sin(0.15 * t)),
                        args.target_speed * math.sin(0.3 * math.sin(0.15 * t))])
                target.pos = target.pos + target.vel * dt
                distractor.pos = distractor.pos + distractor.vel * dt

            counts = Counter(labels)
            total = len(labels)
            in_fov = sum(r["in_fov"] for r in rows)
            tracked = [r for r in rows if r["label"] == "C"]
            follow_ok = sum(1 for r in tracked if 2.0 <= r["range_m"] <= 8.0
                            and r["in_fov"])
            ranges = [r["range_m"] for r in rows]
            extra = dict(min_range_m=round(float(min(ranges)), 2))

            # --- reacquisition event metrics (SOP 04 doc, section 6.4) ---
            lost_events, relocks, search_path = [], 0, 0.0
            i = 0
            while i < len(rows):
                if not rows[i]["in_fov"]:
                    j = i
                    while j < len(rows) and not rows[j]["in_fov"]:
                        j += 1
                    if (j - i) * 0.1 >= 0.5:      # a genuine loss event
                        event = dict(start_t=rows[i]["t"],
                                     revisible_t=rows[j]["t"] if j < len(rows) else None,
                                     reacquired=False, ttr=None, censored=j >= len(rows))
                        for r in rows[j:]:
                            if r["label"] == "C":
                                event["reacquired"] = True
                                event["ttr"] = round(r["t"] - event["revisible_t"], 2)
                                break
                        lost_events.append(event)
                    i = j
                else:
                    i += 1
            wrong_streak = 0
            in_search = False
            for r in rows:
                in_search = in_search or r.get("search", False)
                if in_search and r["label"] == "W":
                    wrong_streak += 1
                    if wrong_streak == 10:        # >=1 s on the wrong person
                        relocks += 1
                else:
                    wrong_streak = 0
                if r.get("search"):
                    search_path += r["v"] * 0.1
            extra.update(
                lost_events=len(lost_events),
                reacquired=sum(1 for e in lost_events if e["reacquired"]),
                censored=sum(1 for e in lost_events if e["censored"]),
                reacq_ttr=[e["ttr"] for e in lost_events if e["ttr"] is not None],
                false_relocks=relocks,
                search_path_m=round(search_path, 1),
                lost_detail=lost_events)
            if args.scenario == "stop":
                after = [r for r in rows if r["t"] >= args.stop_t]
                extra["range_at_stop"] = next(
                    (r["range_m"] for r in after if r["t"] >= args.stop_t), None)
                extra["min_range_after_stop"] = round(
                    float(min(r["range_m"] for r in after)), 2) if after else None
                extra["ball_stop_s"] = next(
                    (r["t"] for r in after if r["v"] < 0.02), None)
            if args.scenario == "fov_exit":
                exit_row = next((r for r in rows if not r["in_fov"]), None)
                reentered = [r for r in rows
                             if r["t"] >= 24.0 and r["label"] == "C"]
                extra["exit_t"] = exit_row["t"] if exit_row else None
                extra["reacquire_t"] = reentered[0]["t"] if reentered else None
                extra["W_while_lost"] = sum(
                    1 for r in rows if r["label"] == "W")
            if args.scenario == "crossing":
                extra["W_total"] = counts["W"]
                extra["C_last_half"] = sum(
                    1 for r in rows[len(rows) // 2:] if r["label"] == "C")
            results.append(dict(
                episode=episode, frames=total, scenario=args.scenario,
                relocks=relocks,
                C=counts["C"], W=counts["W"], A=counts["A"],
                correct_coverage=counts["C"] / total,
                follow_fraction=follow_ok / total,
                in_fov_fraction=in_fov / total,
                mean_range=float(np.mean(ranges)),
                max_v=float(np.max([r["v"] for r in rows])),
                **extra,
            ))
            print(f"ep{episode}: C={counts['C']} W={counts['W']} A={counts['A']} "
                  f"follow={results[-1]['follow_fraction']:.2f} "
                  f"mean_range={results[-1]['mean_range']:.2f}", flush=True)
            (out / f"ep{episode:02d}_rows.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows))
    finally:
        worker.close()
        if crop_manifest is not None:
            crop_manifest.close()
        scene.close()

    summary = dict(kp=args.kp, kd=args.kd, cruise=args.cruise,
                   target_speed=args.target_speed, episodes=results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"},
                     indent=2))


def _gt_people(seg, people, roll):
    """GT for scoring: visible box (seg), projected full box, world position."""
    out = []
    for person in people:
        mask = seg == person.person_id
        visible_box, pixels = _bbox_from_mask(mask, 30)
        out.append(dict(
            person_id=int(person.person_id),
            avatar=getattr(person, "avatar", ""),
            visible_box=visible_box,
            full_box=person_full_bbox(person, 1.0, roll, FX, CX, CY),
            visible=pixels >= 30,
            world=(float(person.pos[0]), float(person.pos[1])),
            in_fov=visible_box is not None,
        ))
    return out


if __name__ == "__main__":
    main()
