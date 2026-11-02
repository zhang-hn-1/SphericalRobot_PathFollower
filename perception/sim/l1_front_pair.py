#!/usr/bin/env python3
"""L1 data generator for the 20-robot front stereo pair.

Renders the *published* image the real driver produces: upright portrait
768x1024 (2x2-binned 2048x1536), both eyes toed out by 6 degrees, camera at
the lens centre 0.497 m above the ground, pendulum roll applied as a smooth
sinusoid.

Outputs (per eye): RGB image, instance-segmentation mask, YOLO-seg polygons,
and a ground-truth record with visible box, projected full box, distance and
occlusion state.

Smoke test:
    /home/jason/legged_gym/.venv/bin/python perception/sim/l1_front_pair.py \
        --out /tmp/l1smoke --envs 1 --people 2 --frames 8 --episodes 1 --avatars 4
"""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from people_scene import PeopleScene, SceneConfig, _bbox_from_mask

# Front pair, ball-centre frame.  Camera optical centre = lens top face.
FRONT_CAM_FORWARD = 0.21565
FRONT_HALF_BASELINE = 0.31821
FRONT_CAM_UP = 0.097
BALL_RADIUS = 0.40
TOE_OUT_DEG = 6.0
WIDTH, HEIGHT = 768, 1024
HORIZONTAL_FOV_DEG = 78.3


def front_pair_config(**overrides):
    base = SceneConfig(
        width=WIDTH,
        height=HEIGHT,
        horizontal_fov=HORIZONTAL_FOV_DEG,
        cam_forward=FRONT_CAM_FORWARD,
        cam_half_baseline=FRONT_HALF_BASELINE,
        cam_up=FRONT_CAM_UP,
        ball_radius=BALL_RADIUS,
        toe_out_deg=TOE_OUT_DEG,
        # Published images are already upright: no rig rotation, no upright
        # post-rotation.
        rig_image_rotation_deg=0.0,
        save_upright=False,
        half_fan_deg=34.0,
        near_plane=0.08,
    )
    return replace(base, **overrides)


def camera_basis(side, roll_deg, base_yaw=0.0):
    """Optical axis, image-right and image-up unit vectors in the base frame."""
    yaw = base_yaw + side * math.radians(TOE_OUT_DEG)
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    # At yaw = 0: image right is base -Y (robot's right), image up is base +Z.
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    phi = math.radians(roll_deg)
    # Sign fixed empirically against the renderer: at +roll the horizon in the
    # image turns so that a point on the image right moves *up*.
    right_r = right * math.cos(phi) - up * math.sin(phi)
    up_r = right * math.sin(phi) + up * math.cos(phi)
    return forward, right_r, up_r


def project_point(point, side, roll_deg, fx, cx, cy, base_yaw=0.0):
    cam = np.array([FRONT_CAM_FORWARD, side * FRONT_HALF_BASELINE,
                    BALL_RADIUS + FRONT_CAM_UP])
    forward, right, up = camera_basis(side, roll_deg, base_yaw)
    d = np.asarray(point, dtype=float) - cam
    depth = float(d @ forward)
    if depth <= 1e-3:
        return None
    return (cx + fx * float(d @ right) / depth,
            cy - fx * float(d @ up) / depth)


def person_full_bbox(person, side, roll_deg, fx, cx, cy):
    height = getattr(person, "height", 1.675) or 1.675
    ratio = height / 1.675
    hx, hy = 0.17 * ratio, 0.29 * ratio
    us, vs = [], []
    cy_, sy_ = math.cos(person.yaw), math.sin(person.yaw)
    for sx in (-hx, hx):
        for sy in (-hy, hy):
            for z in (0.0, height):
                wx = person.pos[0] + cy_ * sx - sy_ * sy
                wy = person.pos[1] + sy_ * sx + cy_ * sy
                uv = project_point((wx, wy, z), side, roll_deg, fx, cx, cy)
                if uv is None:
                    return None
                us.append(uv[0])
                vs.append(uv[1])
    x1, x2 = max(0.0, min(us)), min(float(WIDTH), max(us))
    y1, y2 = max(0.0, min(vs)), min(float(HEIGHT), max(vs))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def mask_to_polygons(mask, min_area=30):
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        eps = 0.003 * cv2.arcLength(contour, True)
        poly = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        if len(poly) >= 3:
            out.append(poly.astype(float))
    return out


def label_line(polygons):
    """One YOLO-seg line (class + normalised polygon) per instance."""
    lines = []
    for poly in polygons:
        parts = ["0"]
        for x, y in poly:
            parts.append(f"{x / WIDTH:.6f}")
            parts.append(f"{y / HEIGHT:.6f}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--envs", type=int, default=2)
    parser.add_argument("--people", type=int, default=3)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--avatars", type=int, default=12)
    parser.add_argument("--avatar-list", default="",
                        help="comma-separated avatar names; overrides --avatars")
    parser.add_argument("--avatar-dir",
                        default=str(Path(__file__).resolve().parent.parent
                                    / "assets" / "rocketbox"))
    parser.add_argument("--roll-amp", type=float, default=10.0,
                        help="pendulum roll amplitude in degrees (0 disables)")
    parser.add_argument("--roll-period", type=float, default=1.7)
    parser.add_argument("--save-seg-npy", action="store_true", default=False,
                        help="also store the uint16 instance mask (large)")
    args = parser.parse_args()

    out = Path(args.out)
    for sub in ("images", "labels", "seg", "gt"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    cfg = front_pair_config(
        num_envs=args.envs,
        num_people=args.people,
        frames=args.frames,
        episodes=args.episodes,
        seed=args.seed,
        avatar_dir=args.avatar_dir,
        avatar_count=args.avatars,
        avatar_list=args.avatar_list,
        save_seg_every=1,
    )
    scene = PeopleScene(cfg)
    fx = (WIDTH / 2.0) / math.tan(math.radians(HORIZONTAL_FOV_DEG) / 2.0)
    cx, cy = WIDTH / 2.0 - 0.5, HEIGHT / 2.0 - 0.5
    records = []
    dt = cfg.dt
    try:
        for env_index in range(cfg.num_envs):
            for episode in range(cfg.episodes):
                episode_key = env_index * 1000 + episode
                scene.reset(env_index, episode_seed=episode_key)
                for frame in range(cfg.frames):
                    if frame:
                        scene.advance(env_index)
                    t = frame * dt
                    roll = (args.roll_amp * math.sin(2 * math.pi * t /
                                                     args.roll_period)
                            if args.roll_amp else 0.0)
                    scene.set_cameras(env_index, roll_deg=roll)
                    views = scene.render(env_index, save_seg=True)
                    stem = f"e{env_index:03d}_ep{episode:02d}_f{frame:06d}"
                    for eye_index, eye in enumerate(("left", "right")):
                        side = 1.0 if eye == "left" else -1.0
                        view = views[eye_index]
                        img = np.asarray(view["color"])
                        Image.fromarray(img).save(out / "images" /
                                                  f"{stem}_{eye}.png")
                        seg = np.asarray(view["seg"]).astype(np.uint16)
                        if args.save_seg_npy:
                            np.save(out / "seg" / f"{stem}_{eye}.npy", seg.astype(np.uint8))

                        persons = []
                        polys = []
                        for person in scene.people[env_index]:
                            mask = seg == person.person_id
                            box, pixels = _bbox_from_mask(mask, cfg.min_visible_pixels)
                            full = person_full_bbox(person, side, roll, fx, cx, cy)
                            dist = scene.person_distance(env_index, person)
                            if box is not None:
                                polys.extend(mask_to_polygons(mask))
                            persons.append(dict(
                                person_id=int(person.person_id),
                                avatar=getattr(person, "avatar", ""),
                                height=float(getattr(person, "height", 1.675)),
                                pos=[float(person.pos[0]), float(person.pos[1])],
                                yaw=float(person.yaw),
                                visible_box=box,
                                full_box=full,
                                visible_pixels=int(pixels),
                                range_m=round(float(dist), 3),
                                occluded=bool(box is not None and full is not None and
                                              (box[2] - box[0]) < 0.8 * (full[2] - full[0])),
                            ))
                        (out / "labels" / f"{stem}_{eye}.txt").write_text(
                            label_line(polys) + ("\n" if polys else ""))
                        records.append(dict(
                            frame=frame, t=round(t, 4), roll_deg=round(roll, 3),
                            eye=eye, image=f"{stem}_{eye}.png",
                            episode=episode_key, env=env_index,
                            split="val" if env_index == cfg.num_envs - 1 else "train",
                            persons=persons,
                        ))
                print(f"env {env_index} episode {episode} done", flush=True)
    finally:
        scene.close()

    with (out / "gt" / "tracks.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    meta = dict(
        rig="front_pair_20", width=WIDTH, height=HEIGHT,
        horizontal_fov_deg=HORIZONTAL_FOV_DEG, fx=fx, cx=cx, cy=cy,
        cam_forward=FRONT_CAM_FORWARD, baseline=2 * FRONT_HALF_BASELINE,
        cam_height=BALL_RADIUS + FRONT_CAM_UP, toe_out_deg=TOE_OUT_DEG,
        roll_amp_deg=args.roll_amp, roll_period_s=args.roll_period,
        cam_shake_deg=args.roll_amp, dt=dt,
        avatars=list(scene.avatar_pool), avatar_dir=args.avatar_dir,
        seed=args.seed, split_rule="last environment is val",
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # Ultralytics dataset description.  One image per eye.
    train = sorted({r["image"] for r in records if r["split"] == "train"})
    val = sorted({r["image"] for r in records if r["split"] == "val"})
    (out / "train.txt").write_text(
        "\n".join(str((out / "images" / n).resolve()) for n in train))
    (out / "val.txt").write_text(
        "\n".join(str((out / "images" / n).resolve()) for n in val))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        f"train: train.txt\nval: val.txt\n"
        f"names:\n  0: person\n")
    print(f"wrote {len(records)} frames ({len(train)} train / {len(val)} val images) to {out}")


if __name__ == "__main__":
    main()
