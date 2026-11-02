#!/usr/bin/env python3
"""Crossing sequences for the 20-robot front pair: the failure case the SOP targets.

Two people walk past each other in opposite directions at a fixed depth in front
of a stationary robot, so their image boxes overlap and merge for a short
window.  Ground truth keeps the true identity, so "did the tracker keep the
right person through the crossing" can be scored frame by frame.

    /home/jason/legged_gym/.venv/bin/python perception/sim/l1_crossing.py \
        --out /home/jason/perception_data/l1_crossing_v1 --episodes 4 --frames 150
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from l1_front_pair import (FRONT_HALF_BASELINE, HEIGHT, HORIZONTAL_FOV_DEG,
                           WIDTH, camera_basis, front_pair_config,
                           person_full_bbox)
from people_scene import PeopleScene, _bbox_from_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--distance", type=float, default=5.0)
    parser.add_argument("--lateral", type=float, default=0.65)
    parser.add_argument("--speed", type=float, default=0.4)
    parser.add_argument("--depth-gap", type=float, default=0.35)
    parser.add_argument("--roll-amp", type=float, default=10.0)
    parser.add_argument("--num-walls", type=int, default=0,
                        help="random walls; 0 keeps the measurement corridor "
                             "clear so the crossing event is isolated")
    parser.add_argument("--num-pillars", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--avatars", type=int, default=8)
    parser.add_argument("--avatar-list", default="",
                        help="comma-separated avatar names; overrides --avatars")
    parser.add_argument("--avatar-dir",
                        default=str(Path(__file__).resolve().parent.parent
                                    / "assets" / "rocketbox"))
    args = parser.parse_args()

    out = Path(args.out)
    for sub in ("images", "labels", "gt"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    cfg = front_pair_config(num_envs=1, num_people=2, frames=args.frames,
                            episodes=args.episodes, seed=args.seed,
                            avatar_dir=args.avatar_dir,
                            avatar_count=args.avatars,
                            avatar_list=args.avatar_list,
                            num_walls=args.num_walls,
                            num_pillars=args.num_pillars,
                            save_seg_every=1,
                            keep_in_fan=False)
    scene = PeopleScene(cfg)
    fx = (WIDTH / 2.0) / math.tan(math.radians(HORIZONTAL_FOV_DEG) / 2.0)
    cx, cy = WIDTH / 2.0 - 0.5, HEIGHT / 2.0 - 0.5
    dt = cfg.dt
    records = []
    try:
        for episode in range(args.episodes):
            scene.reset(0, episode_seed=args.seed + episode)
            people = sorted(scene.people[0], key=lambda p: p.person_id)
            # Person 1 is the target and starts on the robot's right; person 2
            # is the distractor on the left, 0.35 m further away.
            start = [
                np.array([args.distance, -args.lateral]),
                np.array([args.distance + args.depth_gap, +args.lateral]),
            ]
            vel = [
                np.array([0.0, +args.speed]),
                np.array([0.0, -0.75 * args.speed]),
            ]
            for frame in range(args.frames):
                t = frame * dt
                for person, p0, v in zip(people, start, vel):
                    person.pos = p0 + v * t
                    person.vel = v
                scene._apply_people(0)
                roll = (args.roll_amp * math.sin(2 * math.pi * t / 1.7)
                        if args.roll_amp else 0.0)
                scene.set_cameras(0, roll_deg=roll)
                views = scene.render(0)
                stem = f"ep{episode:02d}_f{frame:06d}"
                for eye_index, eye in enumerate(("left", "right")):
                    side = 1.0 if eye == "left" else -1.0
                    view = views[eye_index]
                    Image.fromarray(np.asarray(view["color"])).save(
                        out / "images" / f"{stem}_{eye}.png")
                    people_gt = []
                    for person in people:
                        mask = view["seg"] == person.person_id
                        box, pixels = _bbox_from_mask(mask, cfg.min_visible_pixels)
                        full = person_full_bbox(person, side, roll, fx, cx, cy)
                        people_gt.append(dict(
                            person_id=int(person.person_id),
                            avatar=getattr(person, "avatar", ""),
                            pos=[round(float(person.pos[0]), 4),
                                 round(float(person.pos[1]), 4)],
                            visible_box=box, full_box=full,
                            visible=box is not None,
                            visible_pixels=int(pixels),
                        ))
                    records.append(dict(
                        episode=episode, frame=frame, t=round(t, 4),
                        eye=eye, image=f"{stem}_{eye}.png", roll_deg=round(roll, 3),
                        target_id=1, persons=people_gt,
                    ))
            print(f"episode {episode} done", flush=True)
    finally:
        scene.close()

    with (out / "gt" / "tracks.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (out / "meta.json").write_text(json.dumps(dict(
        rig="front_pair_20", scenario="crossing", width=WIDTH, height=HEIGHT,
        fx=fx, cx=cx, cy=cy, dt=dt, baseline=2 * FRONT_HALF_BASELINE,
        distance=args.distance, lateral=args.lateral, speed=args.speed,
        depth_gap=args.depth_gap, roll_amp_deg=args.roll_amp,
        episodes=args.episodes, frames=args.frames, seed=args.seed,
    ), indent=2))
    print(f"wrote {len(records)} frames to {out}")


if __name__ == "__main__":
    main()
