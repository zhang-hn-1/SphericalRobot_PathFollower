"""Multi-person Isaac Gym scene for the Rotunbot perception layer.

This is a *pipeline* simulator, not a photorealistic renderer.  By default each
person is composed of simple rigid primitives (torso, head, two arms, two
legs).  Passing ``--avatar-dir perception/assets/rocketbox`` switches to the
115 Rocketbox avatars: one URDF actor per person, textured, with a per-person
height.  The primitive path stays the default so results recorded before the
avatars existed remain reproducible.

What the scene gives you per frame:

* stereo RGB (640x480 by default), from the robot camera rig
* depth and instance-segmentation buffers
* per-person visible bounding boxes derived from the segmentation mask
* stable per-person IDs and a designated target ID

No physics is used for the people: gravity is zero and their root states are
teleported along scripted trajectories each frame.

Standalone smoke test::

    source local_env.sh
    /home/jason/legged_gym/.venv/bin/python -m perception.sim.people_scene \
        --num-envs 2 --people 5 --frames 5 --out /tmp/people_smoke

The scene is intentionally independent of ``legged_gym`` so it can run with the
Isaac Gym venv without importing the path-follower task.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_rig import base_to_env  # noqa: E402
from isaacgym import gymapi


# ---------------------------------------------------------------- config


@dataclass
class SceneConfig:
    num_envs: int = 2
    num_people: int = 5
    frames: int = 100
    episodes: int = 1  # episodes per environment
    dt: float = 0.1  # 10 Hz perception cadence
    seed: int = 0

    width: int = 640
    height: int = 480
    horizontal_fov: float = 57.4
    near_plane: float = 0.10
    far_plane: float = 60.0

    # Toe-out of each eye, degrees; 0 keeps the historical parallel rig.
    toe_out_deg: float = 0.0

    # Robot camera rig, base_link frame.  The ball centre sits ~0.40 m above
    # the ground, so the world height is 0.40 + 0.05 = 0.45 m.
    cam_forward: float = 0.20
    cam_half_baseline: float = 0.30
    cam_up: float = 0.05
    ball_radius: float = 0.40

    # The frozen rig lies down: the optical axis is base +X (forward), but
    # image right is base +Z (up) and image down is base -Y (right).  Isaac
    # Gym cameras look along their own +X with image right = +Y, so the rig
    # pose is reproduced by rolling the camera -90 degrees about +X.
    rig_image_rotation_deg: float = -90.0
    rig_roll_deg: float = 0.0  # static rig roll, for ablations

    # Body roll from the pendulum joint.  The geometric effect is about
    # +/-10 degrees even though the joint itself travels +/-30.
    body_roll_deg: float = 0.0

    # Fraction of the image covered by the robot's own shell (46.24% per eye
    # in the frozen rig).  Applied as an overlay so the detector sees the
    # same occlusion it will see in deployment.
    self_occlusion_fraction: float = 0.0
    self_occlusion_mask_path: str = ""

    # Store frames rotated into the upright orientation, so that people stand
    # up and a detector pretrained on upright people transfers.  Training and
    # inference then both live in the upright space.
    save_upright: bool = False

    # Put both eyes into the training set.  The self-occlusion is
    # complementary, so each eye sees a different part of every person.
    train_both_eyes: bool = True

    # People spawn in a fan in front of the robot.
    min_distance: float = 2.0
    max_distance: float = 15.0
    # The rig's vertical FOV is 44.6 deg (+/-22.3 deg) and the horizontal
    # one 57.4 deg, but the image axes are rotated: the *narrow* axis maps
    # to the robot's left-right.  People must spawn inside the narrow
    # cone or they are outside the frame entirely.
    half_fan_deg: float = 20.0
    min_speed: float = 0.0
    max_speed: float = 1.2

    # Keep people inside the forward fan.  The search experiments turn this
    # off so a person can walk out of the rig's narrow field of view and the
    # controller has something to find.
    keep_in_fan: bool = True

    num_walls: int = 6
    num_pillars: int = 4
    world_half_extent: float = 18.0

    save_depth_every: int = 0  # 0 = never
    save_seg_every: int = 0
    min_visible_pixels: int = 30

    # Optional Rocketbox avatars (perception/assets/rocketbox).  When set,
    # each person is a single URDF actor instead of the six primitive parts.
    # The primitive path stays the default so results recorded before the
    # avatars existed remain reproducible.
    avatar_dir: str = ""
    avatar_count: int = 12  # how many of the avatars to load (0 = all)
    avatar_yaw_offset_deg: float = 0.0
    # Explicit comma-separated avatar names.  Overrides the seeded subset so
    # dev/test can use disjoint identities (unseen-avatar evaluation).
    avatar_list: str = ""

    def env_origin(self, index: int) -> tuple[float, float, float]:
        # One row of environments, far enough apart that cameras never see
        # into a neighbouring scene.
        return (index * 60.0, 0.0, 0.0)


@dataclass
class Person:
    person_id: int  # 1-based; 0 is background in the segmentation image
    pos: np.ndarray  # world xy
    vel: np.ndarray  # world xy
    yaw: float
    color: tuple[float, float, float]
    parts: list = field(default_factory=list)  # [(actor_handle, local_xyz)]
    height: float = 1.675  # metres; Rocketbox avatars span 1.43-1.87
    avatar: str = ""       # Rocketbox name, empty when primitives are used
    # Avatar branch: every candidate avatar is pre-spawned for this person
    # slot and hidden at z=-100.  Isaac Gym Preview 4 has no destroy_actor,
    # so per-episode variety is done by switching which pre-spawned actor is
    # placed at the person's position.
    avatar_actors: dict = field(default_factory=dict)


# ---------------------------------------------------------------- scene


class PeopleScene:
    def __init__(self, cfg: SceneConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.gym = gymapi.acquire_gym()
        self.sim = self._create_sim()
        self._init_avatar_pool()
        self.assets = self._build_assets()
        self.envs = []
        self.cams = []  # [(left_handle, right_handle)]
        self.people = []  # list per env of Person
        self.static_actors = []
        self.camera_height = cfg.ball_radius + cfg.cam_up
        self._occlusion_masks = self._load_occlusion_masks()
        self._build_scenes()

    def _load_occlusion_masks(self):
        """Load the left/right self-occlusion masks, if configured."""
        path = self.cfg.self_occlusion_mask_path
        if not path:
            if self.cfg.self_occlusion_fraction > 0.0:
                raise ValueError(
                    "self_occlusion_fraction needs an explicit mask file; "
                    "run perception/make_self_occlusion_mask.py first"
                )
            return [None, None]
        base = Path(path)
        masks = []
        for eye in ("left", "right"):
            candidate = base.with_name(f"{base.stem}_{eye}{base.suffix}")
            if not candidate.exists():
                candidate = base
            if candidate.suffix == ".npy":
                mask = np.load(candidate)
            else:
                mask = np.asarray(Image.open(candidate).convert("L"))
            mask = (mask > 0).astype(np.uint8) * 255
            if mask.shape != (self.cfg.height, self.cfg.width):
                raise ValueError(
                    f"mask {candidate} has shape {mask.shape}, expected "
                    f"{(self.cfg.height, self.cfg.width)}"
                )
            masks.append(mask)
        return masks

    # -- setup ------------------------------------------------------

    def _create_sim(self):
        params = gymapi.SimParams()
        params.up_axis = gymapi.UP_AXIS_Z
        params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)  # people are teleported
        params.dt = self.cfg.dt
        params.substeps = 1
        params.use_gpu_pipeline = False
        params.physx.use_gpu = True
        params.physx.solver_type = 1
        params.physx.num_position_iterations = 2
        params.physx.num_velocity_iterations = 1
        sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, params)
        if sim is None:
            raise RuntimeError("Isaac Gym failed to create a simulation")

        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(sim, plane)
        return sim

    def _build_assets(self):
        opts = gymapi.AssetOptions()
        opts.fix_base_link = True
        wall = self.gym.create_box(self.sim, 2.0, 0.4, 2.0, opts)
        pillar = self.gym.create_capsule(self.sim, 0.5, 1.2, opts)

        dynamic = gymapi.AssetOptions()
        dynamic.fix_base_link = False
        torso = self.gym.create_box(self.sim, 0.34, 0.22, 0.62, dynamic)
        head = self.gym.create_sphere(self.sim, 0.115, dynamic)
        leg = self.gym.create_capsule(self.sim, 0.065, 0.70, dynamic)
        arm = self.gym.create_capsule(self.sim, 0.050, 0.55, dynamic)
        return dict(wall=wall, pillar=pillar, torso=torso, head=head,
                    leg=leg, arm=arm, avatar_assets=self._load_avatar_assets())

    # -- Rocketbox avatars ------------------------------------------
    #
    # The baked assets are one URDF per avatar with a single link (visual =
    # OBJ mesh, collision = cylinder).  Isaac Gym cannot play skinned FBX, so
    # the mesh is a static A-pose; the scene teleports the actor each step,
    # exactly like the primitive parts.

    def _init_avatar_pool(self):
        """Choose which avatars this run loads (seeded, so runs repeat)."""
        self.avatar_pool = []
        self.avatar_heights = {}
        cfg = self.cfg
        if not cfg.avatar_dir:
            return
        root = Path(cfg.avatar_dir)
        if not root.is_dir():
            raise ValueError(f"avatar_dir {root} does not exist")
        names = sorted(p.name for p in root.iterdir()
                       if (p / f"{p.name}.urdf").exists())
        if not names:
            raise ValueError(f"no <name>.urdf found under {root}")
        for name in names:
            report = root / name / "report.json"
            height = 1.75
            if report.exists():
                try:
                    height = float(json.loads(report.read_text())["height_m"])
                except (KeyError, ValueError, OSError):
                    pass
            self.avatar_heights[name] = height
        if cfg.avatar_list:
            requested = [n.strip() for n in cfg.avatar_list.split(",")
                         if n.strip()]
            missing = [n for n in requested
                       if n not in self.avatar_heights]
            if missing:
                raise ValueError(f"avatar_list names not found: {missing[:5]}")
            names = requested
        else:
            rng = np.random.default_rng(cfg.seed + 7717)
            if 0 < cfg.avatar_count < len(names):
                names = sorted(rng.choice(names, cfg.avatar_count,
                                          replace=False))
        self.avatar_pool = names

    def _load_avatar_assets(self):
        if not self.avatar_pool:
            return {}
        opts = gymapi.AssetOptions()
        # Nothing is simulated (the scene teleports actors), so gravity and
        # a fixed base would only get in the way.
        for attr, value in (("fix_base_link", False),
                            ("disable_gravity", True),
                            ("use_mesh_materials", True),
                            ("collapse_fixed_joints", True)):
            if hasattr(opts, attr):
                setattr(opts, attr, value)
        assets = {}
        for name in self.avatar_pool:
            folder = Path(self.cfg.avatar_dir) / name
            assets[name] = self.gym.load_asset(
                self.sim, str(folder), f"{name}.urdf", opts)
        return assets

    def _avatar_yaw_quat(self):
        return gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0.0, 0.0, 1.0),
            math.radians(self.cfg.avatar_yaw_offset_deg))

    def _spawn_avatar(self, env, person_id, name):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, -100.0)  # hidden until reset
        pose.r = self._avatar_yaw_quat()
        return self.gym.create_actor(
            env, self.assets["avatar_assets"][name], pose,
            f"person_{person_id}", 0, 0, person_id)

    def _hide_actor(self, env, actor):
        state = self.gym.get_actor_rigid_body_states(
            env, actor, gymapi.STATE_NONE)
        state["pose"]["p"].fill((0.0, 0.0, -100.0))
        self.gym.set_actor_rigid_body_states(
            env, actor, state, gymapi.STATE_ALL)

    def _add_static(self, env, rng):
        cfg = self.cfg
        for _ in range(cfg.num_walls):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(
                float(rng.uniform(2.0, cfg.max_distance)),
                float(rng.uniform(-10.0, 10.0)),
                1.0,
            )
            pose.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 0, 1), float(rng.uniform(0, math.pi))
            )
            actor = self.gym.create_actor(env, self.assets["wall"], pose, "wall", 0, 0, 0)
            self.gym.set_rigid_body_color(
                env, actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                gymapi.Vec3(0.55, 0.55, 0.58),
            )
            self.static_actors.append((env, actor))
        for _ in range(cfg.num_pillars):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(
                float(rng.uniform(2.0, cfg.max_distance)),
                float(rng.uniform(-10.0, 10.0)),
                1.1,
            )
            pose.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 1, 0), -math.pi / 2.0
            )
            actor = self.gym.create_actor(env, self.assets["pillar"], pose, "pillar", 0, 0, 0)
            self.gym.set_rigid_body_color(
                env, actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                gymapi.Vec3(0.45, 0.47, 0.50),
            )
            self.static_actors.append((env, actor))

    def _build_scenes(self):
        cfg = self.cfg
        props = gymapi.CameraProperties()
        props.width = cfg.width
        props.height = cfg.height
        props.horizontal_fov = cfg.horizontal_fov
        props.near_plane = cfg.near_plane
        props.far_plane = cfg.far_plane
        props.enable_tensors = True

        for env_index in range(cfg.num_envs):
            origin = cfg.env_origin(env_index)
            env = self.gym.create_env(
                self.sim,
                gymapi.Vec3(*origin),
                gymapi.Vec3(origin[0] + 50.0, origin[1] + 30.0, 10.0),
                1,
            )
            self.envs.append(env)
            self._add_static(env, self.rng)

            left = self.gym.create_camera_sensor(env, props)
            right = self.gym.create_camera_sensor(env, props)
            self.cams.append((left, right))

            people = []
            for person_index in range(cfg.num_people):
                people.append(self._create_person(env, person_index + 1))
            self.people.append(people)

    def _create_person(self, env, person_id):
        cfg = self.cfg
        if self.avatar_pool:
            # Pre-spawn every candidate for this slot; only one is placed at
            # the person's position at a time.
            actors = {
                name: self._spawn_avatar(env, person_id, name)
                for name in self.avatar_pool
            }
            name = self.avatar_pool[(person_id - 1) % len(self.avatar_pool)]
            return Person(
                person_id=person_id, pos=np.zeros(2), vel=np.zeros(2), yaw=0.0,
                color=(1.0, 1.0, 1.0),
                parts=[(actors[name], np.zeros(3), self._avatar_yaw_quat())],
                height=self.avatar_heights[name], avatar=name,
                avatar_actors=actors,
            )
        color = self._color_from_index(person_id - 1)
        # Local offsets are in the person frame: x forward, y left, z up.
        # Every part of one person shares the segmentation id so the mask is
        # the union of the parts.  Isaac Gym capsules are built along +X, so
        # limbs get a -90 degree rotation about Y to stand upright.
        upright = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0.0, 1.0, 0.0), -math.pi / 2.0
        )
        identity = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        parts = [
            (self.assets["torso"], (0.0, 0.0, 1.10), identity),
            (self.assets["head"], (0.0, 0.0, 1.56), identity),
            (self.assets["leg"], (0.0, 0.09, 0.42), upright),
            (self.assets["leg"], (0.0, -0.09, 0.42), upright),
            (self.assets["arm"], (0.0, 0.24, 1.05), upright),
            (self.assets["arm"], (0.0, -0.24, 1.05), upright),
        ]
        handles = []
        for asset, offset, base_quat in parts:
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, -100.0)  # hidden until reset
            actor = self.gym.create_actor(
                env, asset, pose, f"person_{person_id}", 0, 0, person_id
            )
            self.gym.set_rigid_body_color(
                env, actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                gymapi.Vec3(*color),
            )
            handles.append(
                (actor, np.asarray(offset, dtype=np.float64), base_quat)
            )
        person = Person(
            person_id=person_id,
            pos=np.zeros(2),
            vel=np.zeros(2),
            yaw=0.0,
            color=color,
            parts=handles,
        )
        return person

    PALETTE = np.array(
        [
            [0.85, 0.15, 0.15], [0.15, 0.75, 0.20], [0.15, 0.25, 0.85],
            [0.90, 0.80, 0.10], [0.80, 0.20, 0.75], [0.10, 0.75, 0.80],
            [0.95, 0.50, 0.10], [0.55, 0.55, 0.55], [0.20, 0.20, 0.20],
            [0.95, 0.95, 0.95],
        ]
    )

    def _color_from_index(self, index):
        base = self.PALETTE[int(index) % len(self.PALETTE)]
        return tuple(
            float(v) for v in np.clip(base + self.rng.normal(0, 0.03, 3), 0, 1)
        )

    # -- episode ----------------------------------------------------

    def reset(self, env_index, episode_seed=None):
        rng = np.random.default_rng(
            self.cfg.seed + env_index * 100003 + (0 if episode_seed is None else episode_seed)
        )
        cfg = self.cfg
        # Distinct clothing colours within one episode.  This is a proxy
        # assumption: it makes the appearance cue meaningful without a ReID
        # model.  Real footage will need learned embeddings.
        color_indices = rng.permutation(len(self.PALETTE))[: len(self.people[env_index])]
        for person, color_index in zip(self.people[env_index], color_indices):
            distance = rng.uniform(cfg.min_distance, cfg.max_distance)
            angle = np.deg2rad(rng.uniform(-cfg.half_fan_deg, cfg.half_fan_deg))
            person.pos = np.array([distance * math.cos(angle), distance * math.sin(angle)])
            speed = rng.uniform(cfg.min_speed, cfg.max_speed)
            # Walk roughly across the field of view, but keep the heading
            # inside the fan so the reflection logic in ``advance`` does not
            # immediately kick in.  Half the people cross left-to-right and
            # half right-to-left, which keeps the lateral distribution
            # symmetric.
            if rng.random() < 0.5:
                heading = angle + rng.uniform(0.15, 0.75)
            else:
                heading = angle - rng.uniform(0.15, 0.75)
            heading = float(np.clip(heading, -np.deg2rad(cfg.half_fan_deg - 1.0),
                                    np.deg2rad(cfg.half_fan_deg - 1.0)))
            person.vel = speed * np.array([math.cos(heading), math.sin(heading)])
            person.yaw = float(rng.uniform(-math.pi, math.pi))
            person.color = self._color_from_index(int(color_index))
            if not self.avatar_pool:
                for actor, _offset, _base_quat in person.parts:
                    self.gym.set_rigid_body_color(
                        self.envs[env_index], actor, 0,
                        gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*person.color),
                    )
        if self.avatar_pool:
            self._respawn_avatars(env_index, rng)
        self._apply_people(env_index)

    def _respawn_avatars(self, env_index, rng):
        """Give every person a fresh avatar at the start of an episode.

        Avatars are sampled without replacement when the pool is large
        enough, so two people in the same episode do not look identical.
        The avatars were all created when the scene was built; Isaac Gym
        Preview 4 cannot destroy an actor, so this hides the previous one and
        promotes an already-existing actor to the person's position.
        """
        env = self.envs[env_index]
        people = self.people[env_index]
        if len(self.avatar_pool) >= len(people):
            names = [str(n) for n in rng.choice(
                self.avatar_pool, size=len(people), replace=False)]
        else:
            names = [str(rng.choice(self.avatar_pool)) for _ in people]
        for person, name in zip(people, names):
            for actor, _offset, _quat in person.parts:
                self._hide_actor(env, actor)
            actor = person.avatar_actors[name]
            person.parts = [(actor, np.zeros(3), self._avatar_yaw_quat())]
            person.avatar = name
            person.height = self.avatar_heights[name]

    def _apply_people(self, env_index):
        env = self.envs[env_index]
        for person in self.people[env_index]:
            cos_y, sin_y = math.cos(person.yaw), math.sin(person.yaw)
            rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
            yaw_quat = gymapi.Quat(
                0.0, 0.0, math.sin(person.yaw / 2.0), math.cos(person.yaw / 2.0)
            )
            for actor, offset, base_quat in person.parts:
                local_xy = rot @ offset[:2]
                world = np.array([
                    person.pos[0] + local_xy[0],
                    person.pos[1] + local_xy[1],
                    offset[2],
                ])
                state = self.gym.get_actor_rigid_body_states(
                    env, actor, gymapi.STATE_NONE
                )
                state["pose"]["p"].fill(tuple(world))
                q = yaw_quat * base_quat
                state["pose"]["r"].fill((q.x, q.y, q.z, q.w))
                self.gym.set_actor_rigid_body_states(env, actor, state, gymapi.STATE_ALL)

    def advance(self, env_index, dt=None):
        """Move the people one step, keeping them in the forward fan.

        The bounce is done in polar coordinates.  Reflecting only the lateral
        velocity component (the previous implementation) let people drift
        systematically towards +Y, which meant almost every person ended up
        on the robot's left -- and the right eye, which is the one blocked on
        that side, could then barely see anyone.
        """
        dt = self.cfg.dt if dt is None else dt
        cfg = self.cfg
        limit = math.radians(cfg.half_fan_deg)
        for person in self.people[env_index]:
            person.pos = person.pos + person.vel * dt

            angle = math.atan2(person.pos[1], person.pos[0])
            distance = float(np.linalg.norm(person.pos))
            if distance < 1e-6:
                angle, distance = 0.0, cfg.min_distance
                person.pos = np.array([distance, 0.0])

            # Reflect the heading off the fan edges and off the near/far
            # limits, so the motion stays inside the fan and stays symmetric.
            if cfg.keep_in_fan and abs(angle) > limit:
                # Mirror the heading about the edge.
                edge = limit if angle > 0 else -limit
                heading = math.atan2(person.vel[1], person.vel[0])
                reflected = 2.0 * edge - heading
                speed = float(np.linalg.norm(person.vel))
                person.vel = speed * np.array([math.cos(reflected), math.sin(reflected)])
                person.pos = np.array([
                    distance * math.cos(edge), distance * math.sin(edge)
                ])
                angle = edge
                distance = float(np.linalg.norm(person.pos))

            if distance < cfg.min_distance:
                person.pos = person.pos / max(distance, 1e-6) * (cfg.min_distance + 0.2)
                person.vel = -person.vel
            elif distance > cfg.max_distance:
                person.pos = person.pos / distance * (cfg.max_distance - 0.2)
                person.vel = -person.vel
            person.yaw += 0.0  # keep facing; gait is not modelled
        self._apply_people(env_index)

    # -- rendering --------------------------------------------------

    def set_cameras(self, env_index, roll_deg=0.0, base_yaw=0.0,
                    base_xy=(0.0, 0.0)):
        """Place both cameras to reproduce the frozen Rotunbot rig.

        The real rig lies down.  Measured from the frozen installation:

        * image right  = base +Z (up)
        * image down   = base -Y (right)
        * optical axis = base +X (forward)

        An Isaac Gym camera looks along its own +X with image right = +Y and
        image down = -Z.  Rolling the camera by **-90 degrees about the
        optical axis** maps that to image right = +Z and image down = -Y,
        which is the frozen rig.

        Verified empirically with a vertical pillar: at 0 degrees the pillar
        stands upright in the image, at -90 degrees it lies down, which is
        how a person appears in the real rig.
        """
        env = self.envs[env_index]
        z = self.camera_height
        # Camera poses are in the *environment* frame, like actor poses; do not
        # add env_origin here or every env except the first looks at empty space.
        rotation_deg = self.cfg.rig_image_rotation_deg + self.cfg.rig_roll_deg + roll_deg
        toe = math.radians(getattr(self.cfg, "toe_out_deg", 0.0))
        for cam, side in zip(self.cams[env_index],
                             (+self.cfg.cam_half_baseline, -self.cfg.cam_half_baseline)):
            base_point = np.array([self.cfg.cam_forward, side, z])
            # side is +baseline on the left eye and -baseline on the right,
            # so its sign also gives the toe-out direction.
            sgn = 1.0 if side > 0 else -1.0
            direction = np.array([math.cos(toe), sgn * math.sin(toe), 0.0])
            position = base_to_env(base_point, base_xy, base_yaw)
            target = base_to_env(base_point + 10.0 * direction,
                                 base_xy, base_yaw)
            self.gym.set_camera_location(
                cam, env, gymapi.Vec3(*position), gymapi.Vec3(*target))
            if rotation_deg:
                # Roll about the camera's *local* optical axis.  The old
                # left-multiplication rolled about world +X, which is only
                # the optical axis at base_yaw = 0; with the base turned it
                # would tilt the horizon instead of reproducing the rig.
                transform = self.gym.get_camera_transform(self.sim, env, cam)
                roll = gymapi.Quat.from_axis_angle(
                    gymapi.Vec3(1.0, 0.0, 0.0), math.radians(rotation_deg)
                )
                transform.r = transform.r * roll
                self.gym.set_camera_transform(cam, env, transform)

    def render(self, env_index, save_depth=False, save_seg=False):
        env = self.envs[env_index]
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        try:
            out = []
            for cam in self.cams[env_index]:
                color = self.gym.get_camera_image(
                    self.sim, env, cam, gymapi.IMAGE_COLOR
                ).reshape(self.cfg.height, self.cfg.width, 4)[:, :, :3]
                depth = self.gym.get_camera_image(
                    self.sim, env, cam, gymapi.IMAGE_DEPTH
                )
                seg = self.gym.get_camera_image(
                    self.sim, env, cam, gymapi.IMAGE_SEGMENTATION
                )
                out.append(dict(color=color, depth=depth, seg=seg))
        finally:
            self.gym.end_access_image_tensors(self.sim)

        # Apply the frozen rig's self-occlusion.  The mask lives in the camera
        # frame, so it is applied after rendering and before anything else
        # looks at the pixels.
        for index, view in enumerate(out):
            mask = self._occlusion_masks[index]
            if mask is not None:
                view["color"] = view["color"].copy()
                view["color"][mask == 0] = 0
                view["seg"] = view["seg"].copy()
                view["seg"][mask == 0] = 0
        return out

    def full_bbox(self, person, cam_y=None):
        """Project the person's full 3-D extent into the left image.

        The visible mask shrinks under occlusion, but the tracker is expected to
        keep following the whole person.  This box is the right reference for
        "did we track the right person"; the segmentation box stays the right
        reference for "what can the detector see".

        The projection uses the frozen rig pose (image right = base +Z,
        image down = base -Y), so a person standing on the ground appears
        *lying down* in the image, exactly as in the real rig.
        """
        cfg = self.cfg
        cam_y = cfg.cam_half_baseline if cam_y is None else cam_y
        cam = np.array([cfg.cam_forward, cam_y, self.camera_height])
        height = getattr(person, "height", 1.675) or 1.675
        ratio = height / 1.675
        hx, hy = 0.17 * ratio, 0.29 * ratio  # torso half-width, arm half-span
        z0, z1 = 0.0, height
        cos_y, sin_y = math.cos(person.yaw), math.sin(person.yaw)
        fx = (cfg.width / 2.0) / math.tan(math.radians(cfg.horizontal_fov) / 2.0)
        us, vs = [], []
        for sx in (-hx, hx):
            for sy in (-hy, hy):
                for z in (z0, z1):
                    wx = person.pos[0] + cos_y * sx - sin_y * sy
                    wy = person.pos[1] + sin_y * sx + cos_y * sy
                    forward = wx - cam[0]
                    if forward <= 1e-3:
                        return None
                    # Rig pose: image right is base +Z (up), image down is
                    # base -Y (right).  So the "right" image coordinate comes
                    # from the height above the camera, and the "down"
                    # coordinate from the lateral offset to the robot's right.
                    up = z - cam[2]
                    right_of_robot = -(wy - cam[1])
                    us.append(cfg.width / 2.0 + fx * up / forward)
                    vs.append(cfg.height / 2.0 + fx * right_of_robot / forward)
        x1, x2 = max(0.0, min(us)), min(float(cfg.width), max(us))
        y1, y2 = max(0.0, min(vs)), min(float(cfg.height), max(vs))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def person_distance(self, env_index, person):
        """Distance from the left camera to the person's torso centre.

        Both poses are in the environment frame; a common translation cancels,
        so no env origin is needed.
        """
        cam = np.array([
            self.cfg.cam_forward,
            self.cfg.cam_half_baseline,
            self.camera_height,
        ])
        torso = np.array([
            person.pos[0],
            person.pos[1],
            0.63 * (getattr(person, "height", 1.675) or 1.675),
        ])
        return float(np.linalg.norm(torso - cam))

    def close(self):
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None


# ---------------------------------------------------------------- CLI


def _bbox_from_mask(mask, min_pixels):
    ys, xs = np.nonzero(mask)
    if len(xs) < min_pixels:
        return None, 0
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], int(len(xs))


def run(cfg: SceneConfig, out_dir: Path, target_mode="random"):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    (out_dir / "labels").mkdir(exist_ok=True)
    (out_dir / "gt").mkdir(exist_ok=True)
    (out_dir / "depth").mkdir(exist_ok=True)
    (out_dir / "seg").mkdir(exist_ok=True)

    scene = PeopleScene(cfg)
    records = []
    try:
        for env_index in range(cfg.num_envs):
            split = "val" if env_index == cfg.num_envs - 1 else "train"
            scene.set_cameras(env_index)
            for episode in range(cfg.episodes):
                episode_key = env_index * 1000 + episode
                scene.reset(env_index, episode_seed=episode_key)
                target_id = None
                for frame in range(cfg.frames):
                    if frame:
                        scene.advance(env_index)
                    save_depth = cfg.save_depth_every and frame % cfg.save_depth_every == 0
                    save_seg = cfg.save_seg_every and frame % cfg.save_seg_every == 0
                    views = scene.render(env_index, save_depth=save_depth, save_seg=save_seg)
                    left = views[0]
                    right = views[1]
                    stem = f"e{env_index:03d}_ep{episode:02d}_f{frame:06d}"

                    # Each eye has its own visible box: the self-occlusion is
                    # complementary (the left eye is blocked on the robot's
                    # right, the right eye on the robot's left), so the two
                    # eyes see different parts of every person.  Labels must
                    # be computed per eye, and a person visible in *either*
                    # eye belongs in the dataset.
                    def eye_boxes(view):
                        found = {}
                        for person in scene.people[env_index]:
                            mask = view["seg"] == person.person_id
                            box, pixels = _bbox_from_mask(mask, cfg.min_visible_pixels)
                            if box is not None:
                                found[person.person_id] = (box, pixels)
                        return found

                    left_boxes = eye_boxes(left)
                    right_boxes = eye_boxes(right)

                    persons = []
                    for person in scene.people[env_index]:
                        if person.person_id not in left_boxes \
                                and person.person_id not in right_boxes:
                            continue
                        l_box, l_px = left_boxes.get(person.person_id, (None, 0))
                        r_box, r_px = right_boxes.get(person.person_id, (None, 0))
                        persons.append(
                            dict(
                                id=int(person.person_id),
                                bbox=[float(v) for v in l_box] if l_box else None,
                                bbox_right=[float(v) for v in r_box] if r_box else None,
                                visible_px=int(l_px),
                                visible_px_right=int(r_px),
                                full_bbox=scene.full_bbox(person),
                                distance_m=round(scene.person_distance(env_index, person), 2),
                                world_xy=[round(float(person.pos[0]), 3), round(float(person.pos[1]), 3)],
                                color=[round(float(c), 3) for c in person.color],
                                avatar=person.avatar,
                                height=round(float(person.height), 4),
                            )
                        )
                    if target_id is None and persons:
                        if target_mode == "nearest":
                            target_id = min(persons, key=lambda p: p["distance_m"])["id"]
                        else:
                            target_id = int(scene.rng.choice([p["id"] for p in persons]))
                    for item in persons:
                        item["is_target"] = item["id"] == target_id

                    # Optionally store the UPRIGHT version of the frame.  The
                    # rig view has people lying down, which wrecks a detector
                    # pretrained on upright people; rotating first recovers
                    # it exactly (verified on COCO samples).  Training and
                    # inference both happen in this upright space.
                    saved_left = left["color"]
                    saved_right = right["color"]
                    if cfg.save_upright:
                        from rig_upright import map_bbox_to_upright, upright_from_rig
                        saved_left, _ = upright_from_rig(left["color"])
                        saved_right, _ = upright_from_rig(right["color"])
                        for person in persons:
                            # A person can be visible in only one eye, so
                            # both boxes are optional; rotate the ones that
                            # exist and leave the others as None.
                            if person.get("bbox"):
                                person["bbox_rig"] = list(person["bbox"])
                                person["bbox"] = map_bbox_to_upright(
                                    person["bbox"], (cfg.height, cfg.width)
                                )
                            if person.get("bbox_right"):
                                person["bbox_right"] = map_bbox_to_upright(
                                    person["bbox_right"], (cfg.height, cfg.width)
                                )
                            if person.get("full_bbox"):
                                # The full-body projection is in the rig
                                # frame too, so it has to be rotated as well.
                                person["full_bbox"] = map_bbox_to_upright(
                                    person["full_bbox"], (cfg.height, cfg.width)
                                )

                    Image.fromarray(saved_left).save(out_dir / "images" / f"{stem}_left.png")
                    Image.fromarray(saved_right).save(out_dir / "images" / f"{stem}_right.png")
                    if save_depth:
                        np.save(out_dir / "depth" / f"{stem}_left.npy", left["depth"])
                    if save_seg:
                        # The mask must live in the same orientation as the
                        # saved image.  Rotating the colour frame but not the
                        # instance mask silently trains segmentation on
                        # transposed labels (found 2026-10-26: the 960 seg
                        # model had visible-box IoU 0.02 with the right area).
                        seg_left = left["seg"]
                        seg_right = right["seg"]
                        if cfg.save_upright:
                            seg_left, _ = upright_from_rig(seg_left)
                            seg_right, _ = upright_from_rig(seg_right)
                        np.save(out_dir / "seg" / f"{stem}_left.npy",
                                np.asarray(seg_left).astype(np.uint16))
                        np.save(out_dir / "seg" / f"{stem}_right.npy",
                                np.asarray(seg_right).astype(np.uint16))

                    # Labels are per eye: the right eye sees a different part
                    # of every person because the self-occlusion is
                    # complementary.  Writing the left-eye labels onto the
                    # right-eye image would be wrong.
                    img_w = cfg.height if cfg.save_upright else cfg.width
                    img_h = cfg.width if cfg.save_upright else cfg.height

                    def label_text(boxes):
                        lines = []
                        for x1, y1, x2, y2 in boxes:
                            cx = (x1 + x2) / 2.0 / img_w
                            cy = (y1 + y2) / 2.0 / img_h
                            w = (x2 - x1) / img_w
                            h = (y2 - y1) / img_h
                            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                        return "\n".join(lines)

                    (out_dir / "labels" / f"{stem}_left.txt").write_text(
                        label_text([p["bbox"] for p in persons if p.get("bbox")])
                    )
                    (out_dir / "labels" / f"{stem}_right.txt").write_text(
                        label_text([p["bbox_right"] for p in persons if p.get("bbox_right")])
                    )

                    records.append(
                        dict(
                            episode=episode_key,
                            env=env_index,
                            split=split,
                            frame=frame,
                            stem=stem,
                            target_id=target_id,
                            persons=persons,
                        )
                    )
    finally:
        scene.close()

    with (out_dir / "gt" / "tracks.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    meta = dict(
        num_envs=cfg.num_envs,
        num_people=cfg.num_people,
        frames=cfg.frames,
        episodes=cfg.episodes,
        width=cfg.width,
        height=cfg.height,
        horizontal_fov=cfg.horizontal_fov,
        cam_half_baseline=cfg.cam_half_baseline,
        camera_height=cfg.ball_radius + cfg.cam_up,
        seed=cfg.seed,
        color_space="RGB",
        split_rule="last environment is val, the rest are train",
        avatar_dir=cfg.avatar_dir,
        avatar_count=len(scene.avatar_pool),
        avatars=list(scene.avatar_pool),
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Ultralytics dataset description.  Images are all in one directory; the
    # split lists are written as symlink-free text lists through train.txt /
    # val.txt so the same directory layout works with the standard loader.
    #
    # Both eyes go into the training set.  The self-occlusion is
    # complementary, so the two eyes see different parts of every person and
    # together they cover the whole forward cone; training on one eye only
    # throws that away.
    train_stems = sorted({r["stem"] for r in records if r["split"] == "train"})
    val_stems = sorted({r["stem"] for r in records if r["split"] == "val"})
    eyes = ("left", "right") if cfg.train_both_eyes else ("left",)
    (out_dir / "train.txt").write_text(
        "\n".join(
            str((out_dir / "images" / f"{s}_{eye}.png").resolve())
            for s in train_stems for eye in eyes
        )
    )
    (out_dir / "val.txt").write_text(
        "\n".join(
            str((out_dir / "images" / f"{s}_{eye}.png").resolve())
            for s in val_stems for eye in eyes
        )
    )
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--people", type=int, default=5)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-mode", default="random", choices=["random", "nearest"])
    parser.add_argument("--save-depth-every", type=int, default=0)
    parser.add_argument("--save-seg-every", type=int, default=0)
    parser.add_argument("--rig-image-rotation", type=float, default=-90.0,
                        help="rig pose: image right = base +Z, image down = base -Y")
    parser.add_argument("--rig-roll", type=float, default=0.0)
    parser.add_argument("--body-roll", type=float, default=0.0)
    parser.add_argument("--occlusion-mask", default="",
                        help="mask base path; _left/_right variants are used if present")
    parser.add_argument("--avatar-dir", default="",
                        help="baked Rocketbox dir, e.g. perception/assets/rocketbox")
    parser.add_argument("--avatar-list", default="",
                        help="comma-separated explicit avatar names; overrides --avatars")
    parser.add_argument("--avatars", type=int, default=12,
                        help="how many avatars to load into the sim (0 = all)")
    parser.add_argument("--avatar-yaw-offset", type=float, default=0.0,
                        help="yaw applied to the mesh so it faces +X")
    parser.add_argument("--save-upright", action="store_true",
                        help="store frames rotated so people stand up")
    args = parser.parse_args(argv)

    cfg = SceneConfig(
        num_envs=args.num_envs,
        num_people=args.people,
        frames=args.frames,
        episodes=args.episodes,
        seed=args.seed,
        save_depth_every=args.save_depth_every,
        save_seg_every=args.save_seg_every,
        rig_image_rotation_deg=args.rig_image_rotation,
        rig_roll_deg=args.rig_roll,
        body_roll_deg=args.body_roll,
        self_occlusion_mask_path=args.occlusion_mask,
        save_upright=args.save_upright,
        avatar_dir=args.avatar_dir,
        avatar_count=args.avatars,
        avatar_yaw_offset_deg=args.avatar_yaw_offset,
        avatar_list=args.avatar_list,
    )
    records = run(cfg, Path(args.out), target_mode=args.target_mode)
    print(f"wrote {len(records)} frames to {args.out}")
    for record in records[:3]:
        print(json.dumps(record))


if __name__ == "__main__":
    main()
