# ReplicaCAD apartment stage (open-source 3D scene)

Source: **SAPIEN-ReplicaCAD** mirror
(https://github.com/angli66/SAPIEN-ReplicaCAD), original scene assets from
Meta's ReplicaCAD.

License: **CC BY-NC 4.0** — attribution required, non-commercial use only.
See `LICENSE.txt`.

Files:

- `stage.obj`, `stage.mtl`, `Image_*.png`: `frl_apartment_stage.glb` converted
  with Blender 2.82.  Doors are retained in the visual shell; the stage demo
  does not derive a 2D occupancy map from it.
- `stage.urdf`: static visual-only URDF used by `perception/sim/stage_follow.py`.

The scene is used for research/simulation only. The original GLB is not
included here.
