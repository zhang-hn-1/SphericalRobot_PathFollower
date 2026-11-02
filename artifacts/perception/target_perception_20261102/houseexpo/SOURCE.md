# HouseExpo floorplans (open-source scene source)

Source: **HouseExpo — A Large-scale 2D Indoor Layout Dataset**  
Repository: https://github.com/TeaganLi/HouseExpo  
License: **MIT** (see `LICENSE`)  
Paper: Tingguang Li et al., *HouseExpo: A Large-scale 2D Indoor Layout Dataset
for Learning-based Indoor Navigation*.

## What is included

`raw/` contains 12 selected floorplan JSON files from the public HouseExpo
archive.  `houseexpo_*.json` are converted single-scene maps in the local map
schema used by `perception/sim/map_perception.py` and
`perception/sim/map_follow.py`.

The converter is `perception/tools/houseexpo_to_maps.py`:

```bash
python3 perception/tools/houseexpo_to_maps.py \
    --raw /path/to/HouseExpo/json \
    --out perception/assets/maps/houseexpo \
    --limit 12 --max-scan 482 --scale 1.35
```

## Conversion notes

- `verts` is a wall polyline; every consecutive segment becomes a thin wall
  polygon (0.10 m thick, 1.5 m high). Door gaps are preserved.
- Start/goal are the centres of two different room bounding boxes.
- The whole apartment is scaled by **1.35×** because the HouseExpo doors are
  often ~0.8 m and the spherical robot is ~0.8 m in diameter.  This is an
  explicit robot-size adaptation, not the original dataset scale.
- `index.json` records the selected floorplans, room count and scaled route
  length.

The full HouseExpo archive is not copied here; only these selected floorplans
are included so the scene suite stays small and reproducible.  Attribution
must be kept when redistributing these derived maps.
