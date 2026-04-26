# Phase 2 Packet Ingest Design

## Implemented Path

Phase 2 now has a direct packet-backed scene loader.

If `source_path` contains `packets.jsonl`, `Scene(...)` will load the dataset through the OpenVINS packet path instead of the COLMAP path.

## Source Layout

Expected input layout:

- `packets.jsonl`
- `images/cam*/frame_*.png`

The packet export root itself is the mapper input.

## Current Behavior

The packet loader in `scene/dataset_readers.py` does the following:

1. reads `packets.jsonl`
2. skips malformed trailing packet lines caused by interrupted export
3. creates one `CameraInfo` per exported image
4. computes camera poses from:
   - `body_to_world`
   - `camera_to_body`
5. derives FoV values from exported intrinsics and image resolution
6. splits train and test views with the same `eval`/holdout convention used elsewhere
7. builds a minimal initialization point cloud from exported sparse tracks
8. falls back to a deterministic forward-ray seed if triangulation is empty
9. stores the generated seed as `phase2_points3d.ply`

## Fixed-Pose Stance

This Phase 2 implementation is fixed-pose only.

- packet poses are treated as authoritative
- no pose refinement path is introduced
- the existing training loop is unchanged
- the existing Gaussian initialization and optimization code is unchanged after scene load

## Current Limitations

- the loader uses the packet images directly and does not undistort them
- the first seed cloud is intentionally minimal
- sparse-track triangulation uses a simple geometric filter, not a bundle-adjusted reconstruction
- interrupted packet exports may leave a truncated final line, which is skipped rather than repaired

## Usage

Point the existing training entrypoint at a packet export root:

```bash
python train_nomask.py -s /path/to/phase1_export_root -m /path/to/output
```

As long as `/path/to/phase1_export_root` contains `packets.jsonl`, the packet loader is selected automatically.
