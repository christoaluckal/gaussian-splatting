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
3. optionally subsamples the packet sequence at load time through:
   - `--packet_stride`
   - `--packet_offset`
4. creates one `CameraInfo` per retained exported image
5. computes camera poses from:
   - `body_to_world`
   - `camera_to_body`
6. resolves a rectified pinhole camera model from the exported packet calibration
7. uses rectified intrinsics for FoV derivation and sparse-track rays
8. optionally mirrors packet images and packet sparse-track coordinates through:
   - `--packet_flip_lr`
   - `--packet_flip_ud`
9. splits train and test views with the same `eval`/holdout convention used elsewhere
10. builds a minimal initialization point cloud from exported sparse tracks
11. falls back to a deterministic forward-ray seed if triangulation is empty
12. stores the generated seed as:
   - `phase2_points3d.ply` for the full packet set
   - `phase2_points3d_strideN_offsetK.ply` for subsampled packet runs

## Fixed-Pose Stance

This Phase 2 implementation is fixed-pose only.

- packet poses are treated as authoritative
- no pose refinement path is introduced
- the existing training loop is unchanged
- the existing Gaussian initialization and optimization code is unchanged after scene load

Dataset-level implication:

- when `--packet_stride` is used, the retained packet subset becomes the whole mapper dataset
- training, evaluation, seed-cloud construction, and EDGS initialization all operate on that same retained subset

## Current Limitations

- the loader still trusts the exported packet poses as-is
- the loader does not yet compensate `camera_imu_time_offset_sec`
- the first seed cloud is intentionally minimal
- sparse-track triangulation uses a simple geometric filter, not a bundle-adjusted reconstruction
- interrupted packet exports may leave a truncated final line, which is skipped rather than repaired

Current packet-image handling:

- packet `radtan` images are undistorted on load
- the packet sparse-track pixels are rectified into the same pinhole camera model before ray construction
- optional packet flips are applied consistently to both images and packet sparse-track coordinates

## Usage

Point the existing training entrypoint at a packet export root:

```bash
python train_nomask.py -s /path/to/phase1_export_root -m /path/to/output
```

As long as `/path/to/phase1_export_root` contains `packets.jsonl`, the packet loader is selected automatically.

Example with dataset-level packet subsampling and held-out eval:

```bash
python train_nomask.py \
  -s /path/to/phase1_export_root \
  -m /path/to/output \
  --eval \
  --packet_stride 10 \
  --packet_offset 0
```
