# Packet EDGS Pitfalls

## Scope

This note records the current packet-backed OpenVINS + EDGS issues that are easy to miss when porting runs between machines.

Relevant files:

- `scene/dataset_readers.py`
- `scene/gaussian_model.py`
- `edgs_init.py`
- `train_nomask.py`

## Transform Convention

OpenVINS packet quaternions are not consumed with the plain standard `xyzw` matrix expansion directly.

The packet loader expands each quaternion and then transposes the matrix in `_quat_xyzw_to_rotmat(...)`.

Reason:

- the packet fields are labeled as transforms such as `body_to_world` and `camera_to_body`
- sparse-track validation showed that using the plain expansion as-is produced bad geometry
- transposing the expanded matrix reduced sparse-track median normalized reprojection error from roughly `0.17` to roughly `0.002`

Practical implication:

- do not remove the transpose as a cleanup
- if packet export code changes, re-run sparse-track reprojection validation before trusting EDGS geometry
- a visually bad EDGS point cloud is a strong signal to re-check transform direction, not only EDGS hyperparameters

## Packet Intrinsics

Packet `radtan` images are undistorted on load and sparse-track pixels are rectified into the same pinhole model.

The loader uses the rectified intrinsics for FoV derivation.

Important limitation:

- the current 3DGS projection matrix path is FoV-centered
- it does not explicitly carry principal-point offsets through the renderer projection matrix
- this is less severe when the rectified principal point is near image center, but it remains a calibration approximation

If high-quality packet renders plateau with correct transforms, principal-point support is a likely next calibration issue to inspect.

## EDGS Frame Ordering

Packet-backed EDGS selection must be packet-ordered before skip/cap controls are applied.

Current behavior:

- when packet metadata exists, EDGS sorts by:
  - `packet_index`
  - `camera_id`
  - `timestamp_sec`
  - `image_name`
- `--edgs_skip_frames 9` then means every 10th packet-backed camera in time order

Previous pitfall:

- `Scene(...)` usually shuffles training cameras
- applying `--edgs_skip_frames` to that shuffled list made an apparent stride-10 EDGS run sample arbitrary cameras
- this produced misleading EDGS initialization behavior

## Dataset Stride vs EDGS Stride

There are two distinct stride surfaces.

`--packet_stride` and `--packet_offset` change the whole loaded dataset:

- training cameras
- eval cameras
- packet seed cloud
- EDGS input cameras

`--edgs_skip_frames`, `--edgs_packet_window_size`, and `--edgs_max_frames` only change the cameras used by EDGS initialization.

For an ablation where training should still see every retained frame but EDGS should use every 10th frame, prefer:

```bash
--edgs_skip_frames 9
```

For an ablation where the whole mapper should operate on every 10th exported packet, use:

```bash
--packet_stride 10
```

Do not treat these as equivalent.

## Resolution Scale

`--resolution_scales` is not only a visualization setting.

It controls:

- training camera tensor resolution
- eval camera tensor resolution
- W&B eval images
- EDGS camera tensors, because EDGS receives the finest configured scale

For the `848x480` packet stream:

- `--resolution_scales 1` loads `848x480`
- `--resolution_scales 2` loads `424x240`

So a run at scale 2 can have improving PSNR while still looking soft when inspected at larger display size.

## Iteration-0 Inspection

Fresh runs save initialized Gaussians before the first optimizer step:

```text
point_cloud/iteration_0/point_cloud.ply
```

Use this file to inspect:

- packet sparse seed quality
- EDGS appended Gaussian geometry
- whether bad geometry exists before training or appears later during optimization

This save is skipped when resuming from `--start_checkpoint`.

## simple-knn Initialization Warning

On this port, `simple-knn` can fail during initial scale estimation even for a tiny packet seed cloud.

Observed warning:

```text
[WARN] simple-knn distCUDA2 failed during initial scale estimation ... Falling back to torch.cdist.
```

For the packet seed cloud with about 1.6k points, this is not real model-size VRAM pressure. It indicates a likely stale or incompatible `simple-knn` CUDA extension.

The fallback is acceptable for small packet seeds. To remove the warning, rebuild `submodules/simple-knn` in the exact Python / PyTorch / CUDA environment used for training.

## Recommended Debug Order

When EDGS initialization looks bad:

1. Inspect `point_cloud/iteration_0/point_cloud.ply`.
2. Confirm startup logs show expected packet counts, seed point count, and camera resolution.
3. Confirm `[EDGS init] Selected ... packet range ... time span ...` matches the intended stride/window.
4. Re-check sparse-track reprojection under the current transform convention if geometry is wildly wrong.
5. Only then tune EDGS hyperparameters such as `matches_per_ref`, `num_refs`, `nns_per_ref`, and `scaling_factor`.
