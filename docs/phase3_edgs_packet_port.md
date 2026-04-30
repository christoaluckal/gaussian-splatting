# Phase 3 EDGS Packet Port

## Goal

Port EDGS initialization from a COLMAP-shaped offline assumption to a packet-scene assumption without changing the correspondence core or the optimizer loop.

## Current Start State

EDGS already runs through `apply_edgs_initialization(...)` in `edgs_init.py`.

That path still delegates the actual dense-correspondence work to the EDGS repo, but the mapper-side preparation has now started to become packet-aware.

## Implemented First Step

The packet scene loader now attaches packet metadata to each training camera:

- `packet_index`
- `timestamp_sec`
- `camera_id`
- `frame_id`
- packet camera model / calibration metadata needed for rectified packet-image loading

The mapper-side EDGS wrapper now supports restricting initialization to a contiguous packet window when packet-backed cameras are available.

New knobs:

- `--edgs_packet_window_size`
- `--edgs_packet_window_anchor {start,middle,end}`
- `--edgs_skip_frames`
- `--edgs_max_frames`

Current behavior:

- if `edgs_packet_window_size <= 0`, EDGS sees the full training camera list
- if dataset-level packet subsampling is active through `--packet_stride` / `--packet_offset`, EDGS only sees that retained dataset subset
- if `edgs_packet_window_size > 0` and packet metadata exists, EDGS sees only a fixed contiguous packet-frame set
- the selected set size is `max(edgs_nns_per_ref, edgs_packet_window_size)`
- after that fixed set is chosen, `edgs_skip_frames` keeps every `(skip_frames + 1)`th frame
- after skipping, `edgs_max_frames` optionally caps the EDGS frame set
- if fewer than two packet-backed cameras are available, EDGS falls back to the unfiltered training list or aborts initialization cleanly

Important distinction:

- `--packet_stride` / `--packet_offset` change the whole packet dataset used by training, evaluation, the seed cloud, and EDGS
- `--edgs_packet_window_size`, `--edgs_skip_frames`, and `--edgs_max_frames` only further restrict the camera set seen by EDGS initialization

## Why This Is The Right First Step

- it preserves the existing EDGS correspondence implementation
- it moves the packet/COLMAP distinction to the mapper-side preparation seam
- it makes the first packet-native ablation possible:
  full replay cameras versus packet-window cameras

## Remaining Phase 3 Work

- define a better packet-window policy than simple contiguous slicing
- decide how reference-frame selection should interact with packet ordering
- validate that packet-backed color images behave well with RoMa on this data
- run the first sparse-only versus sparse-plus-EDGS packet ablation
