# OpenVINS + EDGS Roadmap

## Scope

This is the canonical roadmap and status document for the `frankenstein` OpenVINS-to-Gaussian-splatting pipeline.

It replaces the older split between:

- `openvins_edgs_plan.md`
- `docs/openvins_edgs_phases/*.md`

The project stance remains unchanged:

- OpenVINS owns visual-inertial state estimation
- `gaussian-splatting` owns Gaussian initialization, append, and refinement
- the first useful system is replay-first and file-backed, not live-first

## Current Architecture

What already exists in tree:

- offline `gaussian-splatting` mapping with COLMAP-backed initialization
- optional EDGS / RoMa initialization on top of the base mapper
- split-scene append as an internal precedent for future packet-window append
- ROS2 OpenVINS replay on:
  - converted real bags such as `bags/table_02_ros2`
  - TartanAir-style sequences converted into ROS2 bags
- Phase 1 packet export from OpenVINS replay
- direct packet-backed mapper loading from `packets.jsonl`
- packet-window EDGS initialization controls on the mapper side

What still does not exist:

- a frozen packet schema document and validator
- packet-window append for incremental map growth
- a live transport path that reuses the same packet contract
- a complete comparison harness versus Photo-SLAM

## Phase Status

### Phase 0: Freeze The Mapping Baseline

Status: complete

Completed:

- the mapper-side baseline contract is frozen in `phase0_mapping_baseline.md`
- append semantics, EDGS initialization semantics, LoD scheduling semantics, and runtime logging semantics are treated as fixed reference behavior
- the ROS/OpenVINS path is defined as one bag-driven source, not a required `base` / `split` dataset layout

Primary reference docs:

- `phase0_mapping_baseline.md`
- `initialization_behavior.md`
- `current_runner_behavior.md`
- `logging_behavior.md`
- `edgs_pipeline_notes.md`

### Phase 1: Define The OpenVINS Export Contract

Status: in progress

Goal:

- make OpenVINS emit replayable packet exports that can drive the mapper without COLMAP at runtime

Implemented so far:

- ROS2 replay works for `bags/table_02_ros2`
- TartanAir sequences can be converted into ROS2 bags with RGB, depth, IMU, a manifest, and a GT sidecar CSV
- the Phase 1 exporter writes:
  - `packets.jsonl`
  - `images/cam*/frame_*.png`
- exported packets already contain:
  - timestamp
  - frame id
  - image path
  - intrinsics and distortion
  - `camera_to_body`
  - `body_to_camera`
  - `body_to_world`
  - `world_to_body`
  - camera-IMU time offset
  - pose covariance
  - sparse tracks

Still open:

- freeze the on-disk packet schema explicitly
- add a packet validator
- add a manifest next to `packets.jsonl`
- decide whether exported frames remain replay-frame based or become a stricter keyframe subset
- fix the remaining shutdown-only crash on replay teardown

Primary reference docs:

- `openvins_replay_runbook.md`
- `phase2_packet_ingest_design.md`

### Phase 2: Build A Frozen-Pose Mapper Input Path

Status: in progress

Goal:

- feed OpenVINS-exported packets into the current mapper while holding poses fixed

Implemented so far:

- `Scene(...)` auto-detects `packets.jsonl` and uses the packet-backed path
- the loader builds camera objects from packet poses and intrinsics
- the loader creates a minimal sparse seed from packet sparse tracks and falls back to a deterministic seed when needed
- low-memory packet-backed training smoke tests have completed in this workspace

Still open:

- harden the seed-generation path
- preserve the current mapper behavior while broadening packet-backed coverage
- decide whether further calibration work is needed beyond the current rectified packet path

Primary reference docs:

- `phase2_packet_ingest_design.md`
- `packet_edgs_pitfalls.md`

### Phase 3: Port EDGS Initialization To Packet Input

Status: started

Goal:

- make EDGS initialization operate on OpenVINS packet windows instead of COLMAP-shaped scene prep

Implemented so far:

- packet-scene cameras now carry packet metadata into the mapper
- EDGS can restrict initialization with:
  - `--edgs_packet_window_size`
  - `--edgs_packet_window_anchor`
  - `--edgs_skip_frames`
  - `--edgs_max_frames`
- packet-window EDGS smoke runs have completed on sample packet data

Still open:

- refine packet-window selection beyond simple contiguous slicing
- define packet-native dense seed behavior more explicitly
- run the first clean sparse-only versus sparse-plus-EDGS packet ablation

Primary reference docs:

- `phase3_edgs_packet_port.md`
- `packet_edgs_pitfalls.md`

### Phase 4: Map Growth By Incremental Append

Status: not started

Goal:

- replace offline split-scene append prototyping with append blocks built from packet windows drawn from one replay stream

Next steps:

1. define the packet-window-to-append-block contract
2. keep block building asynchronous from the active training loop
3. reuse the current append seam instead of introducing a second merge path

### Phase 5: Introduce Online Transport After Replay

Status: not started

Goal:

- add a live transport path only after replay and file-backed packet ingest are stable

Next steps:

1. mirror the packet schema in ROS2 transport
2. add a live packet logger
3. emit queue depth, packet age, mapper lag, and dropped-packet diagnostics

### Phase 6: Compare Against Photo-SLAM

Status: not started

Goal:

- compare the decoupled OpenVINS + mapper design against the tightly coupled Photo-SLAM baseline without conflating unrelated variables

Planned comparison axes:

- time to first usable render
- PSNR / SSIM / LPIPS
- Gaussian count
- peak memory
- mapper lag
- robustness under motion and weak texture

## Immediate Resume Point

As of this workspace copy, the most useful current path is:

1. replay OpenVINS on `bags/table_02_ros2` or a generated TartanAir ROS2 bag
2. export Phase 1 packets to a retained packet root
3. point `gaussian-splatting/train_nomask.py` at that packet root
4. use Phase 3 EDGS controls for packet-window initialization ablations

For TartanAir in this tree, the retained packet export root is currently:

- `bags/tartanair_packets`

The mapper entrypoint already recognizes that layout directly.
