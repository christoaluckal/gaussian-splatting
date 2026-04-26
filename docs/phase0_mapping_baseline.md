# Phase 0 Mapping Baseline

## Scope

This document freezes the mapper-side baseline that should remain stable while the OpenVINS and ROS 2 ingestion path is introduced.

It is the Phase 0 contract for the proposed OpenVINS + EDGS system.

## Core clarification

For the ROS/OpenVINS version, the upstream source is assumed to be one bag-driven sensor stream.

That means:

- there is no required explicit delineation between `base` and `split` in the input data source
- `base` and `split` should not be treated as required user-facing dataset concepts for the ROS path
- the current split-scene machinery in `frankenstein_base` is only an internal precedent for append behavior

Phase 0 therefore freezes mapper behavior, not dataset partitioning.

## Frozen mapper invariants

The following logic is considered fixed for Phase 0:

- Gaussian append semantics
- EDGS initialization semantics
- LoD scheduling semantics
- runtime logging semantics
- comparison-runner semantics

These are the behaviors that later OpenVINS integration must preserve unless a later phase explicitly changes them.

## Source-of-truth documents

Phase 0 depends on the following documents as the current implementation contract:

- `initialization_behavior.md`
- `current_runner_behavior.md`
- `logging_behavior.md`
- `handoff.md`
- `edgs_pipeline_notes.md`

The split-scene behavior document is still relevant, but only as an internal append precedent:

- `colmap_splitter_behavior.md`

## What is preserved in Phase 0

### 1. SfM / COLMAP initialization reference

The mapper can initialize from a COLMAP point cloud through the existing `Scene(...)` and `GaussianModel.create_from_pcd(...)` path.

This remains the offline reference initialization path.

### 2. EDGS initialization reference

The mapper can augment the COLMAP seed with EDGS / RoMa correspondences through `edgs_init.py`.

This remains the offline dense-initialization reference path.

### 3. Internal append reference

The mapper already supports appending prebuilt Gaussian blocks through the split-scene machinery.

For Phase 0 this means:

- append behavior exists already
- append behavior is mapper-internal
- append behavior is not yet the ROS/OpenVINS ingestion interface

Later phases should reuse the append seam, not the current split-scene dataset convention.

### 4. Existing LoD behavior

The LoD policy described in `current_runner_behavior.md` is frozen.

Important current rule:

- `resolution_scales` are authoritative
- the runner passes `-r 1`
- LoD promotion and reset behavior in `train_nomask.py` should be treated as baseline behavior, not redesign targets for Phase 1

### 5. Existing runtime and experiment logging

The logging surfaces described in `logging_behavior.md` are frozen.

That includes:

- W&B naming and grouping
- local CSV outputs
- initialization-time runtime metrics
- EDGS initialization timing metrics
- iteration-time training and eval metrics

## What Phase 0 explicitly does not assume

- a ROS-facing `base` scene
- a ROS-facing `split` scene
- multiple bag files as the primary input abstraction
- a live online append pipeline
- tracker-mapper feedback

## Phase 0 interpretation of "split append"

In the original offline pipeline, split append is implemented via sibling COLMAP model directories such as `model0`, `model1`, and `model2`.

For the OpenVINS roadmap, that should be interpreted narrowly:

- it proves the mapper can merge newly created Gaussian blocks
- it exposes a usable append seam
- it does not define the long-term external data contract

The future ROS/OpenVINS system should replace the current split-scene producer, not reproduce it verbatim.

## Phase 0 deliverable

Phase 0 is complete when the team agrees to preserve the following mapper-side contract while building the ROS/OpenVINS bridge:

- initialization remains whatever `initialization_behavior.md` describes today
- LoD remains whatever `current_runner_behavior.md` describes today
- append remains whatever the current Gaussian merge path does today
- logging remains whatever `logging_behavior.md` describes today

In short:

- one bag is the future frontend data source
- the current mapper logic stays fixed
- split-scene machinery is only the internal prototype for append

## Handoff implication

If later work appears to require changing append semantics, EDGS initialization semantics, or LoD semantics before OpenVINS packet ingestion exists, that work is no longer Phase 0.

It should be treated as a later-phase mapping change and justified separately.
