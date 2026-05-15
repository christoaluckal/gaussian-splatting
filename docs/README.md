# Frankenstein Base Docs

This directory is the working knowledge base for the active `frankenstein_base` pipeline.

Current documents:

- `current_runner_behavior.md`: describes the copied `gaussian-splatting` runner and LoD behavior as it exists now.
- `colmap_splitter_behavior.md`: describes the current split-scene generation contract consumed by the runner.
- `phase0_mapping_baseline.md`: freezes the mapper-side baseline for the OpenVINS roadmap and clarifies that split append is an internal precedent, not a required ROS input layout.
- `phase2_packet_ingest_design.md`: describes the implemented packet-backed scene loader and fixed-pose Phase 2 mapper path.
- `phase3_edgs_packet_port.md`: describes the packet-window EDGS initialization seam and the current fixed-set selection rule.
- `packet_edgs_pitfalls.md`: records packet transform, EDGS stride, resolution-scale, iteration-0 save, and simple-knn pitfalls observed during the OpenVINS packet port.
- packet docs now also cover dataset-level packet subsampling, rectified packet-image loading, and optional packet image flips used by the active loader.
- `handoff.md`: operational handoff covering validated behavior, machine requirements, and known portability limits.
- `initialization_behavior.md`: describes the active initialization behavior in `frankenstein_base`, including the direct EDGS / RoMa bridge and split-block handling.
- `logging_behavior.md`: describes the current W&B, CSV, and runtime logging surfaces, including EDGS init timing metrics.
- `experiment_reporting.md`: describes the local collated CSV/Markdown report, including EDGS/LoD inference for short run names and the post-init max GPU metric.
- `edgs_pipeline_notes.md`: summarizes how EDGS launches experiments and performs RoMa-based Gaussian initialization.
- `roma_integration_plan.md`: initial integration plan for bringing EDGS-style RoMa initialization into `frankenstein_base`.
- `frankenstein_base_access_plan.md`: file ownership and dependency plan for making the integration self-contained inside `frankenstein_base`.

Related phase tracker:

- `../docs/openvins_edgs_phases/phase_0.md`
- `../docs/openvins_edgs_phases/phase_1.md`
- `../docs/openvins_edgs_phases/phase_2.md`
- `../docs/openvins_edgs_phases/phase_3.md`
