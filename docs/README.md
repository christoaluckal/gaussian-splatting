# Frankenstein Base Docs

This directory is the working knowledge base for the active `frankenstein_base` pipeline.

Current documents:

- `current_runner_behavior.md`: describes the copied `gaussian-splatting` runner and LoD behavior as it exists now.
- `colmap_splitter_behavior.md`: describes the current split-scene generation contract consumed by the runner.
- `handoff.md`: operational handoff covering validated behavior, machine requirements, and known portability limits.
- `initialization_behavior.md`: describes the active initialization behavior in `frankenstein_base`, including the direct EDGS / RoMa bridge and split-block handling.
- `logging_behavior.md`: describes the current W&B, CSV, and runtime logging surfaces, including EDGS init timing metrics.
- `edgs_pipeline_notes.md`: summarizes how EDGS launches experiments and performs RoMa-based Gaussian initialization.
- `roma_integration_plan.md`: initial integration plan for bringing EDGS-style RoMa initialization into `frankenstein_base`.
- `frankenstein_base_access_plan.md`: file ownership and dependency plan for making the integration self-contained inside `frankenstein_base`.
