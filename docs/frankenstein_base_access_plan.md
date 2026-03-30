# Frankenstein Base Access Plan

## Purpose

This document describes how future RoMa integration work should access files, modules, and dependencies in the copied `frankenstein_base` tree.

The goal is to make `frankenstein_base` the implementation authority, not a thin wrapper around `EDGS`.

## Ownership rule

Future code changes should be made only under `frankenstein_base` unless a deliberate shared dependency extraction is planned.

In practice:

- `EDGS` is reference material
- `gaussian-splatting` is the immediate upstream baseline
- `frankenstein_base` is the code that will change

## Local versus external dependencies

### What should stay local to `frankenstein_base`

Future implementation modules:

- RoMa initialization helpers
- argument parsing for RoMa options
- scene-block initialization policy
- runner wiring for RoMa-enabled experiment variants
- any future docs describing implemented behavior

Suggested local file targets:

- `frankenstein_base/train_nomask.py`
- `frankenstein_base/run_exp.py`
- `frankenstein_base/arguments/__init__.py`
- `frankenstein_base/scene/__init__.py`
- `frankenstein_base/scene/gaussian_model.py`
- `frankenstein_base/docs/*`
- new `frankenstein_base/roma_init/*`

### What should remain reference-only during implementation

Reference sources:

- `EDGS/source/corr_init.py`
- `EDGS/source/trainer.py`
- `EDGS/configs/train.yaml`
- `EDGS/README.md`

These should guide the port, but `frankenstein_base` should not import them at runtime.

## RoMa dependency plan

There are three practical ways to access RoMa in the future:

### Option 1: repo-level vendored dependency

- keep a single RoMa copy at repo level
- import it from `frankenstein_base` with a controlled path setup or installation step

Pros:

- avoids code duplication

Cons:

- easy to regress into EDGS-path coupling

### Option 2: `frankenstein_base`-local vendored copy

- copy or submodule RoMa under `frankenstein_base/submodules/RoMa`

Pros:

- strongest locality
- clearest ownership for reproducible training

Cons:

- duplicates a dependency already present elsewhere in the repo

### Option 3: environment-installed RoMa package

- require `romatch` to be installed in the active environment

Pros:

- simplest runtime import path

Cons:

- weaker reproducibility unless exact version pinning is documented

Recommended plan:

- prefer an explicit installable dependency or a local submodule for `frankenstein_base`
- avoid `sys.path.append('../submodules/RoMa')`-style imports copied from EDGS

## File access map for the future port

### Runner layer

Primary file:

- `frankenstein_base/run_exp.py`

Responsibilities:

- define experiment matrix
- pass RoMa-related flags into `train_nomask.py`
- keep output naming stable and explicit

Should not do:

- direct RoMa inference
- scene mutation

### Training entrypoint layer

Primary file:

- `frankenstein_base/train_nomask.py`

Responsibilities:

- parse RoMa options
- decide whether base and split blocks use RoMa initialization
- call local initialization helpers
- preserve existing LoD and logging behavior

Should not do:

- own the low-level triangulation logic inline

### Scene/block management layer

Primary file:

- `frankenstein_base/scene/__init__.py`

Responsibilities:

- load `model0`
- load split extension blocks from sibling `modelN` directories
- eventually decide whether each block is initialized from PCD, RoMa, or both

This is the likely place where extension-block policy will live.

### Geometry/state mutation layer

Primary file:

- `frankenstein_base/scene/gaussian_model.py`

Responsibilities:

- continue to own Gaussian append / concat semantics
- support any optimizer rebuild or tensor bookkeeping needed after RoMa appends

Potential need:

- confirm that current append helpers are sufficient for repeated post-construction initialization

### Local initialization helper layer

Suggested future files:

- `frankenstein_base/roma_init/corr_init.py`
- `frankenstein_base/roma_init/matching.py`
- `frankenstein_base/roma_init/triangulation.py`

Responsibilities:

- select reference and neighbor cameras
- run RoMa
- sample and filter correspondences
- triangulate points
- convert points into Gaussian tensors

## Recommended import policy

Future `frankenstein_base` code should prefer:

- relative imports within `frankenstein_base`
- environment-installed third-party modules
- explicit local submodule paths only when documented and stable

Avoid:

- importing from `EDGS/source/*`
- importing from `gaussian-splatting/*`
- relying on the EDGS vendored `submodules/gaussian-splatting` tree at runtime

## Migration checklist for the future coding pass

1. Confirm how RoMa will be provided to `frankenstein_base`:
   local submodule or environment install.
2. Create a local initialization package under `frankenstein_base`.
3. Port the minimal EDGS fast path first.
4. Add CLI flags to `train_nomask.py`.
5. Integrate base-scene initialization.
6. Integrate extension-block initialization.
7. Update `run_exp.py` to emit RoMa-enabled experiment variants.
8. Replace these planning docs with implementation docs once behavior is real.
