# Colmap Splitter Behavior

## Scope

This document describes the scene-layout contract consumed by `frankenstein_base`.

It is not a full design note for the splitter implementation. It only captures the assumptions the training pipeline currently makes.

## Required layout

The split-scene pipeline assumes directories like:

- `<scene>_base/model0`
- `<scene>_split2/model0`
- `<scene>_split3/model0`

Within a split scene parent directory, `frankenstein_base/scene/__init__.py` expects sibling folders:

- `model0`
- `model1`
- `model2`
- and so on

`model0` is the initial block.

`model1+` are eager extension blocks loaded during scene construction.

## Base-scene expectations

Base-scene training expects:

- source path points to `model0`
- parent directory name ends with `_base`

The runner only warns when the naming convention is missing, but split inference depends on the split naming pattern.

## Split-scene expectations

Split-scene training expects:

- source path points to `model0`
- parent directory name matches `<scene>_splitN`

The runner uses that `N` to infer how many extension blocks should exist.

For example:

- `home_split2/model0` means one extension block is expected
- `home_split3/model0` means two extension blocks are expected

## Scene loading behavior

`Scene.__init__(...)` does the following for split scenes:

1. load `model0`
2. preload all extension blocks immediately
3. store extension cameras in `self.extension_set`
4. store extension Gaussian sets in `self.x_gauss`

This means split training is eager, not lazy.

## Extension behavior

When `scene.extend()` is triggered later in training:

- extension train cameras are appended into the active camera lists
- extension test cameras are appended into the active test camera lists
- extension Gaussians are merged through `concat_new_gaussian(...)`

## EDGS interaction

When EDGS init is off:

- each block is initialized from its COLMAP point cloud only

When EDGS init is on:

- `model0` can receive EDGS initialization
- extension blocks can also receive EDGS initialization when `--edgs_init_extensions` is enabled

## Known limitation

The split scene loader assumes the sibling block directories exist.

If a scene path does not actually have the expected `model1`, `model2`, and so on, split-scene loading will fail.

This is why plain one-folder scenes such as `./scene` did not work for the split-oriented pipeline, while `home_split2/model0` did.
