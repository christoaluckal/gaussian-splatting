# COLMAP Splitter Behavior

## Scope

This document describes the current behavior of the scripts in [`colmap_splitter`](/home/christoa/Workspace/splatting/gaussian-splatting/colmap_splitter):

- [`split.py`](/home/christoa/Workspace/splatting/gaussian-splatting/colmap_splitter/split.py)
- [`split_list.py`](/home/christoa/Workspace/splatting/gaussian-splatting/colmap_splitter/split_list.py)
- [`split_tree.py`](/home/christoa/Workspace/splatting/gaussian-splatting/colmap_splitter/split_tree.py)
- [`split_xyz.py`](/home/christoa/Workspace/splatting/gaussian-splatting/colmap_splitter/split_xyz.py)
- shared writer/parser utilities in [`common.py`](/home/christoa/Workspace/splatting/gaussian-splatting/colmap_splitter/common.py)

It is intentionally descriptive of the current implementation.

## Input scene contract

All splitter scripts expect a COLMAP text scene rooted at:

- `<scene>/sparse_txt/cameras.txt`
- `<scene>/sparse_txt/images.txt`
- `<scene>/sparse_txt/points3D.txt`

They also expect the source image tree at:

- `<scene>/images`

Optional multi-resolution image trees are copied when present:

- `<scene>/images_2`
- `<scene>/images_4`
- `<scene>/images_8`

The splitters do not read from `sparse/0`. They read from `sparse_txt`.

## Shared output behavior

All four scripts now share the same core COLMAP serialization path through [`common.py`](/home/christoa/Workspace/splatting/gaussian-splatting/colmap_splitter/common.py).

Current shared behavior:

- each output model is written under `<dst>/modelN/sparse/0`
- `images.txt`, `points3D.txt`, and `cameras.txt` are written for each model
- `test.txt` is written only when `--num_test > 0`
- image files are copied into each model-specific `images` directory and filtered to only the selected image names
- `images_2`, `images_4`, and `images_8` are filtered and copied too when they exist in the source scene

Current consistency rules:

- 3D points are assigned uniquely to the first split group that references them
- point tracks are pruned to only image IDs that belong to that group
- image observations that reference points outside the final group are rewritten to `-1`
- points with empty tracks after pruning are dropped from that model
- if test images are sampled, their `POINT3D_ID` entries are rewritten to `-1` in `test.txt`
- `points3D.txt` keeps only points that still appear in the train split for that model

This means the generated `images.txt` and `points3D.txt` stay mutually consistent instead of containing stale cross-group references.

## Script-specific behavior

### `split.py`

[`split.py`](/home/christoa/Workspace/splatting/gaussian-splatting/colmap_splitter/split.py) performs a sequential two-way split.

Arguments:

- `-s`: source scene
- `-m`: destination root
- `-m1`: first output model name, default `model0`
- `-m2`: second output model name, default `model1`
- `-f`: split frame image name
- `--num_test`: number of held-out test images per output model

Current behavior:

- images up to and including `-f` go to the first model
- images after `-f` go to the second model
- if `-f` is omitted, the last image is used, so the second model will typically be empty
- if the requested split frame is missing, the script raises an error

Example:

```bash
python colmap_splitter/split.py \
  -s /path/to/scene \
  -m /path/to/scene_split \
  -m1 model0 \
  -m2 model1 \
  -f frame_00042.png \
  --num_test 10
```

### `split_list.py`

[`split_list.py`](/home/christoa/Workspace/splatting/gaussian-splatting/colmap_splitter/split_list.py) performs a contiguous N-way split by image order.

Arguments:

- `-s`: source scene
- `-m`: destination root
- `--split_num`: number of output models
- `--default`: force a single output model regardless of `--split_num`
- `--num_test`: number of held-out test images per output model

Current behavior:

- images are split into contiguous blocks in file order from `images.txt`
- the split is as even as possible
- when the image count does not divide evenly, earlier groups receive one extra image
- output models are named `model0`, `model1`, `model2`, and so on
- `--default` collapses the output to just `model0`

Example:

```bash
python colmap_splitter/split_list.py \
  -s /path/to/scene \
  -m /path/to/scene_split3 \
  --split_num 3 \
  --num_test 5
```

### `split_tree.py`

[`split_tree.py`](/home/christoa/Workspace/splatting/gaussian-splatting/colmap_splitter/split_tree.py) groups images by camera-center proximity.

Arguments:

- `-s`: source scene
- `-m`: destination root
- `--dist`: camera-center neighborhood radius
- `--default`: force a single output model

Current behavior:

- camera centers are recovered from COLMAP poses in `images.txt`
- a KD-tree neighborhood query groups nearby cameras using the radius from `--dist`
- the first output group contains the anchor images
- each anchor neighborhood becomes its own additional output model
- if no non-trivial neighborhood is found, the script emits a single model containing all images
- `--default` also emits a single model containing all images

Current limitation:

- this clustering rule preserves the current script logic, but it is heuristic rather than a robust connected-components clustering pass

Example:

```bash
python colmap_splitter/split_tree.py \
  -s /path/to/scene \
  -m /path/to/scene_tree \
  --dist 0.1
```

### `split_xyz.py`

[`split_xyz.py`](/home/christoa/Workspace/splatting/gaussian-splatting/colmap_splitter/split_xyz.py) is the radial camera-layout splitter and remains the reference implementation.

Arguments:

- `-s`: source scene
- `-m`: destination root
- `--split_num`: number of radial wedges
- `--num_test`: number of held-out test images per output model

Current behavior:

- camera centers are computed from COLMAP poses
- centers are PCA-aligned before splitting
- aligned camera centers are projected onto the XY plane
- images are assigned to angular wedges around the mean projected center
- output models are named `model0`, `model1`, `model2`, and so on

Example:

```bash
python colmap_splitter/split_xyz.py \
  -s /path/to/scene \
  -m /path/to/scene_xyz4 \
  --split_num 4 \
  --num_test 5
```

## Relationship to the runner

[`run_exp.py`](/home/christoa/Workspace/splatting/gaussian-splatting/run_exp.py) assumes split scenes already exist.

Typical conventions used with the runner are:

- base scene: `<scene>_base/model0`
- split scene: `<scene>_splitN/model0`

The runner does not invoke these splitter scripts itself. It consumes scenes that were created ahead of time.
