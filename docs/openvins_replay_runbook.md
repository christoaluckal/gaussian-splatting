# OpenVINS Replay Runbook

## Scope

This is the canonical replay/runbook document for the `frankenstein` OpenVINS bridge work.

It replaces:

- `openvins_ws/src/open_vins/docs/ros2_table_02_runbook.md`
- `openvins_ws/src/open_vins/docs/tartanair_ros2_openvins_runbook.md`

The two active replay sources are:

- `bags/table_02_ros2`
- TartanAir-style sequences such as `/mnt/share/nas/tartan/CoalMine/Data_omni/P0000`

## Build And Source

From the OpenVINS workspace:

```zsh
cd /home/christoa/Workspace/splatting/frankenstein/openvins_ws
source /opt/ros/humble/setup.zsh
colcon build --event-handlers console_cohesion+ --packages-select ov_core ov_init ov_msckf ov_eval
source install/setup.zsh
```

If the workspace is already built, the two `source` commands are enough.

## Common Replay Notes

The replay-time Phase 1 exporter is controlled with:

- `export_phase1_packets:=true`
- `export_phase1_root:=<packet_export_root>`
- `export_phase1_include_sparse_tracks:=true`

Recommended packet export root in this workspace:

- `/home/christoa/Workspace/splatting/frankenstein/bags/tartanair_packets`

This matches the retained packet dataset already used by the mapper-side TartanAir sweep scripts.

Exporter outputs:

- `<packet_export_root>/packets.jsonl`
- `<packet_export_root>/images/cam*/frame_*.png`

Current packet contents include:

- timestamp
- frame id
- saved image path
- current intrinsics and distortion
- `camera_to_body`
- `body_to_camera`
- `body_to_world`
- `world_to_body`
- camera-IMU time offset
- pose covariance in position-orientation order
- sparse tracked features at the exported frame

Known replay limitation:

- there is still a shutdown-time crash path on `Ctrl-C` / node teardown

## Replay Source 1: `bags/table_02_ros2`

### Launch

Terminal 1:

```zsh
source /opt/ros/humble/setup.zsh
source /home/christoa/Workspace/splatting/frankenstein/openvins_ws/install/setup.zsh
ros2 launch ov_msckf subscribe.launch.py \
  config:=rpng_plane \
  use_stereo:=false \
  max_cameras:=1 \
  export_phase1_packets:=true \
  export_phase1_root:=/tmp/phase1_export
```

Terminal 2:

```zsh
source /opt/ros/humble/setup.zsh
ros2 bag play /home/christoa/Workspace/splatting/frankenstein/bags/table_02_ros2
```

### Why `max_cameras:=1` matters

`subscribe.launch.py` still defaults `max_cameras` to `2`.

For this bag, only the `cam0`-equivalent color stream is present, so replay should always use:

- `use_stereo:=false`
- `max_cameras:=1`

### Required fixes already in tree

1. IMU callback thread lifetime bug is fixed in the ROS visualizer path by capturing the message by value.
2. Mono conversion for `rgb8` is fixed so replayed color input is converted safely before tracking.
3. ROS2 bag metadata normalization is fixed so converted bags do not get rejected by `ros2 bag`.

### Converter

The ROS1 to ROS2 converter is:

- `openvins_ws/src/open_vins/ov_msckf/scripts/convert_ros1_bag_to_required_format.py`

Typical usage:

```zsh
python3 openvins_ws/src/open_vins/ov_msckf/scripts/convert_ros1_bag_to_required_format.py \
  --src bags/table_02.bag \
  --dst bags/table_02_ros2
```

## Replay Source 2: TartanAir

### Expected sequence layout

Example sequence root:

```zsh
/mnt/share/nas/tartan/CoalMine/Data_omni/P0000
```

The current path expects:

- `image_lcam_front.zip`
- `depth_lcam_front.zip`
- `imu.zip`

Required IMU arrays inside `imu.zip` or `imu/`:

- `cam_time.npy`
- `imu_time.npy`
- `acc.npy`
- `gyro.npy`
- `pos_global.npy`
- `ori_global.npy`

Optional but used when present:

- `vel_global.npy`

### Create a ROS2 bag

```zsh
cd /home/christoa/Workspace/splatting/frankenstein
source /opt/ros/humble/setup.zsh
source openvins_ws/install/setup.zsh

python3 openvins_ws/src/open_vins/ov_msckf/scripts/write_tartanair_ros2_bag.py \
  --sequence /mnt/share/nas/tartan/CoalMine/Data_omni/P0000 \
  --camera lcam_front \
  --output bags/tartanair_coalmine_p0000_ros2 \
  --force
```

Useful options:

- `--max-frames 100`
- `--camera-fx`, `--camera-fy`, `--camera-cx`, `--camera-cy`
- `--depth-scale`
- `--gt-output <path>`
- `--no-gt-output`

Generated bag contents:

- `/d455/color/image_raw` as `rgb8`
- `/d455/depth/image_raw` as `32FC1`
- `/d455/color/camera_info`
- `/d455/depth/camera_info`
- `/d455/imu`

Generated sidecar files:

- `<bag>/tartanair_manifest.json`
- `<bag>/tartanair_gt.csv` unless `--no-gt-output` is passed

### Check the bag

```zsh
source /opt/ros/humble/setup.zsh
ros2 bag info /home/christoa/Workspace/splatting/frankenstein/bags/tartanair_coalmine_p0000_ros2
```

For `CoalMine/Data_omni/P0000`, expected properties are approximately:

- duration: `69.5 s`
- `696` RGB images
- `696` depth images
- `6950` IMU messages
- `696` camera-info messages per camera-info topic

### Run OpenVINS on the TartanAir bag

Terminal 1:

```zsh
cd /home/christoa/Workspace/splatting/frankenstein/openvins_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh

ros2 launch ov_msckf subscribe.launch.py \
  config:=tartanair \
  use_stereo:=false \
  max_cameras:=1 \
  export_phase1_packets:=true \
  export_phase1_root:=/home/christoa/Workspace/splatting/frankenstein/bags/tartanair_packets
```

Terminal 2:

```zsh
source /opt/ros/humble/setup.zsh
ros2 bag play /home/christoa/Workspace/splatting/frankenstein/bags/tartanair_coalmine_p0000_ros2
```

Expected packet outputs:

- `bags/tartanair_packets/packets.jsonl`
- `bags/tartanair_packets/images/cam0/*.png`

The `tartanair` config uses dynamic initialization. This is required because these sequences begin while already moving.

Typical static-init failure if the wrong init mode is used:

```text
failed static init: no accel jerk detected, platform moving too much
```

### Ground-truth export and plotting

If the bag predates GT sidecar support, export GT directly:

```zsh
cd /home/christoa/Workspace/splatting/frankenstein

python3 openvins_ws/src/open_vins/ov_msckf/scripts/export_tartanair_gt.py \
  --sequence /mnt/share/nas/tartan/CoalMine/Data_omni/P0000 \
  --output bags/tartanair_packets/tartanair_gt.csv
```

Plot OpenVINS output against TartanAir GT:

```zsh
python3 openvins_ws/src/open_vins/ov_msckf/scripts/plot_tartanair_openvins_gt.py \
  --packets bags/tartanair_packets/packets.jsonl \
  --gt bags/tartanair_packets/tartanair_gt.csv \
  --output bags/tartanair_packets/openvins_vs_tartanair_gt_scatter3d.png
```

The plotter defaults to `--alignment se3`.

Other supported alignment modes:

- `--alignment origin`
- `--alignment none`

### Frame and calibration assumptions

The current TartanAir path assumes:

- camera stream: `lcam_front`
- TartanAir body/camera convention: forward-right-down
- IMU/body frame aligned with the TartanAir front camera/body frame
- camera and IMU timestamps synchronized
- `timeshift_cam_imu: 0.0`

The OpenVINS config handles the body/camera convention conversion expected by the estimator. Do not re-convert image or IMU axes unless the source data has already been transformed elsewhere.

### Current TartanAir limits

- the writer creates a ROS2 bag plus GT sidecar files, not a GT topic
- depth is written for mapper/evaluation use, but the current OpenVINS replay path consumes RGB and IMU only
- nominal camera intrinsics are approximate unless a sequence-specific calibration source is provided
- packet export starts after estimator initialization, so `packets.jsonl` begins later than the first bag timestamp
