# DreamDojo Data Tools

This directory contains data conversion and preprocessing utilities to prepare custom robot recordings for training and inference with DreamDojo.

---

## 1. `convert_mcap_to_lerobot.py`

Converts Foxglove MCAP teleoperation files from bimanual robots into the standardized **LeRobot format** expected by DreamDojo's `VideoActionDataset` and `configs/2b_480_640_yam.yaml`.

### What it does:
1. **Fisheye Rectification & Preprocessing:**
   - Reads calibration intrinsics ($K$) and distortion ($D$) from the `/head-camera-info` topic (`kannala_brandt` model).
   - Undistorts the $1920 \times 1200$ raw fisheye stream using `cv2.fisheye`.
   - Center-crops to a $4:3$ aspect ratio ($1600 \times 1200$) and resizes to native DreamDojo resolution ($640 \times 480$).
2. **Video Encoding & Temporal Resampling:**
   - Decodes H.264 packets via PyAV.
   - Downsamples 60 Hz camera recordings to **30 FPS** (standard YAM frame rate).
   - Encodes `top_camera-images-rgb.mp4` directly into `videos/chunk-000/episode_XXXXXX/`.
3. **Kinematics Synchronization:**
   - Synchronizes the ~295 Hz robot state/action streams to the exact 30 FPS video timestamps using linear interpolation.
4. **Vector Assembly (46 dimensions):**
   - Left EE pose (16-dim row-major $4 \times 4$ matrix)
   - Right EE pose (16-dim row-major $4 \times 4$ matrix)
   - Left & Right gripper positions (scalar $[0, 1]$)
   - Left & Right 6-DOF joint angles (radians)
5. **Full Dataset Structure Generation:**
   - Generates chunked Parquet files (`data/chunk-000/episode_XXXXXX.parquet`).
   - Generates `meta/modality.json`, `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl`.
   - Computes normalization statistics (`meta/stats.json`) across the dataset.

---

### Usage

#### Convert a Single MCAP Episode:
```bash
python tools/convert_mcap_to_lerobot.py \
    --input /path/to/20261029_144001_ep0008.mcap \
    --output-dir datasets/my_yam_dataset
```

#### Convert an Entire Directory of MCAPs:
```bash
python tools/convert_mcap_to_lerobot.py \
    --input /path/to/mcap_folder/ \
    --output-dir datasets/my_yam_dataset \
    --fps 30 \
    --source-fps 60
```

---

### Verifying with DreamDojo's Dataloader

Once converted, verify that DreamDojo loads your dataset with zero warnings or errors:

```bash
python -c "
from groot_dreams.dataloader import VideoActionDataset

ds = VideoActionDataset(
    dataset_path='datasets/my_yam_dataset',
    num_frames=13,
    data_split='train',
    embodiment='yam',
)
print('Dataset length:', len(ds))
sample = ds[0]
print('Video tensor shape:', sample['video'].shape)   # Expected: (C, 13, 480, 640)
print('Action tensor shape:', sample['action'].shape) # Expected: (12, 384)
print('Successfully verified!')
"
```

