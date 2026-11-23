#!/usr/bin/env python3
"""
MCAP to LeRobot Dataset Converter for Bimanual Robot (YAM-compatible format)
===========================================================================

Converts Foxglove MCAP recording(s) into the standard LeRobot format ready for 
DreamDojo post-training and inference.

Key features:
- Decodes H.264 video streams from `/head-camera` (and optional wrist cameras).
- Rectifies Kannala-Brandt fisheye distortion using `/head-camera-info` calibration.
- Center-crops to 4:3 and resizes to 480x640 to match DreamDojo native resolution.
- Downsamples 60 Hz video to 30 FPS (standard YAM frame rate).
- Synchronizes 295 Hz robot kinematics to 30 Hz video frame timestamps.
- Assembles 46-dim `observation.state` and `action` vectors matching YAM slot layout.
- Generates chunked Parquet files, MP4 videos, and complete `meta/` directory with stats.

Usage:
------
    # Convert a single MCAP file:
    python tools/convert_mcap_to_lerobot.py \\
        --input /path/to/recording.mcap \\
        --output-dir datasets/my_yam_robot

    # Convert an entire folder of MCAP files:
    python tools/convert_mcap_to_lerobot.py \\
        --input /path/to/mcap_folder \\
        --output-dir datasets/my_yam_robot
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import av
import cv2
import numpy as np
import pandas as pd
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("mcap_to_lerobot")


class JsonDecoderFactory:
    """Fallback decoder factory for JSON-encoded topics (e.g., leader status/buttons)."""
    def match(self, encoding: str) -> bool:
        return encoding == "json"

    def decoder_for(self, channel, schema):
        return lambda data: json.loads(data.decode("utf-8"))


DEFAULT_CROPS = {
    "head_crop": (480, 384, 1440, 1104),       # 960x720 (exact 4:3) -> 640x480
    "three_panel_head": (360, 420, 1560, 960), # 1200x540 (exact 2.222:1) -> 640x288
    "three_panel_wrist": (192, 278, 1728, 1200),  # 1536x922 (exact 5:3) -> 320x192
}


def parse_crop_box(spec: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not spec:
        return None
    parts = [int(x.strip()) for x in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Crop box must be 'x1,y1,x2,y2', got: {spec}")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop box (x2<=x1 or y2<=y1): {parts}")
    return (x1, y1, x2, y2)


class FisheyeRectifier:
    """Precomputes and applies Kannala-Brandt fisheye rectification maps."""
    def __init__(
        self,
        K: np.ndarray,
        D: np.ndarray,
        width: int = 1920,
        height: int = 1200,
        balance: float = 0.0,
        crop_box: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.width = width
        self.height = height
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (width, height), np.eye(3), balance=balance
        )
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), self.new_K, (width, height), cv2.CV_16SC2
        )
        if crop_box is not None:
            x1, y1, x2, y2 = crop_box
            self.x_start = max(0, min(width, x1))
            self.y_start = max(0, min(height, y1))
            self.x_end = max(self.x_start + 1, min(width, x2))
            self.y_end = max(self.y_start + 1, min(height, y2))
        else:
            # Default: 4:3 center crop
            target_aspect = 4.0 / 3.0  # 640 / 480
            crop_width = int(height * target_aspect)  # 1200 * (4/3) = 1600
            self.x_start = max(0, (width - crop_width) // 2)
            self.x_end = self.x_start + crop_width
            self.y_start = 0
            self.y_end = height

    def process(self, raw_bgr: np.ndarray, out_w: int = 640, out_h: int = 480) -> np.ndarray:
        # 1. Undistort
        undistorted = cv2.remap(
            raw_bgr, self.map1, self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        # 2. Crop ROI
        cropped = undistorted[self.y_start:self.y_end, self.x_start:self.x_end]
        # 3. Resize to target panel size
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return resized


def parse_mcap_episode(mcap_path: Path, include_wrists: bool = False) -> Dict:
    """Reads all relevant topics from an MCAP file into memory structures."""
    kinematics = {
        "/left-arm-state": [],
        "/right-arm-state": [],
        "/left-arm-action": [],
        "/right-arm-action": [],
        "/left-ee-state": [],
        "/right-ee-state": [],
        "/left-ee-action": [],
        "/right-ee-action": [],
    }
    video_packets = []
    left_wrist_packets = []
    right_wrist_packets = []
    cam_calibrations: Dict[str, object] = {}
    instruction_text = ""

    wanted_topics = list(kinematics.keys()) + ["/head-camera", "/head-camera-info", "/instruction"]
    if include_wrists:
        wanted_topics += [
            "/left-wrist-camera",
            "/right-wrist-camera",
            "/left-wrist-camera-info",
            "/right-wrist-camera-info",
        ]

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory(), JsonDecoderFactory()])
        for schema, channel, message, decoded in reader.iter_decoded_messages(topics=wanted_topics):
            topic = channel.topic
            t_sec = message.log_time / 1e9

            if topic in kinematics:
                kinematics[topic].append((t_sec, decoded))
            elif topic == "/head-camera":
                video_packets.append((t_sec, decoded.data))
            elif topic == "/left-wrist-camera":
                left_wrist_packets.append((t_sec, decoded.data))
            elif topic == "/right-wrist-camera":
                right_wrist_packets.append((t_sec, decoded.data))
            elif topic.endswith("-camera-info") and topic not in cam_calibrations:
                cam_calibrations[topic] = decoded
            elif topic == "/instruction" and not instruction_text:
                if hasattr(decoded, "data"):
                    instruction_text = decoded.data.strip()

    return {
        "kinematics": kinematics,
        "video_packets": video_packets,
        "left_wrist_packets": left_wrist_packets,
        "right_wrist_packets": right_wrist_packets,
        "cam_calibration": cam_calibrations.get("/head-camera-info"),
        "cam_calibrations": cam_calibrations,
        "instruction": instruction_text or "perform bimanual robot task",
    }


def sync_and_assemble_kinematics(
    kinematics: Dict, target_timestamps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synchronizes the ~300 Hz kinematics streams to the target video timestamps (~30 Hz)
    and concatenates them into 46-dimensional state and action vectors.

    Vector Layout (46 dimensions):
      [0:16]  Left EE pose (4x4 matrix row-major)
      [16:32] Right EE pose (4x4 matrix row-major)
      [32:33] Left gripper position (scalar [0, 1])
      [33:34] Right gripper position (scalar [0, 1])
      [34:40] Left arm joint angles (6 rad)
      [40:46] Right arm joint angles (6 rad)
    """
    n_frames = len(target_timestamps)
    state_arr = np.zeros((n_frames, 46), dtype=np.float64)
    action_arr = np.zeros((n_frames, 46), dtype=np.float64)

    def extract_time_series(topic: str, field_name: str, expected_dim: int):
        entries = kinematics.get(topic, [])
        valid = [
            (e[0], getattr(e[1], field_name))
            for e in entries
            if len(getattr(e[1], field_name, [])) == expected_dim
        ]
        if not valid:
            return np.array([]), np.zeros((0, expected_dim))
        times = np.array([v[0] for v in valid], dtype=np.float64)
        vals = np.array([v[1] for v in valid], dtype=np.float64)
        if vals.ndim == 1:
            vals = vals[:, None]
        return times, vals

    def interpolate_to_targets(times: np.ndarray, vals: np.ndarray) -> np.ndarray:
        if len(times) == 0:
            return np.zeros((n_frames, vals.shape[1] if vals.ndim > 1 else 1))
        out = np.zeros((n_frames, vals.shape[1]))
        for d in range(vals.shape[1]):
            out[:, d] = np.interp(target_timestamps, times, vals[:, d])
        return out

    # Left & Right arm state
    t, v = extract_time_series("/left-arm-state", "pose", 16)
    state_arr[:, 0:16] = interpolate_to_targets(t, v)

    t, v = extract_time_series("/right-arm-state", "pose", 16)
    state_arr[:, 16:32] = interpolate_to_targets(t, v)

    t, v = extract_time_series("/left-ee-state", "position", 1)
    state_arr[:, 32:33] = interpolate_to_targets(t, v)

    t, v = extract_time_series("/right-ee-state", "position", 1)
    state_arr[:, 33:34] = interpolate_to_targets(t, v)

    t, v = extract_time_series("/left-arm-state", "position", 6)
    state_arr[:, 34:40] = interpolate_to_targets(t, v)

    t, v = extract_time_series("/right-arm-state", "position", 6)
    state_arr[:, 40:46] = interpolate_to_targets(t, v)

    # Left & Right arm action
    t, v = extract_time_series("/left-arm-action", "pose", 16)
    action_arr[:, 0:16] = interpolate_to_targets(t, v)

    t, v = extract_time_series("/right-arm-action", "pose", 16)
    action_arr[:, 16:32] = interpolate_to_targets(t, v)

    t, v = extract_time_series("/left-ee-action", "position", 1)
    action_arr[:, 32:33] = interpolate_to_targets(t, v)

    t, v = extract_time_series("/right-ee-action", "position", 1)
    action_arr[:, 33:34] = interpolate_to_targets(t, v)

    t, v = extract_time_series("/left-arm-action", "position", 6)
    action_arr[:, 34:40] = interpolate_to_targets(t, v)

    t, v = extract_time_series("/right-arm-action", "position", 6)
    action_arr[:, 40:46] = interpolate_to_targets(t, v)

    return state_arr, action_arr


def build_rectifier(
    calib, crop_box: Optional[Tuple[int, int, int, int]] = None
) -> Optional[FisheyeRectifier]:
    if calib is not None and getattr(calib, "distortion_model", "") == "kannala_brandt":
        K = np.array(calib.K, dtype=np.float64).reshape((3, 3))
        D = np.array(calib.D, dtype=np.float64)
        return FisheyeRectifier(
            K, D, width=calib.width, height=calib.height, balance=0.0, crop_box=crop_box
        )
    return None


def decode_and_rectify_video(
    video_packets: List[Tuple[float, bytes]],
    rectifier: Optional[FisheyeRectifier],
    target_fps: int = 30,
    source_fps: int = 60,
    out_w: int = 640,
    out_h: int = 480,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Decodes H.264 packets via PyAV, rectifies/crops, and downsamples to target_fps."""
    codec = av.CodecContext.create("h264", "r")
    raw_frames = []
    frame_times = []

    downsample_factor = max(1, round(source_fps / target_fps))
    raw_count = 0

    for t_sec, packet_bytes in video_packets:
        packets = codec.parse(packet_bytes)
        for pkt in packets:
            for av_frame in codec.decode(pkt):
                if raw_count % downsample_factor == 0:
                    bgr = av_frame.to_ndarray(format="bgr24")
                    if rectifier is not None:
                        processed = rectifier.process(bgr, out_w=out_w, out_h=out_h)
                    else:
                        processed = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    raw_frames.append(processed)
                    frame_times.append(t_sec)
                raw_count += 1

    return raw_frames, np.array(frame_times)


def decode_and_sync_wrist_video(
    video_packets: List[Tuple[float, bytes]],
    rectifier: Optional[FisheyeRectifier],
    target_times: np.ndarray,
    out_w: int = 320,
    out_h: int = 192,
) -> Tuple[List[np.ndarray], float]:
    """
    Stream-decodes wrist H.264 packets and matches each target timestamp to its
    nearest decoded wrist frame without holding full-resolution frames in RAM.

    Returns:
        (synced_frames, max_sync_error_ms)
    """
    n_targets = len(target_times)
    best_dt = np.full(n_targets, np.inf, dtype=np.float64)
    best_frames: List[Optional[np.ndarray]] = [None] * n_targets

    codec = av.CodecContext.create("h264", "r")
    for t_sec, packet_bytes in video_packets:
        packets = codec.parse(packet_bytes)
        for pkt in packets:
            for av_frame in codec.decode(pkt):
                diffs = np.abs(target_times - t_sec)
                improved = np.flatnonzero(diffs < best_dt)
                if improved.size > 0:
                    bgr = av_frame.to_ndarray(format="bgr24")
                    if rectifier is not None:
                        processed = rectifier.process(bgr, out_w=out_w, out_h=out_h)
                    else:
                        processed = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    for idx in improved:
                        best_dt[idx] = diffs[idx]
                        best_frames[idx] = processed

    missing = sum(1 for f in best_frames if f is None)
    if missing > 0:
        raise RuntimeError(f"Wrist stream had {missing}/{n_targets} unmatched frames!")
    max_err_ms = float(np.max(best_dt)) * 1000.0
    return best_frames, max_err_ms  # type: ignore[return-value]


def write_video_mp4(frames: List[np.ndarray], out_path: Path, fps: int = 30, crf: int = 20):
    """Encodes list of BGR frames into a compact, fast-seeking H.264 MP4 via PyAV (with OpenCV fallback)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    try:
        container = av.open(str(out_path), mode="w")
        stream = container.add_stream("libx264", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "veryfast", "g": str(fps), "threads": "1"}
        for f in frames:
            av_fr = av.VideoFrame.from_ndarray(f, format="bgr24")
            for pkt in stream.encode(av_fr):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()
    except Exception as e:
        logger.warning(f"PyAV libx264 failed ({e}), falling back to OpenCV mp4v for {out_path}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()


def compute_dataset_statistics(all_states: List[np.ndarray], all_actions: List[np.ndarray]) -> Dict:
    """Computes mean, std, min, max, q01, q99 for observation.state and action."""
    cat_state = np.vstack(all_states)
    cat_action = np.vstack(all_actions)

    def stats_dict(arr: np.ndarray) -> Dict:
        return {
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "min": np.min(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }

    return {
        "observation.state": stats_dict(cat_state),
        "action": stats_dict(cat_action),
    }


def _convert_single_episode(
    ep_idx: int,
    mcap_path: Path,
    output_dir: Path,
    chunk_size: int,
    fps: int,
    source_fps: int,
    camera_mode: str,
    head_crop_box: Optional[Tuple[int, int, int, int]],
    wrist_crop_box: Optional[Tuple[int, int, int, int]],
) -> Optional[Dict]:
    """Converts a single MCAP episode to MP4(s) + Parquet and returns metadata/stats arrays."""
    cv2.setNumThreads(1)
    include_wrists = camera_mode in ("three_panel", "multicam")
    chunk_idx = ep_idx // chunk_size
    chunk_str = f"chunk-{chunk_idx:03d}"

    parsed = parse_mcap_episode(mcap_path, include_wrists=include_wrists)
    calibs = parsed["cam_calibrations"]

    err_l_ms = 0.0
    err_r_ms = 0.0
    video_dir = output_dir / "videos" / chunk_str / f"episode_{ep_idx:06d}"

    if camera_mode == "multicam":
        # Store full 16:10 rectified frames (no baked-in crop!) for all 3 cameras separately:
        #   top_camera: 1280x800 (so Column B crop is native 640x480 with zero upscaling)
        #   left/right wrist: 640x400 (so cropped wrists are >320x192 with zero upscaling)
        head_calib = calibs.get("/head-camera-info")
        full_head_box = (0, 0, head_calib.width, head_calib.height) if head_calib is not None else None
        head_rectifier = build_rectifier(head_calib, crop_box=head_crop_box or full_head_box)

        lw_calib = calibs.get("/left-wrist-camera-info")
        full_lw_box = (0, 0, lw_calib.width, lw_calib.height) if lw_calib is not None else None
        left_wrist_rectifier = build_rectifier(lw_calib, crop_box=wrist_crop_box or full_lw_box)

        rw_calib = calibs.get("/right-wrist-camera-info")
        full_rw_box = (0, 0, rw_calib.width, rw_calib.height) if rw_calib is not None else None
        right_wrist_rectifier = build_rectifier(rw_calib, crop_box=wrist_crop_box or full_rw_box)

        head_frames, frame_times = decode_and_rectify_video(
            parsed["video_packets"],
            head_rectifier,
            target_fps=fps,
            source_fps=source_fps,
            out_w=1280,
            out_h=800,
        )
        if len(head_frames) == 0:
            logger.warning(f"Skipping {mcap_path.name}: No head video frames decoded.")
            return None
        n_frames = len(head_frames)
        write_video_mp4(head_frames, video_dir / "top_camera-images-rgb.mp4", fps=fps)
        del head_frames

        lw_frames, err_l_ms = decode_and_sync_wrist_video(
            parsed["left_wrist_packets"],
            left_wrist_rectifier,
            target_times=frame_times,
            out_w=640,
            out_h=400,
        )
        write_video_mp4(lw_frames, video_dir / "left_camera-images-rgb.mp4", fps=fps)
        del lw_frames

        rw_frames, err_r_ms = decode_and_sync_wrist_video(
            parsed["right_wrist_packets"],
            right_wrist_rectifier,
            target_times=frame_times,
            out_w=640,
            out_h=400,
        )
        write_video_mp4(rw_frames, video_dir / "right_camera-images-rgb.mp4", fps=fps)
        del rw_frames
    elif camera_mode == "three_panel":
        head_rectifier = build_rectifier(calibs.get("/head-camera-info"), crop_box=head_crop_box)
        left_wrist_rectifier = build_rectifier(
            calibs.get("/left-wrist-camera-info"), crop_box=wrist_crop_box
        )
        right_wrist_rectifier = build_rectifier(
            calibs.get("/right-wrist-camera-info"), crop_box=wrist_crop_box
        )
        head_frames, frame_times = decode_and_rectify_video(
            parsed["video_packets"],
            head_rectifier,
            target_fps=fps,
            source_fps=source_fps,
            out_w=640,
            out_h=288,
        )
        if len(head_frames) == 0:
            logger.warning(f"Skipping {mcap_path.name}: No head video frames decoded.")
            return None
        lw_frames, err_l_ms = decode_and_sync_wrist_video(
            parsed["left_wrist_packets"],
            left_wrist_rectifier,
            target_times=frame_times,
            out_w=320,
            out_h=192,
        )
        rw_frames, err_r_ms = decode_and_sync_wrist_video(
            parsed["right_wrist_packets"],
            right_wrist_rectifier,
            target_times=frame_times,
            out_w=320,
            out_h=192,
        )
        frames = [
            np.vstack([h_fr, np.hstack([l_fr, r_fr])])
            for h_fr, l_fr, r_fr in zip(head_frames, lw_frames, rw_frames)
        ]
        assert frames[0].shape == (480, 640, 3), f"Unexpected composite shape: {frames[0].shape}"
        del head_frames, lw_frames, rw_frames
        n_frames = len(frames)
        write_video_mp4(frames, video_dir / "top_camera-images-rgb.mp4", fps=fps)
        del frames
    else:
        head_rectifier = build_rectifier(calibs.get("/head-camera-info"), crop_box=head_crop_box)
        frames, frame_times = decode_and_rectify_video(
            parsed["video_packets"],
            head_rectifier,
            target_fps=fps,
            source_fps=source_fps,
            out_w=640,
            out_h=480,
        )
        if len(frames) == 0:
            logger.warning(f"Skipping {mcap_path.name}: No video frames decoded.")
            return None
        n_frames = len(frames)
        write_video_mp4(frames, video_dir / "top_camera-images-rgb.mp4", fps=fps)
        del frames

    state_arr, action_arr = sync_and_assemble_kinematics(parsed["kinematics"], frame_times)

    # Save Parquet
    data_dir = output_dir / "data" / chunk_str
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"

    df = pd.DataFrame({
        "observation.state": [row for row in state_arr],
        "action": [row for row in action_arr],
        "timestamp": frame_times - frame_times[0],
        "frame_index": np.arange(n_frames),
        "episode_index": ep_idx,
        "task_index": ep_idx,
        "index": np.arange(n_frames),
    })
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

    return {
        "ep_idx": ep_idx,
        "mcap_name": mcap_path.name,
        "state_arr": state_arr,
        "action_arr": action_arr,
        "episode_meta": {
            "episode_index": ep_idx,
            "tasks": [parsed["instruction"]],
            "length": n_frames,
        },
        "task_meta": {
            "task_index": ep_idx,
            "task": parsed["instruction"],
        },
        "err_l_ms": err_l_ms,
        "err_r_ms": err_r_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert Foxglove MCAP to LeRobot Dataset for DreamDojo")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input MCAP file or directory of MCAP files (searched recursively)")
    parser.add_argument("--output-dir", "-o", type=Path, required=True, help="Output directory for LeRobot dataset")
    parser.add_argument("--fps", type=int, default=30, help="Target video & control frame rate (default: 30)")
    parser.add_argument("--source-fps", type=int, default=60, help="Source camera FPS in MCAP (default: 60)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Episodes per chunk folder (default: 1000)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel CPU worker processes (default: 1)")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Skip MCAP files already present in output-dir/meta/conversion_config.json and append only new ones.",
    )
    parser.add_argument(
        "--camera-mode",
        choices=["head", "head_crop", "three_panel", "multicam"],
        default="head",
        help=(
            "Camera layout mode: "
            "'multicam' = recommended zero-duplicate mode saving all 3 full-FOV rectified cameras separately "
            "(top 1280x800, left/right wrists 640x400) for on-the-fly cropping/tiling in the dataloader; "
            "'head' = full 4:3 rectified head camera (legacy default); "
            "'head_crop' = baked-in tabletop-cropped 4:3 head camera; "
            "'three_panel' = baked-in 3-panel 640x480 composite."
        ),
    )
    parser.add_argument(
        "--head-crop",
        type=str,
        default=None,
        help="Optional override for rectified head camera crop 'x1,y1,x2,y2' (in 1920x1200 rectified coords)",
    )
    parser.add_argument(
        "--wrist-crop",
        type=str,
        default=None,
        help="Optional override for rectified wrist camera crop 'x1,y1,x2,y2' (in 1920x1200 rectified coords)",
    )
    args = parser.parse_args()

    include_wrists = args.camera_mode in ("three_panel", "multicam")
    if args.head_crop is not None:
        head_crop_box = parse_crop_box(args.head_crop)
    elif args.camera_mode == "head_crop":
        head_crop_box = DEFAULT_CROPS["head_crop"]
    elif args.camera_mode == "three_panel":
        head_crop_box = DEFAULT_CROPS["three_panel_head"]
    else:
        head_crop_box = None

    if args.wrist_crop is not None:
        wrist_crop_box = parse_crop_box(args.wrist_crop)
    elif args.camera_mode == "three_panel":
        wrist_crop_box = DEFAULT_CROPS["three_panel_wrist"]
    else:
        wrist_crop_box = None

    # Collect input files recursively, sorted deterministically by filename
    if args.input.is_file():
        mcap_files = [args.input]
    elif args.input.is_dir():
        mcap_files = sorted(args.input.rglob("*.mcap"), key=lambda p: (p.name, str(p)))
    else:
        raise FileNotFoundError(f"Input path {args.input} does not exist.")

    if not mcap_files:
        raise ValueError(f"No .mcap files found in {args.input}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Support incremental appending of newly added MCAP files
    existing_mcap_names: List[str] = []
    existing_episodes_meta: List[Dict] = []
    existing_tasks_meta: List[Dict] = []
    prev_err_l_ms = 0.0
    prev_err_r_ms = 0.0
    conv_cfg_path = meta_dir / "conversion_config.json"

    if args.incremental and conv_cfg_path.exists() and (meta_dir / "episodes.jsonl").exists():
        prev_cfg = json.loads(conv_cfg_path.read_text())
        if prev_cfg.get("camera_mode") != args.camera_mode:
            raise ValueError(
                f"Cannot run --incremental: existing dataset has camera_mode={prev_cfg.get('camera_mode')}, "
                f"requested camera_mode={args.camera_mode}."
            )
        existing_mcap_names = list(prev_cfg.get("mcap_files", []))
        prev_err_l_ms = float(prev_cfg.get("max_sync_err_left_ms", 0.0))
        prev_err_r_ms = float(prev_cfg.get("max_sync_err_right_ms", 0.0))
        with open(meta_dir / "episodes.jsonl") as f:
            existing_episodes_meta = [json.loads(line) for line in f if line.strip()]
        if (meta_dir / "tasks.jsonl").exists():
            with open(meta_dir / "tasks.jsonl") as f:
                existing_tasks_meta = [json.loads(line) for line in f if line.strip()]

    existing_set = set(existing_mcap_names)
    new_mcap_files = [p for p in mcap_files if p.name not in existing_set]
    start_ep_idx = len(existing_episodes_meta)

    logger.info(
        f"Found {len(mcap_files)} total MCAP(s) | existing={len(existing_episodes_meta)} | "
        f"new_to_convert={len(new_mcap_files)} | mode={args.camera_mode} | workers={args.workers}"
    )

    if len(new_mcap_files) == 0 and len(existing_episodes_meta) > 0:
        logger.info("Dataset is already up to date (0 new MCAP files to convert).")
        return

    jobs = [(start_ep_idx + i, p) for i, p in enumerate(new_mcap_files)]
    results: List[Dict] = []
    if args.workers <= 1:
        for ep_idx, mcap_path in tqdm(jobs, desc="Converting episodes"):
            res = _convert_single_episode(
                ep_idx, mcap_path, output_dir, args.chunk_size,
                args.fps, args.source_fps, args.camera_mode, head_crop_box, wrist_crop_box,
            )
            if res is not None:
                if include_wrists:
                    logger.info(
                        f"Episode {res['ep_idx']:03d} ({res['mcap_name']}): {res['episode_meta']['length']} frames | "
                        f"max wrist sync err L={res['err_l_ms']:.1f}ms, R={res['err_r_ms']:.1f}ms"
                    )
                results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _convert_single_episode,
                    ep_idx, mcap_path, output_dir, args.chunk_size,
                    args.fps, args.source_fps, args.camera_mode, head_crop_box, wrist_crop_box,
                ): (ep_idx, mcap_path.name)
                for ep_idx, mcap_path in jobs
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Converting episodes"):
                res = fut.result()
                if res is not None:
                    results.append(res)

    results.sort(key=lambda r: r["ep_idx"])
    episodes_meta = existing_episodes_meta + [r["episode_meta"] for r in results]
    tasks_meta = existing_tasks_meta + [r["task_meta"] for r in results]
    all_mcap_names = existing_mcap_names + [r["mcap_name"] for r in results]
    max_sync_err_l_ms = max([prev_err_l_ms] + [r["err_l_ms"] for r in results])
    max_sync_err_r_ms = max([prev_err_r_ms] + [r["err_r_ms"] for r in results])
    if include_wrists:
        logger.info(
            f"Finished {len(results)} new {args.camera_mode} episodes (total={len(episodes_meta)}) | "
            f"worst wrist sync err: L={max_sync_err_l_ms:.1f}ms, R={max_sync_err_r_ms:.1f}ms"
        )

    # Recompute stats across ALL episodes (existing + newly added)
    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    if existing_episodes_meta:
        for ep in existing_episodes_meta:
            ei = ep["episode_index"]
            ci = ei // args.chunk_size
            pq = output_dir / "data" / f"chunk-{ci:03d}" / f"episode_{ei:06d}.parquet"
            df_old = pd.read_parquet(pq)
            all_states.append(np.vstack(df_old["observation.state"].to_list()))
            all_actions.append(np.vstack(df_old["action"].to_list()))
    all_states.extend(r["state_arr"] for r in results)
    all_actions.extend(r["action_arr"] for r in results)

    # Save meta/conversion_config.json for provenance
    conversion_meta = {
        "camera_mode": args.camera_mode,
        "head_crop": list(head_crop_box) if head_crop_box is not None else None,
        "wrist_crop": list(wrist_crop_box) if wrist_crop_box is not None else None,
        "fps": args.fps,
        "source_fps": args.source_fps,
        "num_mcap_files": len(all_mcap_names),
        "mcap_files": all_mcap_names,
    }
    if include_wrists:
        conversion_meta["max_sync_err_left_ms"] = round(max_sync_err_l_ms, 2)
        conversion_meta["max_sync_err_right_ms"] = round(max_sync_err_r_ms, 2)
    with open(conv_cfg_path, "w") as f:
        json.dump(conversion_meta, f, indent=4)

    # Save meta/episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    # Save meta/tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for t in tasks_meta:
            f.write(json.dumps(t) + "\n")

    # Save meta/modality.json
    shared_yam_modality = Path("shared_meta/YAM_modality.json")
    if shared_yam_modality.exists():
        with open(shared_yam_modality) as f_in:
            mod_data = json.load(f_in)
        if args.camera_mode == "multicam":
            mod_data["video"] = {
                "top_camera-images-rgb": {"original_key": "top_camera-images-rgb"},
                "left_camera-images-rgb": {"original_key": "left_camera-images-rgb"},
                "right_camera-images-rgb": {"original_key": "right_camera-images-rgb"},
            }
        else:
            mod_data["video"] = {
                "top_camera-images-rgb": {"original_key": "top_camera-images-rgb"}
            }
        with open(meta_dir / "modality.json", "w") as f_out:
            json.dump(mod_data, f_out, indent=4)

    # Save meta/stats.json
    logger.info("Computing dataset normalization statistics...")
    stats = compute_dataset_statistics(all_states, all_actions)
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    # Save meta/info.json
    top_shape = [800, 1280, 3] if args.camera_mode == "multicam" else [480, 640, 3]
    wrist_shape = [400, 640, 3] if args.camera_mode == "multicam" else [480, 640, 3]
    info = {
        "codebase_version": "v2.0",
        "robot_type": "bimanual_yam",
        "total_episodes": len(episodes_meta),
        "total_frames": sum(e["length"] for e in episodes_meta),
        "fps": args.fps,
        "chunks_size": args.chunk_size,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/episode_{episode_index:06d}/{video_key}.mp4",
        "features": {
            "observation.images.top_camera-images-rgb": {
                "dtype": "video",
                "shape": top_shape,
                "names": ["height", "width", "channel"],
                "info": {"video.fps": args.fps, "video.codec": "h264"},
            },
            "observation.images.right_camera-images-rgb": {
                "dtype": "video",
                "shape": wrist_shape,
                "names": ["height", "width", "channel"],
                "info": {"video.fps": args.fps, "video.codec": "h264"},
            },
            "observation.images.left_camera-images-rgb": {
                "dtype": "video",
                "shape": wrist_shape,
                "names": ["height", "width", "channel"],
                "info": {"video.fps": args.fps, "video.codec": "h264"},
            },
            "observation.state": {
                "dtype": "float64",
                "shape": [46],
                "names": ["state"],
            },
            "action": {
                "dtype": "float64",
                "shape": [46],
                "names": ["action"],
            },
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    logger.info(f"Dataset conversion complete! Successfully saved to: {output_dir}")


if __name__ == "__main__":
    main()

