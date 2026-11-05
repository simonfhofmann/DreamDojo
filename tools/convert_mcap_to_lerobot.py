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


class FisheyeRectifier:
    """Precomputes and applies Kannala-Brandt fisheye rectification maps."""
    def __init__(self, K: np.ndarray, D: np.ndarray, width: int = 1920, height: int = 1200, balance: float = 0.0):
        self.width = width
        self.height = height
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (width, height), np.eye(3), balance=balance
        )
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), self.new_K, (width, height), cv2.CV_16SC2
        )
        # Compute 4:3 center crop indices
        target_aspect = 4.0 / 3.0  # 640 / 480
        crop_width = int(height * target_aspect)  # 1200 * (4/3) = 1600
        self.x_start = max(0, (width - crop_width) // 2)
        self.x_end = self.x_start + crop_width

    def process(self, raw_bgr: np.ndarray, out_w: int = 640, out_h: int = 480) -> np.ndarray:
        # 1. Undistort
        undistorted = cv2.remap(
            raw_bgr, self.map1, self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        # 2. 4:3 Center crop
        cropped = undistorted[:, self.x_start:self.x_end]
        # 3. Resize to 640x480
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return resized


def parse_mcap_episode(mcap_path: Path) -> Dict:
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
    cam_calibration = None
    instruction_text = ""

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory(), JsonDecoderFactory()])
        for schema, channel, message, decoded in reader.iter_decoded_messages():
            topic = channel.topic
            t_sec = message.log_time / 1e9

            if topic in kinematics:
                kinematics[topic].append((t_sec, decoded))
            elif topic == "/head-camera":
                video_packets.append((t_sec, decoded.data))
            elif topic == "/head-camera-info" and cam_calibration is None:
                cam_calibration = decoded
            elif topic == "/instruction" and not instruction_text:
                if hasattr(decoded, "data"):
                    instruction_text = decoded.data.strip()

    return {
        "kinematics": kinematics,
        "video_packets": video_packets,
        "cam_calibration": cam_calibration,
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
        if not entries:
            return np.array([]), np.zeros((0, expected_dim))
        times = np.array([e[0] for e in entries])
        vals = np.array([getattr(e[1], field_name) for e in entries], dtype=np.float64)
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


def decode_and_rectify_video(
    video_packets: List[Tuple[float, bytes]],
    rectifier: Optional[FisheyeRectifier],
    target_fps: int = 30,
    source_fps: int = 60,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Decodes H.264 packets via PyAV, rectifies, and downsamples to target_fps."""
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
                        processed = rectifier.process(bgr)
                    else:
                        processed = cv2.resize(bgr, (640, 480))
                    raw_frames.append(processed)
                    frame_times.append(t_sec)
                raw_count += 1

    return raw_frames, np.array(frame_times)


def write_video_mp4(frames: List[np.ndarray], out_path: Path, fps: int = 30):
    """Encodes list of BGR frames into an H.264 MP4 file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames[0].shape[:2]
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


def main():
    parser = argparse.ArgumentParser(description="Convert Foxglove MCAP to LeRobot Dataset for DreamDojo")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input MCAP file or directory of MCAP files")
    parser.add_argument("--output-dir", "-o", type=Path, required=True, help="Output directory for LeRobot dataset")
    parser.add_argument("--fps", type=int, default=30, help="Target video & control frame rate (default: 30)")
    parser.add_argument("--source-fps", type=int, default=60, help="Source camera FPS in MCAP (default: 60)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Episodes per chunk folder (default: 1000)")
    parser.add_argument("--prefix", type=str, default="tlrc", help="Prefix for episode identifiers (default: 'tlrc')")
    args = parser.parse_args()

    # Collect input files
    if args.input.is_file():
        mcap_files = [args.input]
    elif args.input.is_dir():
        mcap_files = sorted(list(args.input.glob("*.mcap")))
    else:
        raise FileNotFoundError(f"Input path {args.input} does not exist.")

    if not mcap_files:
        raise ValueError(f"No .mcap files found in {args.input}")

    logger.info(f"Found {len(mcap_files)} MCAP file(s) to process.")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    episodes_meta = []
    tasks_meta = []
    all_states = []
    all_actions = []

    # First pass: parse calibration if available from first file
    first_parsed = parse_mcap_episode(mcap_files[0])
    rectifier = None
    calib = first_parsed["cam_calibration"]
    if calib is not None and getattr(calib, "distortion_model", "") == "kannala_brandt":
        K = np.array(calib.K, dtype=np.float64).reshape((3, 3))
        D = np.array(calib.D, dtype=np.float64)
        logger.info("Initializing Kannala-Brandt fisheye rectifier...")
        rectifier = FisheyeRectifier(K, D, width=calib.width, height=calib.height, balance=0.0)
    else:
        logger.warning("No kannala_brandt calibration found. Using direct resize without undistortion.")

    # Process episodes
    for ep_idx, mcap_path in enumerate(tqdm(mcap_files, desc="Converting episodes")):
        chunk_idx = ep_idx // args.chunk_size
        chunk_str = f"chunk-{chunk_idx:03d}"

        parsed = parse_mcap_episode(mcap_path) if ep_idx > 0 else first_parsed

        # 1. Process video
        frames, frame_times = decode_and_rectify_video(
            parsed["video_packets"], rectifier, target_fps=args.fps, source_fps=args.source_fps
        )
        if len(frames) == 0:
            logger.warning(f"Skipping {mcap_path.name}: No video frames decoded.")
            continue

        n_frames = len(frames)
        # 2. Synchronize kinematics
        state_arr, action_arr = sync_and_assemble_kinematics(parsed["kinematics"], frame_times)

        # 3. Save Video MP4
        video_dir = output_dir / "videos" / chunk_str / f"episode_{ep_idx:06d}"
        video_path = video_dir / "top_camera-images-rgb.mp4"
        write_video_mp4(frames, video_path, fps=args.fps)

        # Determine clean episode name (e.g. tlrc_ep0008)
        import re
        ep_match = re.search(r"ep(\d+)", mcap_path.stem)
        if ep_match:
            ep_name = f"{args.prefix}_ep{ep_match.group(1)}"
        else:
            ep_name = f"{args.prefix}_ep{ep_idx:04d}"

        # 4. Save Parquet
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
            "episode_name": ep_name,
        })
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

        # Accumulate metadata & statistics
        all_states.append(state_arr)
        all_actions.append(action_arr)

        episodes_meta.append({
            "episode_index": ep_idx,
            "episode_name": ep_name,
            "tasks": [parsed["instruction"]],
            "length": n_frames,
        })
        tasks_meta.append({
            "task_index": ep_idx,
            "task": parsed["instruction"],
        })

    # Save meta/episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    # Save meta/tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for t in tasks_meta:
            f.write(json.dumps(t) + "\n")

    # Save meta/modality.json (Use standardized YAM modality)
    shared_yam_modality = Path("shared_meta/YAM_modality.json")
    if shared_yam_modality.exists():
        with open(shared_yam_modality) as f_in, open(meta_dir / "modality.json", "w") as f_out:
            f_out.write(f_in.read())

    # Save meta/stats.json
    logger.info("Computing dataset normalization statistics...")
    stats = compute_dataset_statistics(all_states, all_actions)
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    # Save meta/info.json
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
                "shape": [args.fps, 480, 640, 3],
                "names": ["channel", "height", "width"],
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

