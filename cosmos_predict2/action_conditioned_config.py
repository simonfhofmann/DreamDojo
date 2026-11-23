# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cosmos_predict2.config import (
    DEFAULT_NEGATIVE_PROMPT,
    CommonSetupArguments,
    Guidance,
    ModelKey,
    ModelVariant,
    get_model_literal,
)

DEFAULT_MODEL_KEY = ModelKey(variant=ModelVariant.ROBOT_ACTION_COND)


class ActionLoadFn(Protocol):
    def __call__(self, json_data: dict, video_path: str, args: "ActionConditionedInferenceArguments") -> dict: ...


class ActionConditionedSetupArguments(CommonSetupArguments):
    """Setup arguments for action-conditioned inference."""

    config_file: str = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"

    # Override defaults
    # pyrefly: ignore  # invalid-annotation
    model: get_model_literal(ModelVariant.ROBOT_ACTION_COND) = DEFAULT_MODEL_KEY.name
    checkpoint_path: Path | None = Path("checkpoints/DreamDojo/2B_YAM_post-train/iter_000050000/model_ema_bf16.pt")

    checkpoints_dir: str = "checkpoints/exp1201/gr1/checkpoints"
    save_dir: str = "results/action2world"
    num_frames: int = 37
    num_samples: int = 100
    dataset_path: str = "datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot"
    data_split: str = "test"
    single_base_index: bool = False
    deterministic_uniform_sampling: bool = False
    checkpoint_interval: int = 2000
    infinite: bool = False


@dataclass
class ActionConditionedInferenceArguments():
    """Inference arguments for action-conditioned inference."""

    # Output parameters
    save_root: Path = Path("results/action2world")
    """Save root directory."""
    # Model parameters
    chunk_size: int = 12
    """Chunk size for action conditioning."""
    guidance: Guidance = 0
    """Guidance value."""
    resolution: str = "none"
    """Resolution of the video (H,W). By default it will use model trained resolution. 9:16"""

    # Dataset-specific parameters
    camera_id: int = 0
    """Camera ID to use from the dataset."""
    start: int = 0
    """Start index for processing files."""
    end: int = 100
    """End index for processing files."""
    fps_downsample_ratio: int = 1
    """FPS downsample ratio."""
    gripper_scale: float = 1.0
    """Gripper scale factor."""
    gripper_key: str = "continuous_gripper_state"
    """Key for gripper state in JSON data."""
    state_key: str = "state"
    """Key for robot state in JSON data."""

    # Inference options
    reverse: bool = False
    """Whether to reverse the video."""
    single_chunk: bool = False
    """Whether to process only a single chunk."""
    start_frame_idx: int = 0
    """Start frame index."""
    save_fps: int = 10
    """FPS for saving output videos."""
    zero_actions: bool = False
    """Whether to zero the actions."""

    num_latent_conditional_frames: int = 1
    """Number of latent conditional frames (0, 1 or 2)."""
    # pyrefly: ignore  # bad-override
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    """Custom negative prompt for classifier-free guidance."""
