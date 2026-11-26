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

"""Auto-start coverage for subprocesses without modifying application code.

This file is automatically imported by Python and enables coverage tracking
in distributed/multiprocess environments (e.g., torchrun) without requiring
changes to the application code.
"""

try:
    import coverage

    coverage.process_startup()
except ImportError:
    # Coverage not installed, skip
    pass

# Fallback to imageio-ffmpeg's bundled binary when system `ffmpeg` is absent
# (e.g., minimal HPC Apptainer containers). On workstations with system `ffmpeg`,
# this is a fast no-op and the system binary is preferred.
try:
    import os as _os
    import shutil as _shutil

    if _shutil.which("ffmpeg") is None:
        import imageio_ffmpeg as _imageio_ffmpeg

        _ffmpeg_exe = _imageio_ffmpeg.get_ffmpeg_exe()
        if _ffmpeg_exe and _os.path.isfile(_ffmpeg_exe):
            _os.environ.setdefault("IMAGEIO_FFMPEG_EXE", _ffmpeg_exe)
            try:
                import mediapy as _mediapy

                _mediapy.set_ffmpeg(_ffmpeg_exe)
            except ImportError:
                pass
except Exception:
    pass
