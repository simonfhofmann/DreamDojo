"""
Vectorized DKX Bimanual Robot Kinematics, 3-Panel Plucker Rays, and Pairwise 3D Epipolar Geometry.

Verified against `assets/dkx.urdf` and `assets/20261029_144001_ep0008.mcap` to 0.00 mm FK error
and pixel-accurate 3-view projection on `tlrc_bimanual_yam_eval6`.

Convention for `robot_q` (14-D):
    robot_q[..., 0:6]  = left_joint1 .. left_joint6 (rad)
    robot_q[..., 6:12] = right_joint1 .. right_joint6 (rad)
    robot_q[..., 12]   = left_gripper normalized in [0, 1] (0 = open -> -0.045 m, 1 = closed -> 0.0 m)
    robot_q[..., 13]   = right_gripper normalized in [0, 1]

Token layout for `three_panel_1to1` (`480 x 640` canvas, Wan2.1 8x VAE + 2x DiT patchify = 16x16 px/token):
    Token grid: 30 x 40 = 1,200 tokens per latent frame (zero straddling tokens across panel boundaries):
      - head (view_id=0):        rows [0, 18),  cols [0, 40)  -> 720 tokens (640 x 288 panel)
      - left_wrist (view_id=1):  rows [18, 30), cols [0, 20)  -> 240 tokens (320 x 192 panel)
      - right_wrist (view_id=2): rows [18, 30), cols [20, 40) -> 240 tokens (320 x 192 panel)
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF_PATH = _REPO_ROOT / "assets" / "dkx.urdf"
DEFAULT_INTRINSICS_PATH = _REPO_ROOT / "assets" / "dkx_camera_intrinsics.json"

JOINT_ORDER_14: List[str] = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "left_gripper_left",
    "right_gripper_left",
]

CAMERA_NAMES: Tuple[str, str, str] = ("head", "left_wrist", "right_wrist")
CAMERA_OPTICAL_FRAMES: Tuple[str, str, str] = (
    "head_camera_optical_frame",
    "left_wrist_camera_optical_frame",
    "right_wrist_camera_optical_frame",
)

# 16 kinematic link frames ready for Exp 4b and 3D multi-view evaluation
KINEMATIC_LINK_NAMES: Tuple[str, ...] = (
    "left_link1-2",
    "left_link2-3",
    "left_link3-4",
    "left_link4-5",
    "left_link5-6",
    "left_link6-7",
    "left_tool0",
    "left_finger_left",
    "right_link1-2",
    "right_link2-3",
    "right_link3-4",
    "right_link4-5",
    "right_link5-6",
    "right_link6-7",
    "right_tool0",
    "right_finger_left",
)


def _rpy_to_mat_np(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def parse_dkx_urdf(urdf_path: str | Path = DEFAULT_URDF_PATH) -> List[Dict]:
    """Parse `dkx.urdf` into a topologically ordered list of joint specs starting from `base`."""
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    raw_joints: List[Dict] = []
    for j in root.findall("joint"):
        name = j.attrib["name"]
        jtype = j.attrib["type"]
        parent = j.find("parent").attrib["link"]
        child = j.find("child").attrib["link"]
        orig = j.find("origin")
        xyz = [float(v) for v in orig.attrib.get("xyz", "0 0 0").split()] if orig is not None else [0.0, 0.0, 0.0]
        rpy = [float(v) for v in orig.attrib.get("rpy", "0 0 0").split()] if orig is not None else [0.0, 0.0, 0.0]
        ax_el = j.find("axis")
        axis = [float(v) for v in ax_el.attrib["xyz"].split()] if ax_el is not None else [1.0, 0.0, 0.0]
        ax_norm = math.sqrt(sum(a * a for a in axis))
        if ax_norm > 0:
            axis = [a / ax_norm for a in axis]
        lim_el = j.find("limit")
        lower = float(lim_el.attrib["lower"]) if lim_el is not None and "lower" in lim_el.attrib else -math.pi
        upper = float(lim_el.attrib["upper"]) if lim_el is not None and "upper" in lim_el.attrib else math.pi
        mimic_el = j.find("mimic")
        mimic = None
        if mimic_el is not None:
            mimic = (
                mimic_el.attrib["joint"],
                float(mimic_el.attrib.get("multiplier", 1.0)),
                float(mimic_el.attrib.get("offset", 0.0)),
            )
        t0 = np.eye(4, dtype=np.float64)
        t0[:3, :3] = _rpy_to_mat_np(*rpy)
        t0[:3, 3] = xyz
        raw_joints.append(
            {
                "name": name,
                "type": jtype,
                "parent": parent,
                "child": child,
                "T0": t0,
                "axis": np.asarray(axis, dtype=np.float64),
                "lower": lower,
                "upper": upper,
                "mimic": mimic,
            }
        )

    # Topological sort from `base`
    visited = {"base"}
    ordered: List[Dict] = []
    remaining = list(raw_joints)
    while remaining:
        progressed = False
        next_remaining = []
        for j in remaining:
            if j["parent"] in visited:
                ordered.append(j)
                visited.add(j["child"])
                progressed = True
            else:
                next_remaining.append(j)
        if not progressed:
            raise RuntimeError(f"Disconnected URDF joints: {[j['name'] for j in next_remaining]}")
        remaining = next_remaining
    return ordered


def compute_panel_intrinsics(
    intrinsics_path: str | Path = DEFAULT_INTRINSICS_PATH,
    camera_layout: str = "three_panel_1to1",
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Compute exact rectified pinhole 3x3 intrinsics `K_panel` for each panel in `camera_layout`
    (`head`, `left_wrist`, `right_wrist`) without requiring OpenCV at runtime.
    """
    cfg = json.loads(Path(intrinsics_path).read_text())
    raw_w, raw_h = cfg["raw_resolution"]
    layout = cfg["layouts"][camera_layout]
    canvas_h, canvas_w = layout["canvas_hw"]
    top_h = layout["top_height"]
    bot_h = canvas_h - top_h
    half_w = canvas_w // 2

    panel_specs = {
        "head": (layout["head_crop_norm"], (canvas_w, top_h)),
        "left_wrist": (layout["wrist_crop_norm"], (half_w, bot_h)),
        "right_wrist": (layout["wrist_crop_norm"], (canvas_w - half_w, bot_h)),
    }

    k_panels: Dict[str, np.ndarray] = {}
    for cam_name, (crop_norm, (out_w, out_h)) in panel_specs.items():
        cam_cfg = cfg["cameras"][cam_name]
        mid_w, mid_h = cam_cfg["dataset_resolution"]
        k_rect = np.asarray(cam_cfg["K_rect_1920x1200"], dtype=np.float64)

        # Step 1: resize 1920x1200 -> dataset video resolution (mid_w x mid_h)
        k_mid = k_rect.copy()
        k_mid[0, :] *= mid_w / float(raw_w)
        k_mid[1, :] *= mid_h / float(raw_h)

        # Step 2: VideoThreePanelComposite._crop_and_resize exact integer crop + scale
        x1_f, y1_f, x2_f, y2_f = crop_norm
        cx1 = max(0, min(mid_w - 1, int(round(x1_f * mid_w))))
        cy1 = max(0, min(mid_h - 1, int(round(y1_f * mid_h))))
        cx2 = max(cx1 + 1, min(mid_w, int(round(x2_f * mid_w))))
        cy2 = max(cy1 + 1, min(mid_h, int(round(y2_f * mid_h))))
        sx_o = out_w / float(cx2 - cx1)
        sy_o = out_h / float(cy2 - cy1)

        k_panel = k_mid.copy()
        k_panel[0, 2] -= cx1
        k_panel[1, 2] -= cy1
        k_panel[0, :] *= sx_o
        k_panel[1, :] *= sy_o
        k_panels[cam_name] = k_panel

    return k_panels, cfg


def extract_robot_q_from_yam_state(state_46: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Extract unnormalized 14-D `robot_q` from raw 46-D TLRC/DKX `observation.state` (or `action`):
        state[..., 32]    = left_gripper
        state[..., 33]    = right_gripper
        state[..., 34:40] = left_joint1..6 (rad)
        state[..., 40:46] = right_joint1..6 (rad)
    Returns `[..., 14]` ordered as `[q_left(6), q_right(6), grip_left, grip_right]`.
    """
    if isinstance(state_46, torch.Tensor):
        return torch.cat([state_46[..., 34:40], state_46[..., 40:46], state_46[..., 32:34]], dim=-1)
    arr = np.asarray(state_46, dtype=np.float32)
    return np.concatenate([arr[..., 34:40], arr[..., 40:46], arr[..., 32:34]], axis=-1)


def align_robot_q_to_latents(
    robot_q: torch.Tensor,
    target_t_lat: int,
    temporal_compression_ratio: int = 4,
) -> torch.Tensor:
    """
    Align pixel-frame joint trajectory `robot_q` to Wan2.1 causal VAE latent frames.

    Wan2.1 VAE (`WanVAE_.encode`) maps `T_pix = 1 + r * (T_lat - 1)` pixel frames -> `T_lat` latents:
      - latent 0       <- pixel frame [0]
      - latent k (k>0) <- pixel frames [r*(k-1)+1 .. r*k]  (averaged to represent the exact block center)

    If `robot_q` already has `T == target_t_lat`, it is returned as-is.
    """
    if robot_q.ndim == 2:
        robot_q = robot_q.unsqueeze(0)
        squeeze_b = True
    else:
        squeeze_b = False

    b, t_in, d = robot_q.shape
    if t_in == target_t_lat:
        out = robot_q
    elif target_t_lat == 1:
        out = robot_q[:, :1, :]
    else:
        expected_pix = 1 + temporal_compression_ratio * (target_t_lat - 1)
        if t_in == expected_pix:
            q0 = robot_q[:, :1, :]
            q_rest = robot_q[:, 1:, :].reshape(b, target_t_lat - 1, temporal_compression_ratio, d).mean(dim=2)
            out = torch.cat([q0, q_rest], dim=1)
        elif t_in > expected_pix:
            q_trim = robot_q[:, :expected_pix, :]
            q0 = q_trim[:, :1, :]
            q_rest = q_trim[:, 1:, :].reshape(b, target_t_lat - 1, temporal_compression_ratio, d).mean(dim=2)
            out = torch.cat([q0, q_rest], dim=1)
        else:
            raise ValueError(
                f"Cannot align robot_q with T={t_in} to target_t_lat={target_t_lat} "
                f"(expected {target_t_lat} or >= {expected_pix})."
            )
    return out.squeeze(0) if squeeze_b else out


def build_rollout_robot_q(
    q_gt: np.ndarray | torch.Tensor,
    mode: str = "gt",
    joint_lower: Optional[np.ndarray | torch.Tensor] = None,
    joint_upper: Optional[np.ndarray | torch.Tensor] = None,
) -> np.ndarray | torch.Tensor:
    """
    Construct counterfactual rollout-time `robot_q` trajectories for `gt`, `zero`, or `negated` modes.
      - `gt`:      returns `q_gt` unchanged
      - `zero`:    holds initial pose `q_gt[..., 0:1, :]` constant across all timesteps
      - `negated`: reflects joint/gripper deltas around `q_0`: `q_0 - (q_gt - q_0)`, clamped to URDF limits
    """
    is_torch = isinstance(q_gt, torch.Tensor)
    if mode == "gt":
        return q_gt.clone() if is_torch else np.array(q_gt, copy=True)

    if mode == "zero":
        if is_torch:
            q0 = q_gt[..., :1, :]
            return q0.expand_as(q_gt).clone()
        q0 = q_gt[..., :1, :]
        return np.broadcast_to(q0, q_gt.shape).copy()

    if mode == "negated":
        if is_torch:
            q0 = q_gt[..., :1, :]
            q_neg = q0 - (q_gt - q0)
            if joint_lower is not None and joint_upper is not None:
                lo = torch.as_tensor(joint_lower, device=q_neg.device, dtype=q_neg.dtype)
                hi = torch.as_tensor(joint_upper, device=q_neg.device, dtype=q_neg.dtype)
                q_neg = torch.clamp(q_neg, min=lo, max=hi)
            return q_neg
        q0 = q_gt[..., :1, :]
        q_neg = q0 - (q_gt - q0)
        if joint_lower is not None and joint_upper is not None:
            lo = np.asarray(joint_lower, dtype=q_neg.dtype)
            hi = np.asarray(joint_upper, dtype=q_neg.dtype)
            q_neg = np.clip(q_neg, lo, hi)
        return q_neg

    raise ValueError(f"Unsupported rollout mode: {mode!r} (expected 'gt', 'zero', or 'negated').")


def _axis_angle_to_rotmat(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Vectorized Rodrigues rotation formula.
    Args:
        axis:  [3] unit vector
        theta: [...] angles in radians (float32)
    Returns:
        R:     [..., 3, 3]
    """
    c = torch.cos(theta)
    s = torch.sin(theta)
    one_m_c = 1.0 - c
    ux, uy, uz = axis[0], axis[1], axis[2]

    r00 = c + ux * ux * one_m_c
    r01 = ux * uy * one_m_c - uz * s
    r02 = ux * uz * one_m_c + uy * s

    r10 = uy * ux * one_m_c + uz * s
    r11 = c + uy * uy * one_m_c
    r12 = uy * uz * one_m_c - ux * s

    r20 = uz * ux * one_m_c - uy * s
    r21 = uz * uy * one_m_c + ux * s
    r22 = c + uz * uz * one_m_c

    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def compute_pairwise_epipolar_geometry(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    view_ids: torch.Tensor,
    workspace_min: torch.Tensor,
    workspace_max: torch.Tensor,
    sin_parallel_thresh: float = 1e-2,
    max_d_ray: float = 1.5,
) -> Dict[str, torch.Tensor]:
    """
    Compute numerically stable pairwise 3D epipolar geometry for all token pairs `(m, n)` per frame.

    Uses pure batched GEMMs (no `[BT, N, N, 3]` intermediate tensors) for low memory and <0.3 ms latency on A100.

    Args:
        rays_o:        [BT, N, 3] ray origins in `base` frame (float32)
        rays_d:        [BT, N, 3] unit ray directions in `base` frame (float32)
        view_ids:      [N] integer view IDs in {0, 1, 2}
        workspace_min: [3] lower corner of DKX tabletop workspace box in `base` frame
        workspace_max: [3] upper corner of DKX tabletop workspace box in `base` frame
        sin_parallel_thresh: threshold on ||d_m x d_n|| below which we fall back to point-to-line distance
        max_d_ray:     upper clamp (meters) on skew-ray distance

    Returns:
        dict with `[BT, N, N]` float32 tensors:
          - `d_ray`:         closest 3D ray-to-ray distance (m), clamped to `[0, max_d_ray]`, parallel-safe
          - `cos_theta`:     `d_m · d_n` in `[-1, 1]`
          - `sdf_ws`:        continuous signed distance (m) from closest-approach midpoint to workspace box (`[-0.5, 1.5]`)
          - `min_z`:         `min(t_m, t_n)` signed depth along both rays (m), clamped to `[-0.5, 1.5]`
          - `same_view`:     `[N, N]` bool mask (`True` when `view_m == view_n`)
          - `view_pair_id`:  `[N, N]` long tensor in `{0..5}` for unordered view pair `(min(v_m,v_n), max(v_m,v_n))`
    """
    rays_o = rays_o.float()
    rays_d = rays_d.float()
    rays_m = torch.cross(rays_o, rays_d, dim=-1)  # [BT, N, 3] Plucker moments

    # 1. cos(theta) = d_m · d_n
    cos_theta = torch.bmm(rays_d, rays_d.transpose(1, 2)).clamp(-1.0, 1.0)  # [BT, N, N]
    sin_sq = (1.0 - cos_theta * cos_theta).clamp(min=0.0)
    sin_theta = torch.sqrt(sin_sq)

    # 2. Exact Plucker reciprocal product: d_m · m_n + m_m · d_n = (o_n - o_m) · (d_m x d_n)
    recip = torch.bmm(rays_d, rays_m.transpose(1, 2)) + torch.bmm(rays_m, rays_d.transpose(1, 2))

    # 3. Dot products with origin separation w = o_m - o_n
    o_sq = (rays_o * rays_o).sum(dim=-1)  # [BT, N]
    oo = (o_sq.unsqueeze(2) + o_sq.unsqueeze(1) - 2.0 * torch.bmm(rays_o, rays_o.transpose(1, 2))).clamp(min=0.0)
    o_dot_d = (rays_o * rays_d).sum(dim=-1)  # [BT, N]
    w_dot_dm = o_dot_d.unsqueeze(2) - torch.bmm(rays_d, rays_o.transpose(1, 2))  # (o_m - o_n) · d_m
    w_dot_dn = torch.bmm(rays_o, rays_d.transpose(1, 2)) - o_dot_d.unsqueeze(1)  # (o_m - o_n) · d_n

    # 4. Skew-ray distance with smooth fallback to point-to-line distance when d_m || d_n
    d_skew = recip.abs() / sin_theta.clamp(min=sin_parallel_thresh)
    d_par = torch.sqrt((oo - w_dot_dm * w_dot_dm).clamp(min=0.0))
    is_parallel = sin_theta < sin_parallel_thresh
    d_ray = torch.where(is_parallel, d_par, d_skew).clamp(min=0.0, max=max_d_ray)

    # 5. Closest-approach ray depths (t_m, t_n) and 3D midpoint p*_mn
    denom = sin_sq.clamp(min=sin_parallel_thresh * sin_parallel_thresh)
    t_m = (cos_theta * w_dot_dn - w_dot_dm) / denom
    t_n = (w_dot_dn - cos_theta * w_dot_dm) / denom
    # For near-parallel rays, evaluate midpoint at nominal 0.5 m depth
    t_m = torch.where(is_parallel, torch.full_like(t_m, 0.5), t_m).clamp(-2.0, 4.0)
    t_n = torch.where(is_parallel, torch.full_like(t_n, 0.5), t_n).clamp(-2.0, 4.0)
    min_z = torch.minimum(t_m, t_n).clamp(-0.5, 1.5)

    # 6. Exact continuous 3D signed distance from p*_mn to workspace box [workspace_min, workspace_max]
    # Compute axis-by-axis so we never allocate [BT, N, N, 3]
    ws_center = 0.5 * (workspace_min + workspace_max)
    ws_half = 0.5 * (workspace_max - workspace_min)
    out_sq_sum = torch.zeros_like(d_ray)
    max_q = torch.full_like(d_ray, -1e9)
    for ax in range(3):
        p_ax = 0.5 * (
            rays_o[:, :, ax].unsqueeze(2)
            + t_m * rays_d[:, :, ax].unsqueeze(2)
            + rays_o[:, :, ax].unsqueeze(1)
            + t_n * rays_d[:, :, ax].unsqueeze(1)
        )
        q_ax = (p_ax - ws_center[ax]).abs() - ws_half[ax]
        q_pos = q_ax.clamp(min=0.0)
        out_sq_sum = out_sq_sum + q_pos * q_pos
        max_q = torch.maximum(max_q, q_ax)
    sdf_ws = (torch.sqrt(out_sq_sum) + max_q.clamp(max=0.0)).clamp(-0.5, 1.5)

    # 7. Same-view mask and symmetric view-pair ID in {0..5}
    v_m = view_ids.unsqueeze(1)  # [N, 1]
    v_n = view_ids.unsqueeze(0)  # [1, N]
    same_view = v_m == v_n
    # Unordered pair encoding for 3 views:
    # (0,0)->0, (1,1)->1, (2,2)->2, (0,1)/(1,0)->3, (0,2)/(2,0)->4, (1,2)/(2,1)->5
    v_lo = torch.minimum(v_m, v_n)
    v_hi = torch.maximum(v_m, v_n)
    view_pair_id = torch.where(
        same_view,
        v_lo,
        torch.where(v_lo == 0, v_hi + 2, torch.full_like(v_lo, 5)),
    )

    return {
        "d_ray": d_ray,
        "cos_theta": cos_theta,
        "sdf_ws": sdf_ws,
        "min_z": min_z,
        "same_view": same_view,
        "view_pair_id": view_pair_id,
    }


class DKXKinematics(nn.Module):
    """
    Meta-device-safe, vectorized PyTorch kinematics and 3-panel Plucker ray module for the DKX robot.

    All geometric constants are stored as non-persistent buffers and populated inside `init_buffers()`,
    so `with torch.device("meta"): ...` -> `to_empty(device)` -> `init_weights()` works cleanly.
    """

    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_URDF_PATH,
        intrinsics_path: str | Path = DEFAULT_INTRINSICS_PATH,
        camera_layout: str = "three_panel_1to1",
        patch_stride_px: int = 16,
    ) -> None:
        super().__init__()
        self.urdf_path = str(urdf_path)
        self.intrinsics_path = str(intrinsics_path)
        self.camera_layout = camera_layout
        self.patch_stride_px = int(patch_stride_px)

        # Parse URDF topology & intrinsics metadata (pure Python, safe under torch.device("meta"))
        self._joints = parse_dkx_urdf(self.urdf_path)
        self._k_panels_np, self._intrinsics_cfg = compute_panel_intrinsics(self.intrinsics_path, self.camera_layout)

        layout = self._intrinsics_cfg["layouts"][self.camera_layout]
        self.canvas_h, self.canvas_w = int(layout["canvas_hw"][0]), int(layout["canvas_hw"][1])
        self.top_h = int(layout["top_height"])
        assert self.canvas_h % self.patch_stride_px == 0
        assert self.canvas_w % self.patch_stride_px == 0
        assert self.top_h % self.patch_stride_px == 0
        assert (self.canvas_w // 2) % self.patch_stride_px == 0

        self.tok_h = self.canvas_h // self.patch_stride_px  # 30
        self.tok_w = self.canvas_w // self.patch_stride_px  # 40
        self.top_tok_h = self.top_h // self.patch_stride_px  # 18
        self.half_tok_w = self.tok_w // 2  # 20

        # Pre-index joint specs for fast PyTorch forward pass
        self._joint_name_to_q_idx = {name: i for i, name in enumerate(JOINT_ORDER_14)}
        self._num_joints = len(self._joints)

        # Register non-persistent buffers (populated in `init_buffers`)
        self.register_buffer("joint_T0", torch.empty(self._num_joints, 4, 4, dtype=torch.float32), persistent=False)
        self.register_buffer("joint_axis", torch.empty(self._num_joints, 3, dtype=torch.float32), persistent=False)
        self.register_buffer("joint_lower_14", torch.empty(14, dtype=torch.float32), persistent=False)
        self.register_buffer("joint_upper_14", torch.empty(14, dtype=torch.float32), persistent=False)
        self.register_buffer("d_cam_grid", torch.empty(self.tok_h, self.tok_w, 3, dtype=torch.float32), persistent=False)
        self.register_buffer("view_id_grid", torch.empty(self.tok_h, self.tok_w, dtype=torch.long), persistent=False)
        self.register_buffer("workspace_min", torch.empty(3, dtype=torch.float32), persistent=False)
        self.register_buffer("workspace_max", torch.empty(3, dtype=torch.float32), persistent=False)
        self.register_buffer("K_panels", torch.empty(3, 3, 3, dtype=torch.float32), persistent=False)

        if self.joint_T0.device.type != "meta":
            self.init_buffers()

    def _apply(self, fn, recurse: bool = True):
        """Allow device moves (cpu -> cuda) while keeping all geometric constants in exact float32."""
        super()._apply(fn, recurse=recurse)
        if self.joint_T0.device.type != "meta" and self.joint_T0.dtype != torch.float32:
            for name, buf in list(self._buffers.items()):
                if buf is not None and buf.is_floating_point() and buf.dtype != torch.float32:
                    self._buffers[name] = buf.float()
            self.init_buffers()
        return self

    @torch.no_grad()
    def init_buffers(self) -> None:
        """Populate all constant geometric buffers in-place (safe after `to_empty(device)`)."""
        device = self.joint_T0.device
        if device.type == "meta":
            return

        t0_np = np.stack([j["T0"] for j in self._joints], axis=0).astype(np.float32)
        ax_np = np.stack([j["axis"] for j in self._joints], axis=0).astype(np.float32)
        self.joint_T0.copy_(torch.from_numpy(t0_np).to(device))
        self.joint_axis.copy_(torch.from_numpy(ax_np).to(device))

        # Joint limits for 14-D robot_q (12 revolute rad + 2 normalized grippers in [0, 1])
        j_by_name = {j["name"]: j for j in self._joints}
        lower_14 = [j_by_name[n]["lower"] for n in JOINT_ORDER_14[:12]] + [0.0, 0.0]
        upper_14 = [j_by_name[n]["upper"] for n in JOINT_ORDER_14[:12]] + [1.0, 1.0]
        self.joint_lower_14.copy_(torch.tensor(lower_14, dtype=torch.float32, device=device))
        self.joint_upper_14.copy_(torch.tensor(upper_14, dtype=torch.float32, device=device))

        # Workspace box bounds in `base` frame
        ws = self._intrinsics_cfg["workspace_box_base"]
        self.workspace_min.copy_(torch.tensor(ws["min"], dtype=torch.float32, device=device))
        self.workspace_max.copy_(torch.tensor(ws["max"], dtype=torch.float32, device=device))

        # Panel K matrices [3, 3, 3] ordered (head, left_wrist, right_wrist)
        k_stack = np.stack([self._k_panels_np[c] for c in CAMERA_NAMES], axis=0).astype(np.float32)
        self.K_panels.copy_(torch.from_numpy(k_stack).to(device))

        # Exact token-center camera-frame unit rays for each panel (zero boundary straddling)
        stride = float(self.patch_stride_px)
        offset = 0.5 * (stride - 1.0)  # 7.5 px for 16x16 cell [0..15]

        d_cam = np.zeros((self.tok_h, self.tok_w, 3), dtype=np.float32)
        v_id = np.zeros((self.tok_h, self.tok_w), dtype=np.int64)

        def _fill_panel(r0: int, r1: int, c0: int, c1: int, cam_idx: int) -> None:
            K = k_stack[cam_idx]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            rs = np.arange(r1 - r0, dtype=np.float64) * stride + offset
            cs = np.arange(c1 - c0, dtype=np.float64) * stride + offset
            uu, vv = np.meshgrid(cs, rs)  # [H_p, W_p]
            x = (uu - cx) / fx
            y = (vv - cy) / fy
            z = np.ones_like(x)
            dirs = np.stack([x, y, z], axis=-1)
            dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
            d_cam[r0:r1, c0:c1] = dirs.astype(np.float32)
            v_id[r0:r1, c0:c1] = cam_idx

        _fill_panel(0, self.top_tok_h, 0, self.tok_w, 0)  # head
        _fill_panel(self.top_tok_h, self.tok_h, 0, self.half_tok_w, 1)  # left_wrist
        _fill_panel(self.top_tok_h, self.tok_h, self.half_tok_w, self.tok_w, 2)  # right_wrist

        self.d_cam_grid.copy_(torch.from_numpy(d_cam).to(device))
        self.view_id_grid.copy_(torch.from_numpy(v_id).to(device))

    def _ensure_buffers_on_device(self, device: torch.device) -> None:
        if self.joint_T0.device != device:
            self.to(device)
            self.init_buffers()

    def forward_kinematics(self, robot_q: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute `base`-frame 4x4 homogeneous poses for all URDF links.

        Args:
            robot_q: [..., 14] tensor `[q_left(6), q_right(6), grip_left, grip_right]`

        Returns:
            dict mapping link_name -> `[..., 4, 4]` float32 tensor in `base` frame.
        """
        self._ensure_buffers_on_device(robot_q.device)
        q = robot_q.float()
        lead_shape = q.shape[:-1]
        q_flat = q.reshape(-1, 14)
        n = q_flat.shape[0]
        device = q_flat.device

        # Convert normalized gripper [0, 1] -> prismatic displacement [-0.045, 0.0] m
        grip_l_disp = -0.045 * (1.0 - q_flat[:, 12].clamp(0.0, 1.0))
        grip_r_disp = -0.045 * (1.0 - q_flat[:, 13].clamp(0.0, 1.0))
        q_by_name: Dict[str, torch.Tensor] = {}
        for i in range(12):
            q_by_name[JOINT_ORDER_14[i]] = q_flat[:, i]
        q_by_name["left_gripper_left"] = grip_l_disp
        q_by_name["right_gripper_left"] = grip_r_disp
        for j in self._joints:
            if j["mimic"] is not None and j["name"] not in q_by_name:
                mj, mul, off = j["mimic"]
                q_by_name[j["name"]] = q_by_name[mj] * mul + off

        eye4 = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).expand(n, 4, 4)
        poses: Dict[str, torch.Tensor] = {"base": eye4}

        for idx, j in enumerate(self._joints):
            parent_pose = poses[j["parent"]]
            t0 = self.joint_T0[idx]  # [4, 4]
            jtype = j["type"]
            if jtype == "fixed":
                child_pose = torch.matmul(parent_pose, t0)
            elif jtype == "revolute":
                theta = q_by_name[j["name"]]
                rot = _axis_angle_to_rotmat(self.joint_axis[idx], theta)  # [N, 3, 3]
                t_motion = torch.zeros((n, 4, 4), dtype=torch.float32, device=device)
                t_motion[:, :3, :3] = rot
                t_motion[:, 3, 3] = 1.0
                child_pose = torch.matmul(torch.matmul(parent_pose, t0), t_motion)
            elif jtype == "prismatic":
                disp = q_by_name[j["name"]]  # [N]
                t_motion = eye4.clone()
                t_motion[:, :3, 3] = disp.unsqueeze(-1) * self.joint_axis[idx].unsqueeze(0)
                child_pose = torch.matmul(torch.matmul(parent_pose, t0), t_motion)
            else:
                raise ValueError(f"Unsupported URDF joint type: {jtype}")
            poses[j["child"]] = child_pose

        return {k: v.reshape(*lead_shape, 4, 4) for k, v in poses.items()}

    def compute_camera_poses(self, robot_q: torch.Tensor) -> torch.Tensor:
        """
        Compute `c2w` 4x4 camera poses in `base` frame for `(head, left_wrist, right_wrist)`.
        Returns: `[..., 3, 4, 4]` float32 tensor.
        """
        poses = self.forward_kinematics(robot_q)
        return torch.stack([poses[f] for f in CAMERA_OPTICAL_FRAMES], dim=-3)

    def compute_plucker_rays(
        self,
        robot_q: torch.Tensor,
        target_t_lat: Optional[int] = None,
        temporal_compression_ratio: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute per-latent-frame 3-panel 6D Plucker rays in the shared `base` frame.

        Args:
            robot_q: `[B, T_in, 14]` (either `T_in == target_t_lat` or `1 + 4*(target_t_lat - 1)`)
            target_t_lat: optional latent frame count `T_lat` to align to

        Returns:
            dict with:
              - `rays_o`:   `[B, T_lat, 30, 40, 3]` ray origins in `base` (m)
              - `rays_d`:   `[B, T_lat, 30, 40, 3]` unit ray directions in `base`
              - `plucker`:  `[B, T_lat, 30, 40, 6]` `[rays_d, rays_o x rays_d]`
              - `view_ids`: `[30, 40]` long tensor in `{0, 1, 2}`
              - `c2w`:      `[B, T_lat, 3, 4, 4]` camera-to-world poses
              - `q_lat`:    `[B, T_lat, 14]` aligned latent-frame joint poses
        """
        self._ensure_buffers_on_device(robot_q.device)
        if robot_q.ndim == 2:
            robot_q = robot_q.unsqueeze(0)
        q_lat = (
            align_robot_q_to_latents(robot_q, target_t_lat, temporal_compression_ratio)
            if target_t_lat is not None
            else robot_q
        )
        c2w = self.compute_camera_poses(q_lat)  # [B, T_lat, 3, 4, 4]
        b, t_lat = q_lat.shape[:2]

        rays_o = torch.empty((b, t_lat, self.tok_h, self.tok_w, 3), dtype=torch.float32, device=q_lat.device)
        rays_d = torch.empty((b, t_lat, self.tok_h, self.tok_w, 3), dtype=torch.float32, device=q_lat.device)

        panel_slices = (
            (slice(0, self.top_tok_h), slice(0, self.tok_w), 0),
            (slice(self.top_tok_h, self.tok_h), slice(0, self.half_tok_w), 1),
            (slice(self.top_tok_h, self.tok_h), slice(self.half_tok_w, self.tok_w), 2),
        )
        for rs, cs, cam_idx in panel_slices:
            rot = c2w[:, :, cam_idx, :3, :3]  # [B, T, 3, 3]
            pos = c2w[:, :, cam_idx, :3, 3]  # [B, T, 3]
            d_local = self.d_cam_grid[rs, cs]  # [Hp, Wp, 3]
            d_world = torch.einsum("btij,hwj->bthwi", rot, d_local)
            d_world = d_world / d_world.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            rays_d[:, :, rs, cs, :] = d_world
            rays_o[:, :, rs, cs, :] = pos[:, :, None, None, :]

        rays_m = torch.cross(rays_o, rays_d, dim=-1)
        plucker = torch.cat([rays_d, rays_m], dim=-1)
        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "plucker": plucker,
            "view_ids": self.view_id_grid,
            "c2w": c2w,
            "q_lat": q_lat,
        }

    def project_points_to_panel(
        self,
        pts_base: torch.Tensor,
        c2w_cam: torch.Tensor,
        cam_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D points in `base` frame into panel pixel coordinates `(u, v)` and camera depth `z`.

        Args:
            pts_base: `[..., P, 3]` points in `base` frame
            c2w_cam:  `[..., 4, 4]` camera-to-world pose for `cam_idx`
            cam_idx:  0=head, 1=left_wrist, 2=right_wrist
        Returns:
            uv: `[..., P, 2]` panel-local pixel coordinates
            z:  `[..., P]` depth along optical axis (m)
        """
        self._ensure_buffers_on_device(pts_base.device)
        rot = c2w_cam[..., :3, :3]
        pos = c2w_cam[..., :3, 3]
        # p_cam = R^T (p_base - t)
        diff = pts_base.float() - pos.unsqueeze(-2)
        p_cam = torch.einsum("...ji,...pj->...pi", rot, diff)
        z = p_cam[..., 2]
        K = self.K_panels[cam_idx]  # [3, 3]
        u = K[0, 0] * (p_cam[..., 0] / z.clamp(min=1e-6)) + K[0, 2]
        v = K[1, 1] * (p_cam[..., 1] / z.clamp(min=1e-6)) + K[1, 2]
        return torch.stack([u, v], dim=-1), z
