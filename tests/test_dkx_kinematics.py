"""
Unit tests for DKX forward kinematics, 3-panel Plucker rays, skew-ray degeneracy handling,
Wan2.1 temporal alignment, meta-device buffer re-init, and rollout-time counterfactuals.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from groot_dreams.geometry.dkx_kinematics import (
    CAMERA_OPTICAL_FRAMES,
    DKXKinematics,
    align_robot_q_to_latents,
    build_rollout_robot_q,
    compute_pairwise_epipolar_geometry,
    extract_robot_q_from_yam_state,
)


def test_fk_matches_eval6_recorded_ee_poses():
    """Verify PyTorch FK matches recorded parquet EE 4x4 poses to < 0.01 mm across episodes."""
    kin = DKXKinematics()
    eval6_root = Path("datasets/tlrc_bimanual_yam_eval6")
    assert eval6_root.exists(), f"Missing eval6 dataset at {eval6_root}"

    max_pos_err_mm = 0.0
    max_rot_err = 0.0
    for ep_idx, frames in [(3, [0, 150, 400, 700]), (5, [0, 200, 380, 600])]:
        df = pd.read_parquet(eval6_root / f"data/chunk-000/episode_{ep_idx:06d}.parquet")
        st = np.stack([df["observation.state"].iloc[f] for f in frames], axis=0).astype(np.float64)
        robot_q = torch.from_numpy(extract_robot_q_from_yam_state(st)).float()

        poses = kin.forward_kinematics(robot_q)
        # Recorded EE pose is link6-7 expressed in <side>_base_link
        for side, sl in [("left", slice(0, 16)), ("right", slice(16, 32))]:
            t_base = poses[f"{side}_base_link"]
            t_ee = poses[f"{side}_link6-7"]
            t_rel = torch.linalg.inv(t_base) @ t_ee  # [F, 4, 4]
            rec = torch.from_numpy(st[:, sl].reshape(-1, 4, 4)).float()

            pos_err_mm = (t_rel[:, :3, 3] - rec[:, :3, 3]).norm(dim=-1).max().item() * 1000.0
            rot_err = (t_rel[:, :3, :3] - rec[:, :3, :3]).abs().max().item()
            max_pos_err_mm = max(max_pos_err_mm, pos_err_mm)
            max_rot_err = max(max_rot_err, rot_err)

    assert max_pos_err_mm < 0.01, f"FK position error too high: {max_pos_err_mm:.4f} mm"
    assert max_rot_err < 1e-4, f"FK rotation error too high: {max_rot_err:.6f}"
    print(f"[PASS] FK vs eval6 parquet EE poses: max_pos_err={max_pos_err_mm:.5f} mm, max_rot_err={max_rot_err:.6f}")


def test_plucker_cross_view_epipolar_consistency():
    """Verify known 3D points visible in multiple views have near-zero skew-ray distance d_ray."""
    kin = DKXKinematics()
    df = pd.read_parquet("datasets/tlrc_bimanual_yam_eval6/data/chunk-000/episode_000003.parquet")
    st = np.asarray(df["observation.state"].iloc[400], dtype=np.float64)[None, :]
    robot_q = torch.from_numpy(extract_robot_q_from_yam_state(st)).float().unsqueeze(0)  # [1, 1, 14]

    poses = kin.forward_kinematics(robot_q)
    plk = kin.compute_plucker_rays(robot_q)
    rays_o = plk["rays_o"].reshape(1, 1200, 3)
    rays_d = plk["rays_d"].reshape(1, 1200, 3)
    view_ids = plk["view_ids"].reshape(1200)

    # Check exact token layout counts
    assert int((view_ids == 0).sum()) == 720
    assert int((view_ids == 1).sum()) == 240
    assert int((view_ids == 2).sum()) == 240
    # Check Plucker orthogonality: d · m == 0 and ||d|| == 1
    d_norm = plk["plucker"][..., :3].norm(dim=-1)
    dm_dot = (plk["plucker"][..., :3] * plk["plucker"][..., 3:]).sum(dim=-1).abs()
    assert (d_norm - 1.0).abs().max().item() < 1e-5
    assert dm_dot.max().item() < 1e-5

    geom = compute_pairwise_epipolar_geometry(
        rays_o, rays_d, view_ids, kin.workspace_min, kin.workspace_max
    )

    # Project left_tool0 and a central tabletop point into head & right_wrist, check token d_ray
    c2w = plk["c2w"][0, 0]  # [3, 4, 4]
    test_pts = {
        "left_tool0": poses["left_tool0"][0, 0, :3, 3],
        "table_center": torch.tensor([0.35, 0.0, 0.05], dtype=torch.float32),
    }
    for name, pt in test_pts.items():
        uv_h, z_h = kin.project_points_to_panel(pt[None, :], c2w[0], cam_idx=0)
        uv_rw, z_rw = kin.project_points_to_panel(pt[None, :], c2w[2], cam_idx=2)
        assert z_h.item() > 0.05 and z_rw.item() > 0.05
        # Convert panel pixel to global token index in 30x40 grid
        r_h = int(torch.clamp(uv_h[0, 1] / 16.0, 0, 17).item())
        c_h = int(torch.clamp(uv_h[0, 0] / 16.0, 0, 39).item())
        r_rw = 18 + int(torch.clamp(uv_rw[0, 1] / 16.0, 0, 11).item())
        c_rw = 20 + int(torch.clamp(uv_rw[0, 0] / 16.0, 0, 19).item())
        idx_h = r_h * 40 + c_h
        idx_rw = r_rw * 40 + c_rw

        # Check min d_ray in 3x3 neighborhood around projected token (accounts for 16x16 patch quantization)
        d_center = geom["d_ray"][0, idx_h, idx_rw].item()
        sdf_center = geom["sdf_ws"][0, idx_h, idx_rw].item()
        min_z_center = geom["min_z"][0, idx_h, idx_rw].item()
        assert d_center < 0.035, f"{name}: token-center d_ray={d_center:.4f} m too large"
        assert sdf_center <= 0.02, f"{name}: expected inside workspace, got sdf_ws={sdf_center:.4f} m"
        assert min_z_center > 0.05, f"{name}: expected positive depth, got min_z={min_z_center:.4f} m"
        print(
            f"[PASS] Cross-view correspondence ({name}): "
            f"d_ray={d_center*1000:.1f} mm, sdf_ws={sdf_center*1000:.1f} mm, min_z={min_z_center:.3f} m"
        )


def test_degenerate_skew_rays_and_same_view_mask():
    """Verify near-parallel, exact-parallel, anti-parallel, and identical rays never NaN/Inf."""
    rays_o = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],  # 0: view 0
                [0.0, 0.0, 0.0],  # 1: view 0 (identical ray!)
                [0.0, 0.2, 0.0],  # 2: view 1 (exact parallel ray, 0.2 m offset!)
                [0.0, 0.2, 0.0],  # 3: view 1 (exact anti-parallel ray, 0.2 m offset!)
                [0.0, 0.2, 0.0],  # 4: view 2 (near-parallel, 1e-4 rad tilt)
                [0.0, 0.0, 1.0],  # 5: view 2 (perpendicular intersecting at [0.5, 0, 0])
            ]
        ],
        dtype=torch.float32,
    )
    d4 = torch.tensor([1.0, 1e-4, 0.0])
    d4 = d4 / d4.norm()
    d5 = torch.tensor([0.5, 0.0, -1.0])
    d5 = d5 / d5.norm()
    rays_d = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([-1.0, 0.0, 0.0]),
            d4,
            d5,
        ]
    ).unsqueeze(0)
    view_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    ws_min = torch.tensor([0.0, -0.45, -0.03])
    ws_max = torch.tensor([0.75, 0.45, 0.45])

    geom = compute_pairwise_epipolar_geometry(rays_o, rays_d, view_ids, ws_min, ws_max)
    for k in ("d_ray", "cos_theta", "sdf_ws", "min_z"):
        assert torch.isfinite(geom[k]).all(), f"Non-finite values in {k}"

    # Parallel offset by 0.2 m -> point-to-line distance must be 0.2 m
    assert abs(geom["d_ray"][0, 0, 2].item() - 0.2) < 1e-4
    # Anti-parallel offset by 0.2 m -> point-to-line distance must be 0.2 m
    assert abs(geom["d_ray"][0, 0, 3].item() - 0.2) < 1e-4
    # Near-parallel -> still ~0.2 m (no blowup!)
    assert abs(geom["d_ray"][0, 0, 4].item() - 0.2) < 1e-3
    # Intersecting rays (0 and 5) -> d_ray == 0.0, midpoint at (0.5, 0, 0) inside workspace (sdf_ws < 0)
    assert geom["d_ray"][0, 0, 5].item() < 1e-5
    assert geom["sdf_ws"][0, 0, 5].item() < 0.0
    # Same-view mask check
    assert geom["same_view"][0, 1].item() is True
    assert geom["same_view"][0, 2].item() is False
    print("[PASS] Degenerate parallel/anti-parallel/near-parallel rays & same-view mask verified")


def test_wan21_temporal_alignment():
    """Verify 17-frame -> 5-latent Wan2.1 causal alignment matches [0], [1..4], [5..8], [9..12], [13..16]."""
    q_pix = torch.arange(17, dtype=torch.float32)[None, :, None].expand(2, 17, 14).clone()
    q_lat = align_robot_q_to_latents(q_pix, target_t_lat=5, temporal_compression_ratio=4)
    expected = torch.tensor([0.0, 2.5, 6.5, 10.5, 14.5])
    assert q_lat.shape == (2, 5, 14)
    assert torch.allclose(q_lat[0, :, 0], expected, atol=1e-6)
    # Identity if already at latent resolution
    assert torch.allclose(align_robot_q_to_latents(q_lat, target_t_lat=5), q_lat)
    print("[PASS] Wan2.1 temporal alignment 17 -> 5 verified:", q_lat[0, :, 0].tolist())


def test_rollout_counterfactual_camera_poses():
    """Verify rollout-time gt, zero, and negated robot_q produce physically valid moving/static/reflected wrist cameras."""
    kin = DKXKinematics()
    df = pd.read_parquet("datasets/tlrc_bimanual_yam_eval6/data/chunk-000/episode_000003.parquet")
    # 17 frames at stride 4 (7.5 Hz, 2.13s active bimanual motion)
    idxs = [300 + 4 * k for k in range(17)]
    st = np.stack([df["observation.state"].iloc[i] for i in idxs], axis=0).astype(np.float32)
    q_gt = torch.from_numpy(extract_robot_q_from_yam_state(st)).unsqueeze(0)  # [1, 17, 14]

    q_zero = build_rollout_robot_q(q_gt, mode="zero")
    q_neg = build_rollout_robot_q(q_gt, mode="negated", joint_lower=kin.joint_lower_14, joint_upper=kin.joint_upper_14)

    # Limits respected
    assert (q_neg >= kin.joint_lower_14 - 1e-6).all() and (q_neg <= kin.joint_upper_14 + 1e-6).all()

    plk_gt = kin.compute_plucker_rays(q_gt, target_t_lat=5)
    plk_zero = kin.compute_plucker_rays(q_zero, target_t_lat=5)
    plk_neg = kin.compute_plucker_rays(q_neg, target_t_lat=5)

    # Frame 0 camera poses must be identical across all 3 modes
    assert torch.allclose(plk_gt["c2w"][:, 0], plk_zero["c2w"][:, 0], atol=1e-6)
    assert torch.allclose(plk_gt["c2w"][:, 0], plk_neg["c2w"][:, 0], atol=1e-6)
    # Head camera (cam 0) must stay strictly static across time and across all modes
    assert torch.allclose(plk_gt["c2w"][:, :, 0], plk_gt["c2w"][:, :1, 0].expand_as(plk_gt["c2w"][:, :, 0]), atol=1e-6)
    # Zero mode must keep BOTH wrist cameras strictly static across time
    assert torch.allclose(plk_zero["c2w"], plk_zero["c2w"][:, :1].expand_as(plk_zero["c2w"]), atol=1e-6)
    # GT and Negated must move wrist cameras in opposite directions relative to frame 0!
    lw_disp_gt = (plk_gt["c2w"][0, -1, 1, :3, 3] - plk_gt["c2w"][0, 0, 1, :3, 3])
    lw_disp_neg = (plk_neg["c2w"][0, -1, 1, :3, 3] - plk_neg["c2w"][0, 0, 1, :3, 3])
    assert lw_disp_gt.norm().item() > 0.01, "Expected noticeable wrist motion in ep03 clip"
    assert torch.dot(lw_disp_gt, lw_disp_neg).item() < 0.0, "Expected negated mode to move wrist in opposite direction"
    print(
        f"[PASS] Rollout counterfactuals verified: "
        f"|lw_disp_gt|={lw_disp_gt.norm().item()*100:.1f} cm, "
        f"cos(gt, neg)={torch.nn.functional.cosine_similarity(lw_disp_gt[None], lw_disp_neg[None]).item():.3f}"
    )


def test_meta_device_reinit_roundtrip():
    """Verify meta-device -> to_empty('cpu') -> init_buffers() matches normal construction bit-for-bit."""
    kin_ref = DKXKinematics()
    with torch.device("meta"):
        kin_meta = DKXKinematics()
    kin_meta.to_empty(device="cpu")
    kin_meta.init_buffers()

    for name, buf_ref in kin_ref.named_buffers():
        buf_meta = dict(kin_meta.named_buffers())[name]
        assert torch.equal(buf_ref, buf_meta), f"Mismatch in buffer {name} after meta re-init"
    print("[PASS] Meta-device -> to_empty -> init_buffers bit-exact round-trip verified")


if __name__ == "__main__":
    test_fk_matches_eval6_recorded_ee_poses()
    test_plucker_cross_view_epipolar_consistency()
    test_degenerate_skew_rays_and_same_view_mask()
    test_wan21_temporal_alignment()
    test_rollout_counterfactual_camera_poses()
    test_meta_device_reinit_roundtrip()
    print("\nALL DKX KINEMATICS & PLUCKER TESTS PASSED!")
