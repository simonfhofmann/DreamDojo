"""
Verify exact 1-to-1 frame alignment between dataloader `video[:, t]` and exported `robot_q[t]`
on `tlrc_bimanual_yam_eval6`, plus Wan2.1 latent-frame alignment (17 -> 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from groot_dreams.dataloader import MultiVideoActionDataset
from groot_dreams.geometry.dkx_kinematics import (
    DKXKinematics,
    align_robot_q_to_latents,
    extract_robot_q_from_yam_state,
)


def test_dataloader_robot_q_exact_frame_alignment():
    ds = MultiVideoActionDataset(
        num_frames=17,
        dataset_path=["datasets/tlrc_bimanual_yam_eval6"],
        data_split="full",
        camera_layout="three_panel_1to1",
        cr1_embeddings_path="datasets/cr1_empty_text_embeddings.pt",
    )
    le = ds.datasets[0].lerobot_dataset

    # Find a mid-episode sample from bimanual episode 3
    ep3_indices = [i for i, (tid, bidx) in enumerate(le.all_steps) if int(tid) == 3 and int(bidx) == 396]
    assert len(ep3_indices) == 1
    idx = ep3_indices[0]
    sample = ds[idx]

    assert "robot_q" in sample, "Missing robot_q in MultiVideoActionDataset output"
    video = sample["video"]  # [C, 17, 480, 640] uint8
    robot_q = sample["robot_q"]  # [17, 14] float32
    assert video.shape == (3, 17, 480, 640), f"Unexpected video shape: {video.shape}"
    assert robot_q.shape == (17, 14), f"Unexpected robot_q shape: {robot_q.shape}"
    assert torch.isfinite(robot_q).all()

    # Verify frame t=0 matches parquet row `base_index + 4 = 400`, and frame t=k matches `400 + 4*k`!
    df = pd.read_parquet("datasets/tlrc_bimanual_yam_eval6/data/chunk-000/episode_000003.parquet")
    expected_rows = [396 + 4 * (k + 1) for k in range(17)]
    raw_st = np.stack([df["observation.state"].iloc[r] for r in expected_rows], axis=0)
    expected_q = torch.from_numpy(extract_robot_q_from_yam_state(raw_st)).float()
    max_diff = (robot_q - expected_q).abs().max().item()
    assert max_diff == 0.0, f"robot_q mismatch vs parquet rows {expected_rows}: max_diff={max_diff}"

    # Verify delta_actions consistency: first 4 actions correspond to steps 400 -> 404, 408, 412, 416
    # and latent alignment produces [5, 14] with exact t_lat=0 == robot_q[0] and t_lat=1..4 == mean of 4-frame blocks
    kin = DKXKinematics()
    plk = kin.compute_plucker_rays(robot_q.unsqueeze(0), target_t_lat=5)
    assert plk["plucker"].shape == (1, 5, 30, 40, 6)
    assert torch.allclose(plk["q_lat"][0, 0], robot_q[0], atol=1e-6)
    assert torch.allclose(plk["q_lat"][0, 1], robot_q[1:5].mean(dim=0), atol=1e-6)
    assert torch.allclose(plk["q_lat"][0, 4], robot_q[13:17].mean(dim=0), atol=1e-6)

    # Also verify non-trivial motion across the 5 latent frames
    wrist_motion_m = (plk["c2w"][0, -1, 1, :3, 3] - plk["c2w"][0, 0, 1, :3, 3]).norm().item()
    assert wrist_motion_m > 0.005, f"Expected moving left wrist over 2.1s clip, got {wrist_motion_m:.4f} m"
    print(
        f"[PASS] Dataloader robot_q exact alignment verified: shape={tuple(robot_q.shape)}, "
        f"parquet_diff={max_diff:.1f}, plucker={tuple(plk['plucker'].shape)}, "
        f"left_wrist_motion={wrist_motion_m*100:.2f} cm"
    )


if __name__ == "__main__":
    test_dataloader_robot_q_exact_frame_alignment()
