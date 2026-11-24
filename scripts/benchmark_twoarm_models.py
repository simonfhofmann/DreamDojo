#!/usr/bin/env python3
"""
Apples-to-Apples Cross-Model Two-Arm Benchmark Suite
====================================================
Evaluates all DreamDojo models (Zero-Shot, v2 Uncropped, Exp 1 Cropped, Exp 2 Stage 1,
Exp 2 Stage 2) on the exact same held-out bimanual benchmark episodes (`twoarm`),
projecting every model's predicted overhead camera onto the EXACT SAME Shared Tabletop
ROI (x in [0.25, 0.75], y in [0.35, 0.80] in rectified 1920x1200 coords -> 270x480)
so metrics are 100% free of FOV bias.

Outputs:
  - Per-model Shared Tabletop ROI PSNR / SSIM / LPIPS (GT, Zero, Negated; Chunks 1..4)
  - Left-Half vs Right-Half Shared ROI PSNR (two-arm tracking balance)
  - Native Head / Left-Wrist / Right-Wrist breakdown for 3-panel models
  - 6-panel comparison videos [GT | ZeroShot | v2 | Exp1 | Exp2-4k | Exp2-Stg2] on Shared ROI
  - 3-panel native videos [GT | Exp2-4k | Exp2-Stg2]
  - Cached raw rollouts + summary.json in --save-dir (re-running skips already-evaluated models)
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapy
import numpy as np
import piq
import torch
import torchvision

# Normalized (x1, y1, x2, y2) in 1920x1200 / 1280x800 rectified coordinates covered by each layout's head view
LAYOUT_HEAD_BOUNDS: Dict[str, Tuple[float, float, float, float]] = {
    "head": (160.0 / 1920.0, 0.0, 1760.0 / 1920.0, 1.0),
    "head_crop": (0.25, 0.32, 0.75, 0.92),
    "three_panel": (0.1875, 0.35, 0.8125, 0.80),
    "three_panel_1to1": (0.25, 0.44, 0.75, 0.80),
}
# Exact spatial intersection of all 4 head views (native 1:1 288x640 slice on 1280x800 top camera)
SHARED_ROI_NORM: Tuple[float, float, float, float] = (0.25, 0.44, 0.75, 0.80)
SHARED_ROI_HW: Tuple[int, int] = (288, 640)

MODEL_SPECS = [
    {
        "id": "0_zeroshot",
        "label": "Zero-Shot",
        "experiment": "dreamdojo_2b_480_640_tlrc_yam_zeroshot",
        "camera_layout": "head",
        "ckpt_path": "checkpoints/DreamDojo/2B_YAM_post-train/iter_000050000/model_ema_bf16.pt",
    },
    {
        "id": "1_v2_uncropped",
        "label": "v2 Uncrop (21ep)",
        "experiment": "dreamdojo_2b_480_640_tlrc_yam_v2",
        "camera_layout": "head",
        "ckpt_path": "outputs/dreamdojo/tlrc_yam_finetune/tlrc_bimanual_yam_v2/checkpoints/iter_000003000/model_ema_bf16.pt",
    },
    {
        "id": "2_exp1_cropped",
        "label": "Exp1 Crop (60ep)",
        "experiment": "dreamdojo_2b_480_640_tlrc_yam_cropped",
        "camera_layout": "head_crop",
        "ckpt_path": "outputs/dreamdojo/tlrc_yam_finetune/tlrc_bimanual_yam_cropped/checkpoints/iter_000004000/model_ema_bf16.pt",
    },
    {
        "id": "3_exp2_3panel_4k",
        "label": "Exp2 3-Pan (4k)",
        "experiment": "dreamdojo_2b_480_640_tlrc_yam_3panel",
        "camera_layout": "three_panel",
        "ckpt_path": "outputs/dreamdojo/tlrc_yam_finetune/tlrc_bimanual_yam_3panel/checkpoints/iter_000004000/model_ema_bf16.pt",
    },
    {
        "id": "4_exp2_3panel_stg2_5k",
        "label": "Exp2 3-Pan (9k)",
        "experiment": "dreamdojo_2b_480_640_tlrc_yam_3panel_stage2",
        "camera_layout": "three_panel",
        "ckpt_path": "outputs/dreamdojo/tlrc_yam_finetune/tlrc_bimanual_yam_3panel_stage2/checkpoints/iter_000005000/model_ema_bf16.pt",
    },
    {
        "id": "5_exp3_3panel_1to1_hist",
        "label": "Exp3 1:1+Hist",
        "experiment": "dreamdojo_2b_480_640_tlrc_yam_3panel_1to1_hist",
        "camera_layout": "three_panel_1to1",
        "ckpt_path": "outputs/dreamdojo/tlrc_yam_finetune/tlrc_bimanual_yam_3panel_1to1_hist/checkpoints/best/model_ema_bf16.pt",
        "num_cond_latents": 2,
    },
    {
        "id": "6_exp3_3panel_1to1_hist5f",
        "label": "Exp3 1:1+Hist(5f0)",
        "experiment": "dreamdojo_2b_480_640_tlrc_yam_3panel_1to1_hist",
        "camera_layout": "three_panel_1to1",
        "ckpt_path": "outputs/dreamdojo/tlrc_yam_finetune/tlrc_bimanual_yam_3panel_1to1_hist/checkpoints/best/model_ema_bf16.pt",
        "num_cond_latents": 2,
        "init_5frame_history": True,
    },
]


def extract_shared_roi_video(video_thwc: np.ndarray, camera_layout: str) -> np.ndarray:
    """
    Crops a (T, 480, 640, 3) uint8 video from any layout to the exact shared tabletop ROI
    and resizes to (T, 288, 640, 3) uint8 (a zero-interpolation 1:1 slice for head_crop & three_panel_1to1).
    """
    head_vid = (
        video_thwc[:, :288, :, :]
        if camera_layout in ("three_panel", "three_panel_1to1")
        else video_thwc
    )
    H, W = head_vid.shape[1:3]
    lx1, ly1, lx2, ly2 = LAYOUT_HEAD_BOUNDS[camera_layout]
    sx1, sy1, sx2, sy2 = SHARED_ROI_NORM

    rx1 = int(round((sx1 - lx1) / (lx2 - lx1) * W))
    rx2 = int(round((sx2 - lx1) / (lx2 - lx1) * W))
    ry1 = int(round((sy1 - ly1) / (ly2 - ly1) * H))
    ry2 = int(round((sy2 - ly1) / (ly2 - ly1) * H))
    rx1, rx2 = max(0, rx1), min(W, rx2)
    ry1, ry2 = max(0, ry1), min(H, ry2)

    out_h, out_w = SHARED_ROI_HW
    cropped = head_vid[:, ry1:ry2, rx1:rx2]
    if cropped.shape[1] == out_h and cropped.shape[2] == out_w:
        return cropped.copy()
    out = np.empty((video_thwc.shape[0], out_h, out_w, 3), dtype=np.uint8)
    interp = cv2.INTER_AREA if (cropped.shape[1] >= out_h and cropped.shape[2] >= out_w) else cv2.INTER_LINEAR
    for t in range(video_thwc.shape[0]):
        out[t] = cv2.resize(cropped[t], (out_w, out_h), interpolation=interp)
    return out


def select_bimanual_clips(dataset_path: str, num_frames: int = 49) -> List[Tuple[int, int, str]]:
    """
    Selects 1 peak active clip per episode across all episodes in dataset_path:
      - If both arms move in the episode, maximizes geometric mean sqrt(left_L1 * right_L1).
      - If single-arm episode (e.g. autogen), maximizes total active motion (left_L1 + right_L1).
    Ensures base_index >= 16 so 5-frame historical conditioning (t=-4..0) is always available.
    Returns list of (dataset_index, episode_id, description).
    """
    from groot_dreams.dataloader import MultiVideoActionDataset

    ds = MultiVideoActionDataset(
        num_frames=num_frames,
        dataset_path=[dataset_path],
        data_split="full",
        camera_layout="head",
        cr1_embeddings_path="datasets/cr1_empty_text_embeddings.pt",
    )
    le = ds.datasets[0].lerobot_dataset
    ep_ids = sorted(set(int(t) for t, _ in le.all_steps))

    chosen = []
    for eid in ep_ids:
        ep_indices = [i for i, (t, b_idx) in enumerate(le.all_steps) if int(t) == eid and int(b_idx) >= 16]
        if not ep_indices:
            ep_indices = [i for i, (t, _) in enumerate(le.all_steps) if int(t) == eid]
        # Sample ~12 candidate starts across the episode (skipping very first/last 5%)
        lo = int(len(ep_indices) * 0.08)
        hi = max(lo + 1, int(len(ep_indices) * 0.90))
        cands = [ep_indices[int(lo + k * (hi - lo) / 11)] for k in range(12)]

        cand_stats = []
        for idx in cands:
            s = ds[idx]
            act = s["action"][: num_frames - 1, 101:147]
            l_act = float(act[:, 34:40].abs().sum())
            r_act = float(act[:, 40:46].abs().sum())
            cand_stats.append((idx, l_act, r_act))

        has_bimanual = any(l > 0.5 and r > 0.5 for _, l, r in cand_stats)
        best_score = -1.0
        best_idx = cands[len(cands) // 2]
        best_lr = (0.0, 0.0)
        for idx, l_act, r_act in cand_stats:
            if has_bimanual:
                score = float(np.sqrt(max(l_act, 0.0) * max(r_act, 0.0)))
            else:
                score = float(l_act + r_act)
            if score > best_score:
                best_score = score
                best_idx = idx
                best_lr = (l_act, r_act)
        tag = "2arm" if has_bimanual else "1arm"
        desc = f"ep{eid:02d}_{tag}_idx{best_idx}_L{best_lr[0]:.1f}_R{best_lr[1]:.1f}"
        print(
            f"Selected clip for ep {eid} ({tag}): ds_idx={best_idx} "
            f"(left_L1={best_lr[0]:.2f}, right_L1={best_lr[1]:.2f})"
        )
        chosen.append((best_idx, eid, desc))
    return chosen


def compute_video_metrics(
    gt_thwc: np.ndarray, pr_thwc: np.ndarray, lpips_fn, chunk_size: int = 12
) -> Dict:
    """Computes overall and per-chunk PSNR/SSIM/LPIPS plus left-half / right-half PSNR."""
    gt_t = torch.from_numpy(gt_thwc[1:]).permute(0, 3, 1, 2).float() / 255.0
    pr_t = torch.from_numpy(pr_thwc[1:]).permute(0, 3, 1, 2).float() / 255.0
    device = next(lpips_fn.parameters()).device
    gt_t = (torch.from_numpy(gt_thwc[1:]).permute(0, 3, 1, 2).float() / 255.0).to(device)
    pr_t = (torch.from_numpy(pr_thwc[1:]).permute(0, 3, 1, 2).float() / 255.0).to(device)
    n_gen = gt_t.shape[0]
    n_chunks = n_gen // chunk_size
    half_w = gt_t.shape[3] // 2

    chunks = []
    for c in range(n_chunks):
        s, e = c * chunk_size, (c + 1) * chunk_size
        chunks.append(
            {
                "psnr": float(piq.psnr(gt_t[s:e], pr_t[s:e]).item()),
                "ssim": float(piq.ssim(gt_t[s:e], pr_t[s:e]).item()),
                "lpips": float(lpips_fn(gt_t[s:e], pr_t[s:e]).item()),
            }
        )

    return {
        "psnr": float(piq.psnr(gt_t, pr_t).item()),
        "ssim": float(piq.ssim(gt_t, pr_t).item()),
        "lpips": float(lpips_fn(gt_t, pr_t).item()),
        "left_half_psnr": float(piq.psnr(gt_t[:, :, :, :half_w], pr_t[:, :, :, :half_w]).item()),
        "right_half_psnr": float(piq.psnr(gt_t[:, :, :, half_w:], pr_t[:, :, :, half_w:]).item()),
        "chunks": chunks,
    }


def label_frame(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return out


@torch.no_grad()
def run_benchmark(
    dataset_path: str,
    save_dir: Path,
    num_frames: int = 49,
    modes: List[str] = ["gt", "zero", "negated"],
    only_models: List[str] | None = None,
):
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
    from groot_dreams.dataloader import MultiVideoActionDataset

    save_dir.mkdir(parents=True, exist_ok=True)
    clips_meta_path = save_dir / "selected_clips.json"
    if clips_meta_path.exists():
        chosen_clips = [tuple(x) for x in json.loads(clips_meta_path.read_text())]
        print("Loaded cached clip selection:", chosen_clips)
    else:
        chosen_clips = select_bimanual_clips(dataset_path, num_frames=num_frames)
        clips_meta_path.write_text(json.dumps(chosen_clips, indent=2))

    # Build canonical unscaled single-camera ground-truth ROI once per episode from 1:1 head_crop
    canonical_gt_dir = save_dir / "_canonical_gt_roi"
    canonical_gt_dir.mkdir(parents=True, exist_ok=True)
    if not all((canonical_gt_dir / f"ep{eid:02d}_gt_roi.npy").exists() for _, eid, _ in chosen_clips):
        ds_canon = MultiVideoActionDataset(
            num_frames=num_frames,
            dataset_path=[dataset_path],
            data_split="full",
            camera_layout="head_crop",
            cr1_embeddings_path="datasets/cr1_empty_text_embeddings.pt",
        )
        for ds_idx, eid, _ in chosen_clips:
            gt_vid_canon = ds_canon[ds_idx]["video"].permute(1, 2, 3, 0).numpy()
            gt_roi_canon = extract_shared_roi_video(gt_vid_canon, "head_crop")
            np.save(canonical_gt_dir / f"ep{eid:02d}_gt_roi.npy", gt_roi_canon)
        del ds_canon

    lpips_fn = piq.LPIPS(replace_pooling=True)
    chunk_size = 12
    num_chunks = (num_frames - 1) // chunk_size

    specs = MODEL_SPECS
    if only_models:
        specs = [s for s in MODEL_SPECS if s["id"] in only_models]
    else:
        specs = [s for s in MODEL_SPECS if Path(s["ckpt_path"]).exists()]

    for spec in specs:
        mid = spec["id"]
        layout = spec["camera_layout"]
        num_cond_latents = int(spec.get("num_cond_latents", 1))
        init_5f = bool(spec.get("init_5frame_history", False))
        model_dir = save_dir / mid
        model_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = model_dir / "metrics.json"

        # Check if all clip/mode raw videos already cached
        all_cached = all(
            (model_dir / f"ep{eid:02d}_{mode}.npy").exists()
            for _, eid, _ in chosen_clips
            for mode in modes
        )
        if all_cached and metrics_path.exists():
            print(f"Skipping {mid} ({spec['label']}): rollouts and metrics.json already cached.")
            continue

        ds = MultiVideoActionDataset(
            num_frames=num_frames,
            dataset_path=[dataset_path],
            data_split="full",
            camera_layout=layout,
            cr1_embeddings_path="datasets/cr1_empty_text_embeddings.pt",
        )
        ds_hist5 = None
        if init_5f:
            ds_hist5 = MultiVideoActionDataset(
                num_frames=5,
                dataset_path=[dataset_path],
                data_split="full",
                camera_layout=layout,
                cr1_embeddings_path="datasets/cr1_empty_text_embeddings.pt",
            )

        if not all_cached:
            print(
                f"\n{'='*80}\nRunning model: {mid} ({spec['label']}) | layout={layout} "
                f"| cond_latents={num_cond_latents} | init_5f={init_5f}\n{'='*80}"
            )
            v2w = Video2WorldInference(
                experiment_name=spec["experiment"],
                ckpt_path=spec["ckpt_path"],
                s3_credential_path="",
                context_parallel_size=1,
                config_file="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
                offload_diffusion_model=False,
                offload_text_encoder=True,
                offload_tokenizer=False,
            )
            v2w.model.on_train_start()
            v2w.model.eval()
            model_state_t = int(v2w.model.config.state_t)
            model_req_frames = 1 + 4 * (model_state_t - 1)  # 13 for state_t=4, 17 for state_t=5
            model_req_actions = 4 * (model_state_t - 1)     # 12 for state_t=4, 16 for state_t=5

            for ds_idx, eid, desc in chosen_clips:
                data = ds[ds_idx]
                gt_video = data["video"].permute(1, 2, 3, 0).numpy()  # (T, 480, 640, 3) uint8
                np.save(model_dir / f"ep{eid:02d}_gt_ref.npy", gt_video)
                base_actions = data["action"][: num_frames - 1].numpy()

                hist_prefix_frames = None
                hist_prefix_actions = None
                if init_5f and ds_hist5 is not None and ds_idx >= 16:
                    # 4 model steps = 16 raw 30Hz steps prior to ds_idx
                    h_data = ds_hist5[ds_idx - 16]
                    hist_prefix_frames = h_data["video"].transpose(0, 1)  # (5, C, H, W) uint8
                    hist_prefix_actions = h_data["action"][:4].numpy()     # (4, 384)

                for mode in modes:
                    out_npy = model_dir / f"ep{eid:02d}_{mode}.npy"
                    if out_npy.exists():
                        continue
                    actions = base_actions.copy()
                    if mode == "zero":
                        actions[:, 101:147] = 0.0
                    elif mode == "negated":
                        actions[:, 101:147] = -actions[:, 101:147]

                    rollout_frames_u8: List[np.ndarray] = [gt_video[0]]
                    for c_idx in range(num_chunks):
                        i = c_idx * chunk_size
                        use_2lat = (num_cond_latents >= 2 and model_state_t >= 5) and (
                            c_idx > 0 or hist_prefix_frames is not None
                        )
                        if not use_2lat:
                            # 1-frame conditioning (1 latent frame)
                            cond_u8 = rollout_frames_u8[-1]  # (H, W, C) uint8
                            cond_tensor = torch.from_numpy(cond_u8).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
                            pad_count = model_req_frames - 1
                            vid_input = torch.cat(
                                [cond_tensor, torch.zeros_like(cond_tensor).repeat(pad_count, 1, 1, 1)], dim=0
                            ).to(torch.uint8)
                            vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)
                            act_slice = actions[i : i + model_req_actions]
                            if len(act_slice) < model_req_actions:
                                pad_a = np.zeros((model_req_actions - len(act_slice), actions.shape[1]), dtype=actions.dtype)
                                act_slice = np.concatenate([act_slice, pad_a], axis=0)
                            cur_cond_latents = 1
                            keep_start = 1
                            keep_end = 1 + chunk_size
                        else:
                            # 5-frame history conditioning (2 latent frames = 0.53s visual + action memory)
                            if c_idx == 0 and hist_prefix_frames is not None and hist_prefix_actions is not None:
                                cond_tensor = hist_prefix_frames  # (5, C, H, W) uint8
                                h_act = hist_prefix_actions.copy()
                                if mode == "zero":
                                    h_act[:, 101:147] = 0.0
                                elif mode == "negated":
                                    h_act[:, 101:147] = -h_act[:, 101:147]
                                act_slice = np.concatenate([h_act, actions[i : i + chunk_size]], axis=0)  # (16, 384)
                            else:
                                hist5_u8 = np.stack(rollout_frames_u8[-5:], axis=0)  # (5, H, W, C)
                                cond_tensor = torch.from_numpy(hist5_u8).permute(0, 3, 1, 2)  # (5, C, H, W)
                                act_slice = actions[i - 4 : i + chunk_size]  # (16, 384): 4 past + 12 future
                            pad_count = model_req_frames - 5
                            vid_input = torch.cat(
                                [cond_tensor, cond_tensor[-1:].repeat(pad_count, 1, 1, 1)], dim=0
                            ).to(torch.uint8)
                            vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)
                            cur_cond_latents = 2
                            keep_start = 5
                            keep_end = 5 + chunk_size

                        video = v2w.generate_vid2world(
                            prompt="",
                            input_path=vid_input,
                            action=torch.from_numpy(act_slice).float(),
                            guidance=0,
                            num_video_frames=model_req_frames,
                            num_latent_conditional_frames=cur_cond_latents,
                            resolution="480,640",
                            seed=i,
                            negative_prompt="",
                        )
                        video_norm = (video + 1.0) / 2.0
                        video_u8 = (
                            (torch.clamp(video_norm[0], 0, 1) * 255)
                            .to(torch.uint8)
                            .permute(1, 2, 3, 0)
                            .cpu()
                            .numpy()
                        )
                        for f_u8 in video_u8[keep_start:keep_end]:
                            rollout_frames_u8.append(f_u8)

                    rollout_vid = np.stack(rollout_frames_u8, axis=0)
                    np.save(out_npy, rollout_vid)
                    print(f"  Saved {out_npy.name} {rollout_vid.shape}")

            del v2w
            gc.collect()
            torch.cuda.empty_cache()

        # Compute metrics from saved .npy files against CANONICAL single-camera GT ROI
        if torch.cuda.is_available():
            lpips_fn = lpips_fn.cuda()
        model_metrics: Dict = {"id": mid, "label": spec["label"], "camera_layout": layout, "modes": {}}
        for mode in modes:
            per_ep_shared = []
            per_ep_autogen = []
            per_ep_twoarm = []
            per_ep_native = []
            per_ep_l_wrist = []
            per_ep_r_wrist = []
            for _, eid, desc in chosen_clips:
                gt_vid = np.load(model_dir / f"ep{eid:02d}_gt_ref.npy")
                pr_vid = np.load(model_dir / f"ep{eid:02d}_{mode}.npy")

                gt_roi = np.load(canonical_gt_dir / f"ep{eid:02d}_gt_roi.npy")
                pr_roi = extract_shared_roi_video(pr_vid, layout)
                m_shared = compute_video_metrics(gt_roi, pr_roi, lpips_fn, chunk_size)
                per_ep_shared.append(m_shared)
                if "1arm" in desc or eid < 3 and len(chosen_clips) == 6:
                    per_ep_autogen.append(m_shared)
                else:
                    per_ep_twoarm.append(m_shared)
                per_ep_native.append(compute_video_metrics(gt_vid, pr_vid, lpips_fn, chunk_size))

                if layout in ("three_panel", "three_panel_1to1"):
                    per_ep_l_wrist.append(
                        compute_video_metrics(gt_vid[:, 288:, :320], pr_vid[:, 288:, :320], lpips_fn, chunk_size)
                    )
                    per_ep_r_wrist.append(
                        compute_video_metrics(gt_vid[:, 288:, 320:], pr_vid[:, 288:, 320:], lpips_fn, chunk_size)
                    )

            def agg(lst):
                return {
                    "psnr": float(np.mean([x["psnr"] for x in lst])),
                    "ssim": float(np.mean([x["ssim"] for x in lst])),
                    "lpips": float(np.mean([x["lpips"] for x in lst])),
                    "left_half_psnr": float(np.mean([x["left_half_psnr"] for x in lst])),
                    "right_half_psnr": float(np.mean([x["right_half_psnr"] for x in lst])),
                    "chunk_psnr": [
                        float(np.mean([x["chunks"][c]["psnr"] for x in lst])) for c in range(num_chunks)
                    ],
                }

            entry = {"shared_roi": agg(per_ep_shared), "native_full": agg(per_ep_native)}
            if per_ep_autogen:
                entry["shared_roi_autogen3"] = agg(per_ep_autogen)
            if per_ep_twoarm:
                entry["shared_roi_twoarm3"] = agg(per_ep_twoarm)
            if per_ep_l_wrist:
                entry["left_wrist"] = agg(per_ep_l_wrist)
                entry["right_wrist"] = agg(per_ep_r_wrist)
            model_metrics["modes"][mode] = entry

        metrics_path.write_text(json.dumps(model_metrics, indent=2))

    # Aggregate summary across ALL available models
    all_summaries = []
    for spec in MODEL_SPECS:
        mp = save_dir / spec["id"] / "metrics.json"
        if mp.exists():
            all_summaries.append(json.loads(mp.read_text()))
    (save_dir / "summary.json").write_text(json.dumps(all_summaries, indent=2))

    for subset_key, title in [
        ("shared_roi", f"ALL {len(chosen_clips)} HELD-OUT EPISODES (CANONICAL SINGLE-CAMERA TABLETOP ROI 288x640)"),
        ("shared_roi_autogen3", "AUTOGEN-3 SUBSET (3 Single-Arm Pick-and-Place Held-Out Episodes)"),
        ("shared_roi_twoarm3", "TWOARM-3 SUBSET (3 Simultaneous Bimanual Held-Out Episodes)"),
    ]:
        if not all_summaries or subset_key not in all_summaries[0]["modes"]["gt"]:
            continue
        print("\n" + "=" * 114)
        print(title)
        print("=" * 114)
        print(
            f"{'Model':20s} | {'GT PSNR':>8s} | {'GT SSIM':>7s} | {'GT LPIPS':>8s} | "
            f"{'L-Arm':>7s} | {'R-Arm':>7s} | {'GT-Zero':>8s} | {'GT-Neg':>8s} | {'Chk1->4 PSNR'}"
        )
        print("-" * 114)
        for m in all_summaries:
            gt = m["modes"]["gt"][subset_key]
            dz = gt["psnr"] - m["modes"]["zero"][subset_key]["psnr"] if "zero" in m["modes"] else 0.0
            dn = gt["psnr"] - m["modes"]["negated"][subset_key]["psnr"] if "negated" in m["modes"] else 0.0
            chk = " -> ".join(f"{x:.1f}" for x in gt["chunk_psnr"])
            print(
                f"{m['label']:20s} | {gt['psnr']:7.2f}d | {gt['ssim']:7.3f} | {gt['lpips']:8.3f} | "
                f"{gt['left_half_psnr']:6.2f}d | {gt['right_half_psnr']:6.2f}d | {dz:+7.2f}d | {dn:+7.2f}d | {chk}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="datasets/tlrc_bimanual_yam_eval6")
    parser.add_argument("--save-dir", type=Path, default=Path("results/eval6_benchmark"))
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--modes", nargs="+", default=["gt", "zero", "negated"])
    parser.add_argument("--only-models", nargs="+", default=None)
    args = parser.parse_args()
    run_benchmark(
        dataset_path=args.dataset_path,
        save_dir=args.save_dir,
        num_frames=args.num_frames,
        modes=args.modes,
        only_models=args.only_models,
    )
