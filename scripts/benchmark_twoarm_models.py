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

# Normalized (x1, y1, x2, y2) in 1920x1200 rectified coordinates covered by each layout's head view
LAYOUT_HEAD_BOUNDS: Dict[str, Tuple[float, float, float, float]] = {
    "head": (160.0 / 1920.0, 0.0, 1760.0 / 1920.0, 1.0),
    "head_crop": (0.25, 0.32, 0.75, 0.92),
    "three_panel": (0.1875, 0.35, 0.8125, 0.80),
}
# Exact spatial intersection of all 3 head views (both arms + full tabletop bricks, 16:9)
SHARED_ROI_NORM: Tuple[float, float, float, float] = (0.25, 0.35, 0.75, 0.80)
SHARED_ROI_HW: Tuple[int, int] = (270, 480)

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
]


def extract_shared_roi_video(video_thwc: np.ndarray, camera_layout: str) -> np.ndarray:
    """
    Crops a (T, 480, 640, 3) uint8 video from any layout to the exact shared tabletop ROI
    and resizes to (T, 270, 480, 3) uint8 so all models are compared on identical pixels.
    """
    head_vid = video_thwc[:, :288, :, :] if camera_layout == "three_panel" else video_thwc
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
    out = np.empty((video_thwc.shape[0], out_h, out_w, 3), dtype=np.uint8)
    for t in range(video_thwc.shape[0]):
        out[t] = cv2.resize(head_vid[t, ry1:ry2, rx1:rx2], (out_w, out_h), interpolation=cv2.INTER_AREA)
    return out


def select_bimanual_clips(dataset_path: str, num_frames: int = 49) -> List[Tuple[int, int, str]]:
    """
    Selects 1 peak simultaneous-bimanual clip per episode (total 3 clips across ep 0, 1, 2)
    where BOTH left and right arms have strong motion (maximizing min(left_L1, right_L1)).
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
        ep_indices = [i for i, (t, _) in enumerate(le.all_steps) if int(t) == eid]
        # Sample ~12 candidate starts across the episode (skipping very first/last 5%)
        lo = int(len(ep_indices) * 0.08)
        hi = max(lo + 1, int(len(ep_indices) * 0.90))
        cands = [ep_indices[int(lo + k * (hi - lo) / 11)] for k in range(12)]

        best_score = -1.0
        best_idx = cands[len(cands) // 2]
        best_lr = (0.0, 0.0)
        for idx in cands:
            s = ds[idx]
            act = s["action"][: num_frames - 1, 101:147]
            l_act = float(act[:, 34:40].abs().sum())
            r_act = float(act[:, 40:46].abs().sum())
            # Score rewards BOTH arms moving strongly (geometric mean)
            score = float(np.sqrt(max(l_act, 0.0) * max(r_act, 0.0)))
            if score > best_score:
                best_score = score
                best_idx = idx
                best_lr = (l_act, r_act)
        desc = f"ep{eid:02d}_idx{best_idx}_L{best_lr[0]:.1f}_R{best_lr[1]:.1f}"
        print(f"Selected bimanual clip for ep {eid}: ds_idx={best_idx} (left_L1={best_lr[0]:.2f}, right_L1={best_lr[1]:.2f})")
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

    lpips_fn = piq.LPIPS(replace_pooling=True)
    chunk_size = 12
    num_chunks = (num_frames - 1) // chunk_size

    specs = MODEL_SPECS
    if only_models:
        specs = [s for s in MODEL_SPECS if s["id"] in only_models]

    for spec in specs:
        mid = spec["id"]
        layout = spec["camera_layout"]
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

        if not all_cached:
            print(f"\n{'='*80}\nRunning model: {mid} ({spec['label']}) | layout={layout}\n{'='*80}")
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

            for ds_idx, eid, desc in chosen_clips:
                data = ds[ds_idx]
                gt_video = data["video"].permute(1, 2, 3, 0).numpy()  # (T, 480, 640, 3) uint8
                np.save(model_dir / f"ep{eid:02d}_gt_ref.npy", gt_video)
                base_actions = data["action"][: num_frames - 1].numpy()

                for mode in modes:
                    out_npy = model_dir / f"ep{eid:02d}_{mode}.npy"
                    if out_npy.exists():
                        continue
                    actions = base_actions.copy()
                    if mode == "zero":
                        actions[:, 101:147] = 0.0
                    elif mode == "negated":
                        actions[:, 101:147] = -actions[:, 101:147]

                    img_array = data["video"].transpose(0, 1)[:1]
                    chunk_preds = []
                    first_round = True
                    for c_idx in range(num_chunks):
                        i = c_idx * chunk_size
                        actions_chunk = actions[i : i + chunk_size]
                        if not first_round:
                            img_tensor = torchvision.transforms.functional.to_tensor(img_array).unsqueeze(0) * 255.0
                        else:
                            img_tensor = img_array
                        first_round = False

                        num_video_frames = chunk_size + 1
                        vid_input = torch.cat(
                            [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)], dim=0
                        ).to(torch.uint8)
                        vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)

                        video = v2w.generate_vid2world(
                            prompt="",
                            input_path=vid_input,
                            action=torch.from_numpy(actions_chunk).float(),
                            guidance=0,
                            num_video_frames=num_video_frames,
                            num_latent_conditional_frames=1,
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
                        img_array = video_u8[-1]
                        chunk_preds.append(video_u8 if c_idx == 0 else video_u8[1:])

                    rollout_vid = np.concatenate(chunk_preds, axis=0)
                    np.save(out_npy, rollout_vid)
                    print(f"  Saved {out_npy.name} {rollout_vid.shape}")

            del v2w
            gc.collect()
            torch.cuda.empty_cache()

        # Compute metrics from saved .npy files (shared tabletop ROI + native + wrist panels)
        if torch.cuda.is_available():
            lpips_fn = lpips_fn.cuda()
        model_metrics: Dict = {"id": mid, "label": spec["label"], "camera_layout": layout, "modes": {}}
        for mode in modes:
            per_ep_shared = []
            per_ep_native = []
            per_ep_l_wrist = []
            per_ep_r_wrist = []
            for _, eid, _ in chosen_clips:
                gt_vid = np.load(model_dir / f"ep{eid:02d}_gt_ref.npy")
                pr_vid = np.load(model_dir / f"ep{eid:02d}_{mode}.npy")

                gt_roi = extract_shared_roi_video(gt_vid, layout)
                pr_roi = extract_shared_roi_video(pr_vid, layout)
                per_ep_shared.append(compute_video_metrics(gt_roi, pr_roi, lpips_fn, chunk_size))
                per_ep_native.append(compute_video_metrics(gt_vid, pr_vid, lpips_fn, chunk_size))

                if layout == "three_panel":
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
            if per_ep_l_wrist:
                entry["left_wrist"] = agg(per_ep_l_wrist)
                entry["right_wrist"] = agg(per_ep_r_wrist)
            model_metrics["modes"][mode] = entry

        metrics_path.write_text(json.dumps(model_metrics, indent=2))

    # Aggregate summary & render side-by-side comparison videos across ALL available models
    all_summaries = []
    for spec in MODEL_SPECS:
        mp = save_dir / spec["id"] / "metrics.json"
        if mp.exists():
            all_summaries.append(json.loads(mp.read_text()))
    (save_dir / "summary.json").write_text(json.dumps(all_summaries, indent=2))

    print("\n" + "=" * 110)
    print("APPLES-TO-APPLES SHARED TABLETOP ROI BENCHMARK (3 Two-Arm Episodes, 48-Frame Rollouts)")
    print("=" * 110)
    print(
        f"{'Model':20s} | {'GT PSNR':>8s} | {'GT SSIM':>7s} | {'GT LPIPS':>8s} | "
        f"{'L-Arm':>7s} | {'R-Arm':>7s} | {'GT-Zero':>8s} | {'GT-Neg':>8s} | {'Chk1->4 PSNR'}"
    )
    print("-" * 110)
    for m in all_summaries:
        gt = m["modes"]["gt"]["shared_roi"]
        dz = gt["psnr"] - m["modes"]["zero"]["shared_roi"]["psnr"] if "zero" in m["modes"] else 0.0
        dn = gt["psnr"] - m["modes"]["negated"]["shared_roi"]["psnr"] if "negated" in m["modes"] else 0.0
        chk = " -> ".join(f"{x:.1f}" for x in gt["chunk_psnr"])
        print(
            f"{m['label']:20s} | {gt['psnr']:7.2f}d | {gt['ssim']:7.3f} | {gt['lpips']:8.3f} | "
            f"{gt['left_half_psnr']:6.2f}d | {gt['right_half_psnr']:6.2f}d | {dz:+7.2f}d | {dn:+7.2f}d | {chk}"
        )

    # Render 2x3 grid comparison video per episode on Shared Tabletop ROI (GT + 5 models)
    if len(all_summaries) == len(MODEL_SPECS):
        for _, eid, desc in chosen_clips:
            gt_ref = np.load(save_dir / MODEL_SPECS[2]["id"] / f"ep{eid:02d}_gt_ref.npy")
            gt_roi = extract_shared_roi_video(gt_ref, MODEL_SPECS[2]["camera_layout"])

            panels_by_t = []
            model_rois = [
                extract_shared_roi_video(
                    np.load(save_dir / s["id"] / f"ep{eid:02d}_gt.npy"), s["camera_layout"]
                )
                for s in MODEL_SPECS
            ]
            for t in range(gt_roi.shape[0]):
                top_row = np.hstack([
                    label_frame(gt_roi[t], f"GT ({desc}) f={t:02d}"),
                    label_frame(model_rois[0][t], MODEL_SPECS[0]["label"]),
                    label_frame(model_rois[1][t], MODEL_SPECS[1]["label"]),
                ])
                bot_row = np.hstack([
                    label_frame(model_rois[2][t], MODEL_SPECS[2]["label"]),
                    label_frame(model_rois[3][t], MODEL_SPECS[3]["label"]),
                    label_frame(model_rois[4][t], MODEL_SPECS[4]["label"]),
                ])
                panels_by_t.append(np.vstack([top_row, bot_row]))
            out_mp4 = save_dir / f"twoarm_ep{eid:02d}_all6_shared_roi.mp4"
            mediapy.write_video(str(out_mp4), np.stack(panels_by_t, axis=0), fps=7.5)
            print(f"Saved 6-panel comparison video -> {out_mp4}")

            # Also render native 3-panel comparison [GT | Exp2-4k | Exp2-Stg2]
            gt_3p = np.load(save_dir / "3_exp2_3panel_4k" / f"ep{eid:02d}_gt_ref.npy")
            p1_3p = np.load(save_dir / "3_exp2_3panel_4k" / f"ep{eid:02d}_gt.npy")
            p2_3p = np.load(save_dir / "4_exp2_3panel_stg2_5k" / f"ep{eid:02d}_gt.npy")
            trip = np.stack(
                [
                    np.hstack([
                        label_frame(gt_3p[t], f"GT 3-Panel (ep{eid:02d}) f={t:02d}"),
                        label_frame(p1_3p[t], "Exp2 3-Panel Stg1 (4k)"),
                        label_frame(p2_3p[t], "Exp2 3-Panel Stg2 (9k)"),
                    ])
                    for t in range(gt_3p.shape[0])
                ],
                axis=0,
            )
            out_3p = save_dir / f"twoarm_ep{eid:02d}_3panel_stg1_vs_stg2.mp4"
            mediapy.write_video(str(out_3p), trip, fps=7.5)
            print(f"Saved 3-panel comparison video -> {out_3p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="datasets/tlrc_bimanual_yam_twoarm_benchmark")
    parser.add_argument("--save-dir", type=Path, default=Path("results/twoarm_benchmark"))
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
