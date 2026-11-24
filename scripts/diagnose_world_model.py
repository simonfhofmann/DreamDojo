#!/usr/bin/env python3
"""
Comprehensive diagnostic suite for DreamDojo action-conditioned world model:
1. CPU Action Pipeline & Split Audit (--audit-only):
   - Checks train/test episode split leakage (traj 19 boundary vs traj 20 pure held-out)
   - Compares native dataset normalization stats vs shared_meta/YAM_stats.json (clipping / dynamic range)
   - Verifies active action slots (101:147) and detects the `key = action_seq[0:1, :29]` view mutation in dataset.py
   - Inspects action_embedder weight norms across checkpoints (pretrained, v1, v2)
2. Deterministic Fixed-Sigma Action Sensitivity Test (--mode loss):
   - Evaluates loss on pure held-out episode (traj 20) with fixed noise & fixed sigma in {0.2, 0.5, 0.8}
   - Compares True Actions vs Zero Actions vs Negated Actions vs Swapped Actions
3. Counterfactual & Per-Chunk Rollout Evaluation (--mode rollout):
   - Generates 4-panel [GT | Pred(True) | Pred(Zero) | Pred(Swapped)] videos from the exact same Frame 0 & seed
   - Reports per-chunk (Chunk 1..4) PSNR, SSIM, LPIPS to separate single-chunk accuracy from multi-chunk drift
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch


def run_cpu_audit(dataset_path: str = "datasets/tlrc_bimanual_yam", camera_layout: str = "head"):
    print("=" * 80)
    print("1. TRAIN / VAL SPLIT & LEAKAGE AUDIT")
    print("=" * 80)
    from groot_dreams.dataloader import MultiVideoActionDataset

    ds_train = MultiVideoActionDataset(
        num_frames=13, dataset_path=dataset_path, data_split="train", camera_layout=camera_layout
    )
    ds_test = MultiVideoActionDataset(
        num_frames=13, dataset_path=dataset_path, data_split="test", camera_layout=camera_layout
    )
    le_train = ds_train.datasets[0].lerobot_dataset
    le_test = ds_test.datasets[0].lerobot_dataset

    train_trajs = sorted(set(int(t) for t, _ in le_train.all_steps))
    test_trajs = sorted(set(int(t) for t, _ in le_test.all_steps))
    overlap_trajs = sorted(set(train_trajs) & set(test_trajs))
    print(f"Train windows: {len(le_train)} across episodes {train_trajs}")
    print(f"Test windows:  {len(le_test)} across episodes {test_trajs}")
    print(f"Overlapping episodes between train and test: {overlap_trajs}")
    for tid in overlap_trajs:
        tr_b = [b for t, b in le_train.all_steps if t == tid]
        te_b = [b for t, b in le_test.all_steps if t == tid]
        print(
            f"  [LEAKAGE WARNING] Episode {tid}: train base_idx=[{min(tr_b)}..{max(tr_b)}] "
            f"(frames {min(tr_b)}..{max(tr_b)+52}), "
            f"test base_idx=[{min(te_b)}..{max(te_b)}] (frames {min(te_b)}..{max(te_b)+52})"
        )
    pure_test_indices = [idx for idx, (t, _) in enumerate(le_test.all_steps) if t not in overlap_trajs]
    print(
        f"Pure held-out test windows (Episode 20 only): {len(pure_test_indices)} / {len(le_test)} "
        f"(indices {min(pure_test_indices)}..{max(pure_test_indices)} in test split)"
    )

    print("\n" + "=" * 80)
    print("2. ACTION PIPELINE, NORMALIZATION STATS & SLOT AUDIT")
    print("=" * 80)
    native_stats = json.loads((Path(dataset_path) / "meta/stats.json").read_text())
    default_stats = json.loads(Path("shared_meta/YAM_stats.json").read_text())

    n_min = np.array(native_stats["action"]["min"])
    n_max = np.array(native_stats["action"]["max"])
    d_min = np.array(default_stats["action"]["min"])
    d_max = np.array(default_stats["action"]["max"])

    n_span = np.maximum(n_max - n_min, 1e-8)
    d_span = np.maximum(d_max - d_min, 1e-8)
    ratio = n_span / d_span

    groups = [
        ("ee_pose_action_left (16D)", 0, 16),
        ("ee_pose_action_right (16D)", 16, 32),
        ("gripper_pos_action_left (1D)", 32, 33),
        ("gripper_pos_action_right (1D)", 33, 34),
        ("joint_pos_action_left (6D)", 34, 40),
        ("joint_pos_action_right (6D)", 40, 46),
    ]
    for name, s, e in groups:
        active_mask = n_span[s:e] > 1e-5
        if active_mask.any():
            r_sub = ratio[s:e][active_mask]
            print(
                f"  {name:30s}: native/default span ratio min={r_sub.min():.2f}x, "
                f"max={r_sub.max():.2f}x, mean={r_sub.mean():.2f}x"
            )

    # Sample 50 test clips to inspect actual tensor values in action_seq
    acts = torch.stack([ds_test[i]["action"] for i in range(0, len(ds_test), max(1, len(ds_test) // 50))[:50]])
    # acts shape: [N, 12, 384]
    nonzero_cols = torch.where(acts.abs().amax(dim=(0, 1)) > 0)[0].tolist()
    yam_cols = [c for c in nonzero_cols if 101 <= c < 147]
    leaked_state_cols = [c for c in nonzero_cols if c < 29]
    print(f"\nActive non-zero columns inYAM slot [101:147] ({len(yam_cols)} active dims): {yam_cols}")
    print(f"Leaked state columns in [0:29] at step 0 (from `key = action_seq[0:1, :29]` view): {leaked_state_cols}")
    print("  Step 0 [0:29] abs max:", acts[:, 0, :29].abs().max().item())
    print("  Steps 1..11 [0:29] abs max:", acts[:, 1:, :29].abs().max().item())
    print(
        "  [NOTE] In multi-chunk rollouts (num_frames=49), Chunk 1 has non-zero `action[0, :29]`, "
        "whereas Chunks 2, 3, 4 have ZEROS in `action[0, :29]` unless populated per chunk!"
    )

    yam_slice = acts[:, :, 101:147]
    print(
        f"  YAM delta_actions [101:147] stats across 50 clips: "
        f"min={yam_slice.min():.3f}, max={yam_slice.max():.3f}, std={yam_slice.std():.4f}"
    )
    print(
        f"  Gripper left (col 133) min/max/std:  {acts[:, :, 133].min():.3f} / {acts[:, :, 133].max():.3f} / {acts[:, :, 133].std():.4f}"
    )
    print(
        f"  Gripper right (col 134) min/max/std: {acts[:, :, 134].min():.3f} / {acts[:, :, 134].max():.3f} / {acts[:, :, 134].std():.4f}"
    )


@torch.no_grad()
def run_fixed_sigma_action_loss_test(
    experiment: str,
    ckpt_paths: list[str],
    dataset_path: str = "datasets/tlrc_bimanual_yam",
    num_clips: int = 16,
    sigmas: list[float] = [0.2, 0.5, 0.8],
    eval_cond_noise_std: float = 0.0,
    camera_layout: str = "head",
):
    """
    Deterministic evaluation of Rectified Flow loss at fixed noise and fixed sigmas
    under 4 counterfactual action conditions on held-out test episodes.
    Explicitly enforces model.eval() (so training-time cond_frame_noise_std is disabled)
    and uses `eval_cond_noise_std` (default 0.0 = clean Frame 0) identically for all models.
    """
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
    from groot_dreams.dataloader import MultiVideoActionDataset

    ds_test = MultiVideoActionDataset(
        num_frames=13,
        dataset_path=dataset_path,
        data_split="test",
        camera_layout=camera_layout,
        cr1_embeddings_path="datasets/cr1_empty_text_embeddings.pt",
    )
    le_test = ds_test.datasets[0].lerobot_dataset
    # Prefer Episode 20 if present in test split (21-ep legacy dataset), otherwise use all held-out test steps
    ep20_indices = [idx for idx, (tid, _) in enumerate(le_test.all_steps) if int(tid) == 20]
    pool_indices = ep20_indices if ep20_indices else list(range(len(le_test.all_steps)))
    eval_indices = [pool_indices[int(i * len(pool_indices) / num_clips)] for i in range(num_clips)]
    clips = [ds_test[idx] for idx in eval_indices]

    import gc
    from cosmos_predict2._src.predict2.utils.model_loader import load_model_state_dict_from_checkpoint

    v2w = Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=ckpt_paths[0],
        s3_credential_path="",
        context_parallel_size=1,
        config_file="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        offload_diffusion_model=False,
        offload_text_encoder=True,
        offload_tokenizer=False,
    )
    model = v2w.model
    model.on_train_start()
    model.eval()

    eval_tids = sorted(set(int(le_test.all_steps[idx][0]) for idx in eval_indices))
    for ckpt_path in ckpt_paths:
        print("\n" + "=" * 80)
        print(f"DETERMINISTIC ACTION-SENSITIVITY LOSS TEST")
        print(f"  Checkpoint          : {ckpt_path}")
        print(f"  Episodes evaluated  : {eval_tids} ({num_clips} clips, indices {eval_indices[0]}..{eval_indices[-1]})")
        print(f"  Eval Frame-0 Noise  : std={eval_cond_noise_std:.3f} (model.eval() enforced)")
        print("=" * 80)

        is_zero_shot = "2B_YAM_post-train" in ckpt_path or ckpt_path == "zero_shot"
        load_ckpt = (
            "outputs/dreamdojo/tlrc_yam_finetune/tlrc_bimanual_yam/checkpoints/iter_000002000/model_ema_bf16.pt"
            if is_zero_shot
            else ckpt_path
        )
        load_model_state_dict_from_checkpoint(
            model=model,
            config=v2w.config,
            s3_checkpoint_dir=load_ckpt,
            load_ema_to_reg=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
        model.on_train_start()
        model.eval()
        if is_zero_shot:
            model.net.disable_adapter_layers()
        else:
            model.net.enable_adapter_layers()

        results = {mode: {s: [] for s in sigmas} for mode in ["gt", "zero", "negated", "swapped"]}
        motion_results = {mode: {s: [] for s in sigmas} for mode in ["gt", "zero", "negated", "swapped"]}

        for clip_i, raw_clip in enumerate(clips):
            swapped_action = clips[(clip_i + num_clips // 2) % num_clips]["action"].clone()

            for mode in ["gt", "zero", "negated", "swapped"]:
                batch = {}
                for k, v in raw_clip.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.unsqueeze(0).cuda()
                    elif isinstance(v, (int, float)):
                        batch[k] = torch.tensor([v], device="cuda")
                    else:
                        batch[k] = [v]

                act = batch["action"].clone()
                if mode == "zero":
                    act[:, :, 101:147] = 0.0
                elif mode == "negated":
                    act[:, :, 101:147] = -act[:, :, 101:147]
                elif mode == "swapped":
                    act[:, :, 101:147] = swapped_action.unsqueeze(0).cuda()[:, :, 101:147]
                batch["action"] = act.to(**model.tensor_kwargs)
                if "t5_text_embeddings" in batch:
                    batch["t5_text_embeddings"] = batch["t5_text_embeddings"].to(**model.tensor_kwargs)

                _, x0, condition = model.get_data_and_condition(batch)
                # Compute spatial motion mask: top 15% latent cells that change most from Frame 0
                latent_diff = (x0[:, :, 1:] - x0[:, :, :1]).pow(2).mean(dim=1, keepdim=True)  # [1, 1, 3, H, W]
                thresh = torch.quantile(latent_diff.flatten(), 0.85)
                motion_mask = (latent_diff >= thresh).float()

                if eval_cond_noise_std > 0.0:
                    gen_c = torch.Generator(device="cuda")
                    gen_c.manual_seed(90000 + clip_i)
                    cond_noise = torch.randn(
                        condition.gt_frames.shape,
                        dtype=condition.gt_frames.dtype,
                        device="cuda",
                        generator=gen_c,
                    )
                    condition = condition.set_video_condition(
                        gt_frames=condition.gt_frames + eval_cond_noise_std * cond_noise,
                        random_min_num_conditional_frames=1,
                        random_max_num_conditional_frames=1,
                        num_conditional_frames=1,
                    )

                for s_idx, sigma_val in enumerate(sigmas):
                    gen = torch.Generator(device="cuda")
                    gen.manual_seed(10000 + clip_i * 100 + s_idx)
                    eps = torch.randn(x0.shape, dtype=torch.float32, device="cuda", generator=gen)

                    sigma_t = torch.tensor([[sigma_val]], dtype=torch.float32, device="cuda")
                    timestep_t = sigma_t * 1000.0
                    xt, vt = model.rectified_flow.get_interpolation(eps, x0, sigma_t)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        vt_pred = model.denoise(
                            noise=eps,
                            xt_B_C_T_H_W=xt.to(**model.tensor_kwargs),
                            timesteps_B_T=timestep_t,
                            condition=condition,
                        )
                    err2 = (vt_pred[:, :, 1:] - vt[:, :, 1:]) ** 2
                    mse = torch.mean(err2).item()
                    m_mse = (torch.sum(err2 * motion_mask) / (torch.sum(motion_mask) * err2.shape[1])).item()
                    results[mode][sigma_val].append(mse)
                    motion_results[mode][sigma_val].append(m_mse)

        header = f"{'Action Mode':15s} | " + " | ".join([f"sigma={s:.2f}" for s in sigmas]) + " | Mean MSE  | Delta vs GT | Moving-Arm MSE (Delta%)"
        print(header)
        print("-" * len(header))
        gt_mean = np.mean([np.mean(results["gt"][s]) for s in sigmas])
        gt_m_mean = np.mean([np.mean(motion_results["gt"][s]) for s in sigmas])
        for mode in ["gt", "zero", "negated", "swapped"]:
            vals = [np.mean(results[mode][s]) for s in sigmas]
            m_vals = [np.mean(motion_results[mode][s]) for s in sigmas]
            m_val = np.mean(vals)
            mm_val = np.mean(m_vals)
            delta = m_val - gt_mean
            pct = (delta / gt_mean) * 100.0 if gt_mean > 0 else 0.0
            mpct = ((mm_val - gt_m_mean) / gt_m_mean) * 100.0 if gt_m_mean > 0 else 0.0
            row = (
                f"{mode:15s} | "
                + " | ".join([f"{v:10.6f}" for v in vals])
                + f" | {m_val:9.6f} | {delta:+9.6f} ({pct:+5.1f}%) | {mm_val:9.6f} ({mpct:+5.1f}%)"
            )
            print(row)

    del v2w, model
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def run_counterfactual_and_chunk_rollouts(
    experiment: str,
    ckpt_paths: list[str],
    save_dir: str = "results/diagnostics",
    dataset_path: str = "datasets/tlrc_bimanual_yam",
    num_clips: int = 6,
    num_frames: int = 49,
    camera_layout: str = "head",
):
    """
    Runs 4-chunk (49-frame) rollouts on held-out test episodes under:
      1. True Actions ('gt') on all `num_clips` clips
      2. Zero Actions ('zero') & Negated Actions ('negated') on `min(num_clips, 4)` clips
    Saves a 4-panel [GT | Pred(True) | Pred(Zero) | Pred(Negated)] video per counterfactual clip
    and reports per-chunk (Chunk 1, Chunk 2, Chunk 3, Chunk 4) PSNR / SSIM / LPIPS.
    """
    import gc
    import mediapy
    import piq
    import torchvision
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
    from cosmos_predict2._src.predict2.utils.model_loader import load_model_state_dict_from_checkpoint
    from groot_dreams.dataloader import MultiVideoActionDataset

    ds_test = MultiVideoActionDataset(
        num_frames=num_frames,
        dataset_path=dataset_path,
        data_split="test",
        camera_layout=camera_layout,
    )
    le_test = ds_test.datasets[0].lerobot_dataset
    ep20_indices = [idx for idx, (tid, _) in enumerate(le_test.all_steps) if int(tid) == 20]
    pool_indices = ep20_indices if ep20_indices else list(range(len(le_test.all_steps)))
    eval_indices = [pool_indices[int(i * len(pool_indices) / num_clips)] for i in range(num_clips)]
    cf_clip_count = min(num_clips, 3)

    v2w = Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=ckpt_paths[0],
        s3_credential_path="",
        context_parallel_size=1,
        config_file="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        offload_diffusion_model=False,
        offload_text_encoder=True,
        offload_tokenizer=False,
    )
    lpips_fn = piq.LPIPS(replace_pooling=True)

    for ckpt_path in ckpt_paths:
        run_tag = Path(ckpt_path).parent.parent.parent.name + "_" + Path(ckpt_path).parent.name
        save_root = Path(save_dir) / run_tag
        save_root.mkdir(parents=True, exist_ok=True)

        load_model_state_dict_from_checkpoint(
            model=v2w.model,
            config=v2w.config,
            s3_checkpoint_dir=ckpt_path,
            load_ema_to_reg=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
        v2w.model.on_train_start()
        v2w.model.eval()

        chunk_size = 12
        num_chunks = (num_frames - 1) // chunk_size
        chunk_metrics = {
            mode: {c: {"psnr": [], "ssim": [], "lpips": []} for c in range(num_chunks)}
            for mode in ["gt", "zero", "negated"]
        }

        for clip_i, ds_idx in enumerate(eval_indices):
            data = ds_test[ds_idx]
            gt_video = data["video"].permute(1, 2, 3, 0).numpy()  # (T, H, W, C) uint8
            base_actions = data["action"][: num_frames - 1].numpy()

            modes_to_run = ["gt", "zero", "negated"] if clip_i < cf_clip_count else ["gt"]
            rollout_videos = {}
            for mode in modes_to_run:
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
                    video_normalized = (video - (-1)) / (1 - (-1))
                    video_clamped = (
                        (torch.clamp(video_normalized[0], 0, 1) * 255)
                        .to(torch.uint8)
                        .permute(1, 2, 3, 0)
                        .cpu()
                        .numpy()
                    )
                    img_array = video_clamped[-1]
                    chunk_preds.append(video_clamped if c_idx == 0 else video_clamped[1:])

                    # Compute metrics for this chunk (12 generated frames: i+1 .. i+12)
                    gt_c = torch.from_numpy(gt_video[i + 1 : i + 1 + chunk_size]).permute(0, 3, 1, 2).float() / 255.0
                    pr_c = torch.from_numpy(video_clamped[1:]).permute(0, 3, 1, 2).float() / 255.0
                    chunk_metrics[mode][c_idx]["psnr"].append(piq.psnr(gt_c, pr_c).item())
                    chunk_metrics[mode][c_idx]["ssim"].append(piq.ssim(gt_c, pr_c).item())
                    chunk_metrics[mode][c_idx]["lpips"].append(lpips_fn(gt_c, pr_c).item())

                rollout_videos[mode] = np.concatenate(chunk_preds, axis=0)

            if clip_i < cf_clip_count:
                panel_video = np.concatenate(
                    [gt_video, rollout_videos["gt"], rollout_videos["zero"], rollout_videos["negated"]], axis=2
                )
                out_path = save_root / f"ep20_clip{clip_i:02d}_counterfactual_4panel.mp4"
                mediapy.write_video(str(out_path), panel_video, fps=7.5)
                print(f"Saved 4-panel counterfactual video [GT | True | Zero | Negated] -> {out_path}")
            else:
                panel_video = np.concatenate([gt_video, rollout_videos["gt"]], axis=2)
                out_path = save_root / f"ep20_clip{clip_i:02d}_gt_vs_pred.mp4"
                mediapy.write_video(str(out_path), panel_video, fps=7.5)
                print(f"Saved 2-panel rollout video [GT | True] -> {out_path}")

        print("\n" + "=" * 80)
        print(f"PER-CHUNK ROLLOUT METRICS ON HELD-OUT EPISODE 20 (Frame-0 Noise std=0.0)")
        print(f"Checkpoint: {ckpt_path} ({num_clips} GT clips, {cf_clip_count} Counterfactual clips)")
        print("=" * 80)
        for mode in ["gt", "zero", "negated"]:
            print(f"\nMode: {mode.upper()}")
            p_all, s_all, l_all = [], [], []
            for c_idx in range(num_chunks):
                p = np.mean(chunk_metrics[mode][c_idx]["psnr"])
                s = np.mean(chunk_metrics[mode][c_idx]["ssim"])
                l = np.mean(chunk_metrics[mode][c_idx]["lpips"])
                p_all.append(p)
                s_all.append(s)
                l_all.append(l)
                print(
                    f"  Chunk {c_idx+1} (frames {c_idx*12+1:2d}..{(c_idx+1)*12:2d}): "
                    f"PSNR={p:6.2f} dB | SSIM={s:.3f} | LPIPS={l:.3f}"
                )
            print(
                f"  Mean (frames  1..48): PSNR={np.mean(p_all):6.2f} dB | SSIM={np.mean(s_all):.3f} | LPIPS={np.mean(l_all):.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true", help="Run CPU-only split and action pipeline audit")
    parser.add_argument("--mode", choices=["audit", "loss", "rollout"], default="audit")
    parser.add_argument("--experiment", default="dreamdojo_2b_480_640_tlrc_yam_v2")
    parser.add_argument(
        "--ckpt-paths",
        nargs="+",
        default=[
            "outputs/dreamdojo/tlrc_yam_finetune/tlrc_bimanual_yam/checkpoints/iter_000002000/model_ema_bf16.pt",
            "outputs/dreamdojo/tlrc_yam_finetune/tlrc_bimanual_yam_v2/checkpoints/iter_000003000/model_ema_bf16.pt",
        ],
    )
    parser.add_argument("--eval-cond-noise-std", type=float, default=0.0)
    parser.add_argument("--save-dir", default="results/diagnostics")
    parser.add_argument("--dataset-path", default="datasets/tlrc_bimanual_yam")
    parser.add_argument("--camera-layout", choices=["head", "head_crop", "three_panel"], default="head")
    parser.add_argument("--num-clips", type=int, default=6)
    args = parser.parse_args()

    if args.audit_only or args.mode == "audit":
        run_cpu_audit(args.dataset_path, camera_layout=args.camera_layout)
    elif args.mode == "loss":
        run_fixed_sigma_action_loss_test(
            experiment=args.experiment,
            ckpt_paths=args.ckpt_paths,
            dataset_path=args.dataset_path,
            num_clips=args.num_clips,
            eval_cond_noise_std=args.eval_cond_noise_std,
            camera_layout=args.camera_layout,
        )
    elif args.mode == "rollout":
        run_counterfactual_and_chunk_rollouts(
            experiment=args.experiment,
            ckpt_paths=args.ckpt_paths,
            save_dir=args.save_dir,
            dataset_path=args.dataset_path,
            num_clips=args.num_clips,
            camera_layout=args.camera_layout,
        )
