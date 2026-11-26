# Workstation Handoff: DreamDojo Bimanual 3-Panel (`Exp 3` & `Exp 4a`)

> Generated on Karolina HPC (`login3.karolina.it4i.cz`) on 2026-11-26 before handing work back to the local workstation (non-billed).

---

## 1. Branch, Checkpoints & Quick Restore on Workstation

- **Git Branch:** `feat/exp4a-3d-epipolar-relation` (branched cleanly from `feat/history-1to1crop-eval6`)
- **Hugging Face Weights Repo (private):** `simonhofmann/dreamdojo-dkx-3panel`
  - `exp3_3panel_1to1_hist/model_ema_bf16.pt` (`4.1 GB`, best @ `iter_000006000`, `det_val_mse=0.12620`)
  - `exp4a_epipolar_3k/model_ema_bf16.pt` (`4.2 GB`, warm-started from Exp 3 `iter_000006000`, best @ `iter_000003000`, `det_val_mse=0.12550`)
  - `eval6_benchmark/summary.json` (full eval6 metrics across Zero-Shot, Exp 3, Exp 4a)
  - `eval6_benchmark/selected_clips.json` (canonical 6 held-out evaluation clip indices)

### Restore Weights Locally on Workstation
```bash
git fetch origin && git checkout feat/exp4a-3d-epipolar-relation

python - << 'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

repo = "simonhofmann/dreamdojo-dkx-3panel"
targets = {
    "exp3_3panel_1to1_hist/model_ema_bf16.pt":
        "outputs/dreamdojo/tlrc_yam_finetune/tlrc_bimanual_yam_3panel_1to1_hist/checkpoints/best/model_ema_bf16.pt",
    "exp4a_epipolar_3k/model_ema_bf16.pt":
        "outputs/dreamdojo/tlrc_yam_finetune/tlrc_bimanual_yam_3panel_1to1_hist_epipolar_4a/checkpoints/best/model_ema_bf16.pt",
    "eval6_benchmark/summary.json":
        "results/eval6_benchmark/summary.json",
    "eval6_benchmark/selected_clips.json":
        "results/eval6_benchmark/selected_clips.json",
}
for src, dst in targets.items():
    p = hf_hub_download(repo_id=repo, filename=src, repo_type="model")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)
    print("Restored:", dst)
PY
```

---

## 2. Verified Robot & 3-Panel Geometry (`DKX`, `0.00028 mm` FK)

- **Robot identity:** **DKX bimanual desk robot** (YAM-like; uses DreamDojo's `"yam"` action slice `action[:, 101:147]`).
- **Assets committed:**
  - `assets/dkx.urdf` (complete kinematics: `base → left/right arm links 1..6-7, tool0, fingers, left/right_wrist_camera_optical_frame`, plus `head_camera_optical_frame` pitched 45° down at `(0.006, 0, 0.485)`).
  - `assets/dkx_camera_intrinsics.json` (factory ZED X One GS `1920×1200` Kannala-Brandt `K`, `D`, and exact `cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(..., balance=0.0)` rectified pinhole `K_rect_1920x1200`, extracted from `20261029_144001_ep0008.mcap` — the 95 MB `.mcap` itself is intentionally excluded from git).
- **Verified `three_panel_1to1` (`480×640`) pixel & token layout:**
  - Dataset videos are already rectified pinhole images (`1280×800` head, `640×400` wrists).
  - `three_panel_1to1` takes exact 1:1 integer crops (`640×288` head at `y ∈ [0, 288)`, `320×192` left wrist at `y ∈ [288, 480), x ∈ [0, 320)`, `320×192` right wrist at `x ∈ [320, 640)`).
  - Rectified `K_panel`:
    - `head`: `fx=160.73, fy=160.77, cx=313.30, cy=48.66`
    - `left_wrist`: `fx=79.75, fy=79.76, cx=165.56, cy=37.57`
    - `right_wrist`: `fx=80.17, fy=80.21, cx=156.70, cy=59.29`
  - **Spatial DiT tokenization (`8×` Wan2.1 VAE × `2×` DiT patchify = `16×16` px/token):** `30×40 = 1,200` tokens/frame. Panel boundaries (`y=288`, `x=320`) divide evenly by `16` (`18`, `20`) → **zero straddling tokens** (`720` head + `240` left_wrist + `240` right_wrist).
  - **Temporal Wan2.1 causal VAE alignment (`17 → 5` latents):** `t_lat=0` ← pixel `[0]`; `t_lat=k` (`k=1..4`) ← pixel `[4k-3 .. 4k]` (aligned via `align_robot_q_to_latents` in `groot_dreams/geometry/dkx_kinematics.py`).
  - **Dataset size (`tlrc_bimanual_yam_multicam`):** 57 episodes, 64,826 raw 30 Hz frames (~36 min), 60,950 sliding 17-frame clip windows (~928 non-overlapping 2.27 s segments).

---

## 3. Code Changes Included in This Branch

### A. Portable Pipeline & Eval Bug Fixes (from Exp 3 audit)
- `sitecustomize.py`, `cosmos_predict2/_src/predict2/callbacks/every_n_draw_sample.py`: portable `imageio_ffmpeg` fallback for `mediapy` when system `ffmpeg` is missing (no-op if system `ffmpeg` exists).
- `cosmos_predict2/_src/predict2/callbacks/compile_tokenizer.py`: skip `torch.compile` gracefully if `TORCH_COMPILE_DISABLE=1` or no C compiler is present.
- `launch_finetune_tlrc.sh`: auto-fallback to `WANDB_MODE=offline` when no W&B key/`~/.netrc` is configured.
- `configs/2b_480_640_tlrc_yam_3panel_1to1_hist.yaml`: fixed Hydra dict-merge bug by explicitly setting `conditional_frames_probs: {0: 0.0, 1: 0.25, 2: 0.75}`.
- `scripts/benchmark_twoarm_models.py`: fixed `ds_hist5` episode indexing via `(traj_id, base_idx - 16)` lookup map (`max_pixel_diff=0` on all 6 episodes).

### B. Exp 4a: 3D Plücker Encoder + Cross-View Epipolar Relation Attention (`arXiv:1711.11575`)
All new model flags default to `False` so Exp 1–3 configs/checkpoints remain 100% backward-compatible.
- `groot_dreams/geometry/dkx_kinematics.py`: meta-device-safe, `bf16`-safe (`_apply` preserves exact `float32` buffers) vectorized PyTorch FK, rectified `K_panel`, per-latent-frame 6D Plücker rays, rollout counterfactual `build_rollout_robot_q` (`gt`/`zero`/`negated`), and GEMM-based `compute_pairwise_epipolar_geometry` (with parallel-ray point-to-line fallback when `sin(theta) < 1e-2`, `d_ray` clamped to `1.5 m`, continuous workspace box SDF `sdf_ws`, signed ray depth `min_z`, and `same_view` mask).
- `groot_dreams/data/dataset.py`, `groot_dreams/dataloader.py`: export unnormalized `[T_pix, 14]` `robot_q` aligned 1-to-1 with kept video frames.
- `cosmos_predict2/_src/predict2/action/networks/geometric_relation.py`:
  - `PluckerPatchEncoder` (6D Plücker + 3-view embedding, zero-init on `ray_out` only).
  - `EpipolarCrossViewRelationBlock` (inserted after DiT blocks `{6, 13, 20, 26}`, `inner_dim=512`, `num_heads=8`, multi-scale Gaussian epipolar kernel init `alpha_h ∈ [0.5 .. 250.0] m^-2`, Hu et al. sinusoidal MLP, separate same-view bias path initialized to `-2.0`, zero-init on `rel_out` only, activation-checkpointed, distinct layer names so PEFT LoRA never double-wraps it).
- Wired through `action_conditioned_minimal_v1_lvg_dit.py`, `conditioner.py`, and `video2world.py` (keeping `robot_q` in `float32`).
- Configs: `configs/2b_480_640_tlrc_yam_3panel_1to1_hist_epipolar_4a.yaml` & `configs/2b_480_640_tlrc_yam_3panel_1to1_hist_ctrl9k.yaml` (control not yet run).
- `scripts/benchmark_twoarm_models.py`: passes rollout-time `robot_q` (`gt`/`zero`/`negated`) and adds `compute_crossview_3d_consistency` (`XV-Worst` & `XV-Agree` across All / Chk 1–2 / Chk 3–4) + wrist PSNR table.
- Unit tests (all passing): `tests/test_dkx_kinematics.py`, `tests/test_dataloader_robot_q.py`, `tests/test_geometric_relation.py`.

---

## 4. Quantitative Benchmark Summary (`eval6`, 49 frames / 4 chunks)

| Model | Shared ROI PSNR | SSIM | LPIPS | GT−Zero | GT−Neg | Chk 1→4 PSNR | Autogen-3 | TwoArm-3 | L-Wrist | R-Wrist | XV-Worst Ch3-4 (↓) | XV-Agree Ch3-4 (↓) |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **Zero-Shot** | `12.93` | `0.547` | `0.412` | `-1.68` | `+0.52` | `16.7→12.8→11.8→10.5` | `13.83` | `12.03` | — | — | — | — |
| **Exp 3 (6k)** | `22.68` | `0.834` | `0.153` | `+7.10` | `+8.36` | `25.7→22.9→21.6→20.5` | `25.43` | `19.93` | `18.43` | `18.25` | **`0.1438`** | **`0.0813`** |
| **Exp 4a (+3k)** | **`22.92`** | **`0.838`** | **`0.149`** | **`+7.38`** | **`+8.69`** | **`25.8→23.2→21.9→20.8`** | **`25.74`** | **`20.09`** | **`18.76`** | `18.18` | `0.1523` | `0.0876` |

- **Exp 3 training (`5495531`):** 6,000 iters (~4h04m on 8×A100-40GB, 18.7 GB VRAM/GPU), `det_val_mse`: `0.1763 → 0.1262`.
- **Exp 4a training (`5498198`):** +3,000 iters warm-started from Exp 3 `iter_000006000` (~2h51m on 8×A100-40GB, 21.0 GB VRAM/GPU), `det_val_mse`: `0.126213` (iter 0 exact identity) → `0.12733` (250) → LR decay `5e-5→2.5e-5` @ 750 → `0.12589` (1250) → `0.12571` (1750) → LR decay `→1.25e-5` @ 2250 → `0.12548` (2750) / `0.12550` (3000).
- **Qualitative takeaway (user inspection of `[GT | Exp 3 | Exp 4a]` MP4s):** Exp 4a blurs out noticeably less than Exp 3 and commits to sharper, more physically grounded object/arm estimates (consistent with LPIPS improving `0.153 → 0.149` and `GT−Neg` improving `+8.36 → +8.69 dB`, while pixel-wise `XV-Worst` L1 penalizes sharp slightly shifted hypotheses more than blurry averages).

---

## 5. Key Diagnostic Findings & Recommended Next Steps on Workstation

### Root-Cause Finding from Exp 4a Weights
Inspecting `exp4a_epipolar_3k/model_ema_bf16.pt`:
- `plucker_encoder.ray_out.weight`: `rms = 0.00095`, `max = 0.0068`
- `epipolar_blocks.{6,13,20,26}.rel_out.weight`: `rms = 0.00071–0.00099`, `max = 0.0045`
- `log_alpha` (`[0.5 .. 250.0]`) and `view_pair_bias` (`same = -2.000`, `cross = 0.000`) barely moved.

**Why:** One shared LR (`5e-5`, prematurely decayed to `2.5e-5` @ iter 750 because restarting AdamW moments transiently bumped converged Exp 3 LoRA weights, then `1.25e-5` @ iter 2250 because `min_delta=0.0002` missed the `0.12589 → 0.12571` drop) was **too high** for converged Exp 3 LoRA weights and **10–20× too low** for brand-new zero-init 3D layers.

### Concrete Next Experiments (Free to Iterate Locally on Workstation)
1. **5-min Inference-Time Ablation on `exp4a_epipolar_3k`:** Run `ep03`–`ep05` with `plucker_encoder` + `epipolar_blocks` ON vs. bypassed (`robot_q=None`) to isolate how much of Exp 4a's sharper, grounded rollout comes from the 3D modules vs. the +3k LoRA steps.
2. **Differential LR for Warm-Started 3D Modules:** Give `plucker_encoder` + `epipolar_blocks` `lr = 3e-4–5e-4` (`wd = 0` on `log_alpha`/biases) and pretrained LoRA `lr = 1.5e-5–2.5e-5` (or freeze LoRA for 500 warmup steps), with `min_delta = 5e-5` and `warmup_skip_iters = 1000`.
3. **Robot-Agnostic Generalization Options:**
   - **Make Exp 4a 100% embodiment-agnostic:** Replace DKX's hardcoded `workspace_box_base` in `compute_pairwise_epipolar_geometry` with camera-frame / `min_z > 0` normalization so any calibrated multi-cam rig `(c2w, K)` uses identical weights.
   - **Calibration-free / URDF-free path:** Either learned uncalibrated cross-view relation blocks (same `1200×1200` architecture without FK/Plücker) or a pretrained feed-forward 3D prior (**VGGT / DUSt3R / DepthAnything**) as a 3D cross-view consistency loss or auxiliary conditioning.
   - **Option B (URDF depth/mask render conditioning):** Model-agnostic in pixel space (swap URDF per robot) for strongest explicit arm-pose lock across all 3 views.
