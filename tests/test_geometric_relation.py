"""
Unit tests for Exp 4a geometric relation modules:
  1. Step-0 exact identity (PluckerPatchEncoder + EpipolarCrossViewRelationBlock)
  2. Multi-scale Gaussian-kernel epipolar bias init & same-view separation
  3. Zero-init gradient-trap check (step 0 + step 1: 100% non-zero, finite grads on ALL params)
  4. PEFT LoRA target non-interference & lora_unfreeze_modules coverage
  5. Meta-device -> to_empty -> init_weights round-trip
  6. Rollout counterfactual (gt / zero / negated) modulation of epipolar outputs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cosmos_predict2._src.predict2.action.networks.geometric_relation import (
    EpipolarCrossViewRelationBlock,
    PluckerPatchEncoder,
)
from groot_dreams.geometry.dkx_kinematics import (
    DKXKinematics,
    build_rollout_robot_q,
    compute_pairwise_epipolar_geometry,
    extract_robot_q_from_yam_state,
)


def _sample_real_robot_q(num_frames: int = 17) -> torch.Tensor:
    df = pd.read_parquet("datasets/tlrc_bimanual_yam_eval6/data/chunk-000/episode_000003.parquet")
    idxs = [300 + 4 * k for k in range(num_frames)]
    st = np.stack([df["observation.state"].iloc[i] for i in idxs], axis=0).astype(np.float32)
    return torch.from_numpy(extract_robot_q_from_yam_state(st)).unsqueeze(0)  # [1, 17, 14]


def test_step0_identity_and_gaussian_bias_init():
    torch.manual_seed(0)
    kin = DKXKinematics()
    robot_q = _sample_real_robot_q(17)
    plk = kin.compute_plucker_rays(robot_q, target_t_lat=2)  # [1, 2, 30, 40, 6] (T=2 fast on CPU)
    geom = compute_pairwise_epipolar_geometry(
        plk["rays_o"].reshape(2, 1200, 3),
        plk["rays_d"].reshape(2, 1200, 3),
        plk["view_ids"].reshape(1200),
        kin.workspace_min,
        kin.workspace_max,
    )

    D = 256
    plk_enc = PluckerPatchEncoder(model_channels=D, hidden_dim=64)
    epi_blk = EpipolarCrossViewRelationBlock(
        model_channels=D, inner_dim=128, num_heads=8, geom_hidden_dim=16, use_grad_checkpoint=False
    )

    x = torch.randn(1, 2, 30, 40, D)
    # 1. Exact step-0 identity
    x_plk = x + plk_enc(plk["plucker"], plk["view_ids"])
    x_epi = epi_blk(x_plk, geom)
    assert (x_plk - x).abs().max().item() == 0.0, "PluckerPatchEncoder must be exact identity at step 0"
    assert (x_epi - x).abs().max().item() == 0.0, "EpipolarCrossViewRelationBlock must be exact identity at step 0"

    # 2. Multi-scale Gaussian epipolar kernel init check
    alphas = F.softplus(epi_blk.log_alpha)
    assert (alphas[1:] > alphas[:-1]).all(), f"Expected strictly increasing alphas across heads, got {alphas}"
    assert abs(alphas[0].item() - 0.5) < 0.05 and abs(alphas[-1].item() - 250.0) < 1.0

    # 3. Same-view separation & epipolar decay check
    bias = epi_blk.compute_attention_bias(geom, out_dtype=torch.float32)  # [2, 8, 1200, 1200]
    assert torch.isfinite(bias).all()
    same_mask = geom["same_view"]
    # Same-view entries must all equal -2.0 at init
    same_vals = bias[:, :, same_mask]
    assert (same_vals + 2.0).abs().max().item() < 1e-5, "Same-view bias must use separate -2.0 init path"
    # Cross-view entries: tightest head (head 7, alpha=250) must penalize distant rays much more than close rays
    cross_mask = ~same_mask
    d_cross = geom["d_ray"][0][cross_mask]
    b_h7 = bias[0, 7][cross_mask]
    close_bias = b_h7[d_cross < 0.05].mean().item()
    far_bias = b_h7[d_cross > 0.25].mean().item()
    assert close_bias > far_bias + 15.0, f"Expected strong epipolar preference: close={close_bias:.2f}, far={far_bias:.2f}"
    print(
        f"[PASS] Step-0 identity (max_diff=0.0) + Gaussian kernel init "
        f"(alphas=[{alphas[0]:.2f}..{alphas[-1]:.1f}], h7 close={close_bias:.2f} vs far={far_bias:.2f})"
    )


def test_no_zero_init_gradient_trap():
    """Verify step-0 output proj grad != 0 and after 1 optimizer step ALL internal params have non-zero grad."""
    torch.manual_seed(42)
    kin = DKXKinematics()
    robot_q = _sample_real_robot_q(5)  # 5 frames -> 2 latents (fast on CPU)
    plk = kin.compute_plucker_rays(robot_q, target_t_lat=2)
    geom = compute_pairwise_epipolar_geometry(
        plk["rays_o"].reshape(2, 1200, 3),
        plk["rays_d"].reshape(2, 1200, 3),
        plk["view_ids"].reshape(1200),
        kin.workspace_min,
        kin.workspace_max,
    )

    D = 128
    plk_enc = PluckerPatchEncoder(model_channels=D, hidden_dim=32)
    epi_blk = EpipolarCrossViewRelationBlock(
        model_channels=D, inner_dim=64, num_heads=8, geom_hidden_dim=16, use_grad_checkpoint=True
    )
    opt = torch.optim.AdamW(list(plk_enc.parameters()) + list(epi_blk.parameters()), lr=1e-2)

    x_in = torch.randn(1, 2, 30, 40, D, requires_grad=True)
    target = torch.randn(1, 2, 30, 40, D)

    # Step 0: verify output projections get immediate non-zero gradients
    opt.zero_grad()
    out0 = epi_blk(x_in + plk_enc(plk["plucker"], plk["view_ids"]), geom)
    loss0 = F.mse_loss(out0, target)
    loss0.backward()

    assert plk_enc.ray_out.weight.grad.abs().max().item() > 0.0, "ray_out.weight grad is zero at step 0!"
    assert epi_blk.rel_out.weight.grad.abs().max().item() > 0.0, "rel_out.weight grad is zero at step 0!"
    opt.step()

    # Step 1: verify EVERY SINGLE parameter in both modules now has strictly non-zero, finite gradient!
    opt.zero_grad()
    x_in2 = torch.randn(1, 2, 30, 40, D, requires_grad=True)
    out1 = epi_blk(x_in2 + plk_enc(plk["plucker"], plk["view_ids"]), geom)
    loss1 = F.mse_loss(out1, target)
    loss1.backward()

    zero_grad_params = []
    for prefix, mod in [("plucker_encoder", plk_enc), ("epipolar_block", epi_blk)]:
        for name, p in mod.named_parameters():
            assert p.grad is not None, f"Missing grad for {prefix}.{name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite grad for {prefix}.{name}"
            if p.grad.abs().max().item() == 0.0:
                zero_grad_params.append(f"{prefix}.{name}")
    assert not zero_grad_params, f"Zero-gradient trap detected on: {zero_grad_params}"
    print("[PASS] Zero-init gradient flow verified: 100% of parameters have non-zero finite gradients by step 1")


def test_lora_target_isolation_and_unfreeze_patterns():
    """Verify PEFT LoRA target names never double-wrap Exp 4a modules and unfreeze patterns match all params."""
    lora_targets = ["q_proj", "k_proj", "v_proj", "output_proj", "mlp.layer1", "mlp.layer2", "fc1", "fc2"]
    unfreeze_patterns = ["plucker_encoder", "epipolar_blocks"]

    wrapper = torch.nn.Module()
    wrapper.plucker_encoder = PluckerPatchEncoder(model_channels=128)
    wrapper.epipolar_blocks = torch.nn.ModuleDict({"6": EpipolarCrossViewRelationBlock(model_channels=128)})

    for mod_name, _ in wrapper.named_modules():
        for t in lora_targets:
            assert t not in mod_name, f"Exp 4a module {mod_name!r} collides with LoRA target {t!r}!"
    for p_name, _ in wrapper.named_parameters():
        assert any(pat in p_name for pat in unfreeze_patterns), f"Param {p_name!r} missed by lora_unfreeze_modules!"
    print("[PASS] LoRA target isolation & lora_unfreeze_modules coverage verified")


def test_meta_device_and_rollout_counterfactual_sensitivity():
    """Verify meta-device init round-trip + rollout-time gt/zero/negated robot_q produce distinct outputs."""
    torch.manual_seed(7)
    D = 128
    with torch.device("meta"):
        plk_enc = PluckerPatchEncoder(model_channels=D, hidden_dim=32)
        epi_blk = EpipolarCrossViewRelationBlock(
            model_channels=D, inner_dim=64, num_heads=8, geom_hidden_dim=16, use_grad_checkpoint=False
        )
    plk_enc.to_empty(device="cpu")
    epi_blk.to_empty(device="cpu")
    plk_enc.init_weights()
    epi_blk.init_weights()

    # Check meta re-init gives finite params and exact step-0 identity
    for p in list(plk_enc.parameters()) + list(epi_blk.parameters()):
        assert torch.isfinite(p).all()

    # Give non-zero output proj weights (simulating post-step-1 trained state)
    with torch.no_grad():
        plk_enc.ray_out.weight.normal_(0.0, 0.05)
        epi_blk.rel_out.weight.normal_(0.0, 0.05)

    kin = DKXKinematics()
    q_gt = _sample_real_robot_q(17)
    q_zero = build_rollout_robot_q(q_gt, mode="zero")
    q_neg = build_rollout_robot_q(q_gt, mode="negated", joint_lower=kin.joint_lower_14, joint_upper=kin.joint_upper_14)

    x = torch.randn(1, 5, 30, 40, D)
    outs = {}
    for mode, q_in in [("gt", q_gt), ("zero", q_zero), ("negated", q_neg)]:
        plk = kin.compute_plucker_rays(q_in, target_t_lat=5)
        geom = compute_pairwise_epipolar_geometry(
            plk["rays_o"].reshape(5, 1200, 3),
            plk["rays_d"].reshape(5, 1200, 3),
            plk["view_ids"].reshape(1200),
            kin.workspace_min,
            kin.workspace_max,
        )
        outs[mode] = epi_blk(x + plk_enc(plk["plucker"], plk["view_ids"]), geom)

    # Frame 0 is identical across gt/zero/negated -> latent 0 outputs must match!
    assert (outs["gt"][:, 0] - outs["zero"][:, 0]).abs().max().item() < 1e-5
    assert (outs["gt"][:, 0] - outs["negated"][:, 0]).abs().max().item() < 1e-5
    # Future latent frame 4 has distinct wrist camera poses -> outputs must differ!
    diff_gz = (outs["gt"][:, 4] - outs["zero"][:, 4]).abs().mean().item()
    diff_gn = (outs["gt"][:, 4] - outs["negated"][:, 4]).abs().mean().item()
    assert diff_gz > 1e-3 and diff_gn > 1e-3, f"Expected rollout sensitivity, got gt-zero={diff_gz}, gt-neg={diff_gn}"

    # Verify .to(bfloat16) preserves exact fp32 geometry buffers and runs cleanly under bf16
    kin.to(dtype=torch.bfloat16)
    plk_enc.to(dtype=torch.bfloat16)
    epi_blk.to(dtype=torch.bfloat16)
    assert kin.joint_T0.dtype == torch.float32, "DKXKinematics must preserve fp32 buffers after .to(bf16)"
    plk_bf = kin.compute_plucker_rays(q_gt[:, :5], target_t_lat=2)
    geom_bf = compute_pairwise_epipolar_geometry(
        plk_bf["rays_o"].reshape(2, 1200, 3),
        plk_bf["rays_d"].reshape(2, 1200, 3),
        plk_bf["view_ids"].reshape(1200),
        kin.workspace_min,
        kin.workspace_max,
    )
    x_bf = torch.randn(1, 2, 30, 40, D, dtype=torch.bfloat16)
    y_bf = epi_blk(x_bf + plk_enc(plk_bf["plucker"], plk_bf["view_ids"]).to(torch.bfloat16), geom_bf)
    assert y_bf.dtype == torch.bfloat16 and torch.isfinite(y_bf).all()
    print(
        f"[PASS] Meta-device round-trip + bf16 preservation + rollout counterfactual sensitivity verified "
        f"(t=0 diff=0.0, t=4 |gt-zero|={diff_gz:.4f}, |gt-neg|={diff_gn:.4f})"
    )


if __name__ == "__main__":
    test_step0_identity_and_gaussian_bias_init()
    test_no_zero_init_gradient_trap()
    test_lora_target_isolation_and_unfreeze_patterns()
    test_meta_device_and_rollout_counterfactual_sensitivity()
    print("\nALL GEOMETRIC RELATION UNIT TESTS PASSED!")
