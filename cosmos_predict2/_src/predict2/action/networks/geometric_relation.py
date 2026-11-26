# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
3D Multi-View Geometric Relation Attention Modules for DreamDojo (Exp 4a).

Adapts Relation Networks (Hu et al., CVPR 2018, arXiv:1711.11575) to calibrated 3-view
bimanual robotics (`head`, `left_wrist`, `right_wrist`):
  1. `PluckerPatchEncoder`: encodes per-latent-frame 6D Plucker rays + camera view ID
     and injects them into DiT patch tokens via a zero-init output projection.
  2. `EpipolarCrossViewRelationBlock`: per-frame 1200x1200 cross-view attention biased by
     exact 3D skew-ray distance `d_ray` (with parallel-ray fallback), continuous signed
     distance to the workspace box `sdf_ws`, signed ray depth `min_z`, and viewing angle
     `cos_theta`, initialized with a multi-scale per-head Gaussian epipolar kernel
     `-alpha_h * d_ray^2` and separate same-view bias path.

All layer names are chosen so PEFT LoRA target patterns (`q_proj,k_proj,v_proj,output_proj,
mlp.layer1,mlp.layer2,fc1,fc2`) do not double-wrap these modules; instead they are unfrozen
at full rank via `lora_unfreeze_modules` (`plucker_encoder,epipolar_blocks`).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


def encode_epipolar_sinusoidal(
    d_ray: torch.Tensor,
    cos_theta: torch.Tensor,
    sdf_ws: torch.Tensor,
    min_z: torch.Tensor,
    freqs: Tuple[float, ...] = (1.0, 3.0, 9.0),
) -> torch.Tensor:
    """
    Encode 4 continuous pairwise 3D geometric scalars `(d_ray, cos_theta, sdf_ws, min_z)`
    using sinusoidal basis functions (Hu et al., arXiv:1711.11575 Eq. 5).

    Args:
        d_ray, cos_theta, sdf_ws, min_z: `[BT, N_q, N_k]` float32 tensors
        freqs: log-spaced angular frequency multipliers (default 3 freqs -> 24 features)

    Returns:
        `[BT, N_q, N_k, 4 * len(freqs) * 2]` float32 tensor.
    """
    # Stack normalized continuous features: [BT, N_q, N_k, 4]
    feats = torch.stack(
        [
            d_ray * math.pi,
            cos_theta * math.pi,
            sdf_ws * math.pi,
            min_z * math.pi,
        ],
        dim=-1,
    )
    parts = []
    for f in freqs:
        ang = feats * f
        parts.append(torch.sin(ang))
        parts.append(torch.cos(ang))
    return torch.cat(parts, dim=-1)


class PluckerPatchEncoder(nn.Module):
    """
    Per-patch 6D Plucker ray + 3-view ID encoder added to DiT patch embeddings.

    Zero-init ONLY on the final output projection (`ray_out`), keeping `ray_in` and `view_embed`
    non-zero so step 0 is an exact identity while gradients flow cleanly without a zero-init trap.
    """

    def __init__(
        self,
        model_channels: int,
        hidden_dim: int = 256,
        num_views: int = 3,
    ) -> None:
        super().__init__()
        self.model_channels = model_channels
        self.hidden_dim = hidden_dim
        self.num_views = num_views

        self.ray_norm = nn.LayerNorm(6)
        self.ray_in = nn.Linear(6, hidden_dim, bias=True)
        self.view_embed = nn.Embedding(num_views, hidden_dim)
        self.ray_act = nn.GELU(approximate="tanh")
        self.ray_out = nn.Linear(hidden_dim, model_channels, bias=True)

        if self.ray_in.weight.device.type != "meta":
            self.init_weights()

    def init_weights(self) -> None:
        self.ray_norm.reset_parameters()
        nn.init.trunc_normal_(self.ray_in.weight, std=1.0 / math.sqrt(6))
        nn.init.zeros_(self.ray_in.bias)
        nn.init.normal_(self.view_embed.weight, mean=0.0, std=0.02)
        # Zero-init ONLY the residual output projection for exact step-0 identity
        nn.init.zeros_(self.ray_out.weight)
        nn.init.zeros_(self.ray_out.bias)

    def forward(self, plucker_B_T_H_W_6: torch.Tensor, view_ids_H_W: torch.Tensor) -> torch.Tensor:
        """
        Args:
            plucker_B_T_H_W_6: `[B, T, H, W, 6]` float32/bf16 Plucker rays `(d, o x d)`
            view_ids_H_W:      `[H, W]` long view IDs in `{0, 1, 2}`
        Returns:
            `[B, T, H, W, model_channels]` additive token embedding.
        """
        dtype = self.ray_in.weight.dtype
        h = self.ray_in(self.ray_norm(plucker_B_T_H_W_6.to(dtype)))
        v = self.view_embed(view_ids_H_W.to(h.device))[None, None, :, :, :].to(dtype)
        return self.ray_out(self.ray_act(h + v))


class EpipolarCrossViewRelationBlock(nn.Module):
    """
    Per-frame 3D Epipolar Relation Attention Block (Hu et al., arXiv:1711.11575).

    Operates independently per latent frame `t` over the `N = H * W = 1,200` multi-view spatial
    tokens, combining content similarity with a multi-head 3D geometric attention bias:
      - Cross-view pairs (`view_m != view_n`):
          B_G[h, m, n] = -softplus(log_alpha_h) * d_ray^2
                         + geom_mlp(Sinusoidal(d_ray, cos_theta, sdf_ws, min_z))[h]
                         + view_pair_bias[pair_id, h]
      - Same-view pairs (`view_m == view_n`):
          B_G[h, m, n] = same_view_bias[view_id, h]
        (separate bias path so degenerate same-camera d_ray == 0 never pollutes cross-view matching).

    Zero-init ONLY on the final residual output projection (`rel_out`), while `rel_q/k/v`,
    `geom_w1`, `geom_w2` (tiny 1e-3 init), and log-spaced `log_alpha` remain non-zero so:
      - Step 0 is an exact identity (`x_out == x_in`)
      - Step 0 immediately produces non-zero gradients on `rel_out`, and step 1 produces non-zero
        gradients on every internal parameter (no zero-init gradient trap).
    """

    def __init__(
        self,
        model_channels: int,
        inner_dim: int = 512,
        num_heads: int = 8,
        geom_hidden_dim: int = 32,
        geom_freqs: Tuple[float, ...] = (1.0, 3.0, 9.0),
        bias_chunk_size: int = 300,
        use_grad_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        assert inner_dim % num_heads == 0, f"inner_dim={inner_dim} must be divisible by num_heads={num_heads}"
        self.model_channels = model_channels
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.geom_hidden_dim = geom_hidden_dim
        self.geom_freqs = tuple(geom_freqs)
        self.geom_in_dim = 4 * len(self.geom_freqs) * 2  # 24 for 3 freqs
        self.bias_chunk_size = int(bias_chunk_size)
        self.use_grad_checkpoint = bool(use_grad_checkpoint)

        self.pre_norm = nn.LayerNorm(model_channels)
        # Distinct names so PEFT LoRA targets never double-wrap these layers
        self.rel_q = nn.Linear(model_channels, inner_dim, bias=False)
        self.rel_k = nn.Linear(model_channels, inner_dim, bias=False)
        self.rel_v = nn.Linear(model_channels, inner_dim, bias=False)
        self.q_rms = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_rms = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.rel_out = nn.Linear(inner_dim, model_channels, bias=True)

        # Geometric relation bias parameters (Hu et al., arXiv:1711.11575)
        self.log_alpha = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
        self.geom_w1 = nn.Linear(self.geom_in_dim, geom_hidden_dim, bias=True)
        self.geom_w2 = nn.Linear(geom_hidden_dim, num_heads, bias=True)
        # Separate learned offsets for cross-view pairs (indices 3, 4, 5) and same-view pairs (0, 1, 2)
        self.view_pair_bias = nn.Parameter(torch.empty(6, num_heads, dtype=torch.float32))

        if self.rel_q.weight.device.type != "meta":
            self.init_weights()

    def init_weights(self) -> None:
        self.pre_norm.reset_parameters()
        std_in = 1.0 / math.sqrt(self.model_channels)
        nn.init.trunc_normal_(self.rel_q.weight, std=std_in, a=-3 * std_in, b=3 * std_in)
        nn.init.trunc_normal_(self.rel_k.weight, std=std_in, a=-3 * std_in, b=3 * std_in)
        nn.init.trunc_normal_(self.rel_v.weight, std=std_in, a=-3 * std_in, b=3 * std_in)
        self.q_rms.reset_parameters()
        self.k_rms.reset_parameters()

        # Multi-scale Gaussian-kernel epipolar init: softplus(log_alpha) spans [0.5 .. 250.0] m^-2
        # (widths from ~1.4 m broad context down to ~6 cm tight epipolar matching across heads)
        target_alphas = torch.logspace(math.log10(0.5), math.log10(250.0), self.num_heads, dtype=torch.float32)
        # Inverse softplus: log(exp(a) - 1), numerically stable for large a
        inv_sp = torch.where(target_alphas > 20.0, target_alphas, torch.log(torch.expm1(target_alphas).clamp(min=1e-6)))
        with torch.no_grad():
            self.log_alpha.copy_(inv_sp.to(device=self.log_alpha.device, dtype=self.log_alpha.dtype))
            # Initialize same-view bias (0,1,2) to -2.0 so block focuses on cross-view transfer initially,
            # and cross-view pair offsets (3,4,5) to 0.0.
            self.view_pair_bias.zero_()
            self.view_pair_bias[:3, :].fill_(-2.0)

        nn.init.kaiming_uniform_(self.geom_w1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.geom_w1.bias)
        # Small non-zero init on geom_w2 avoids zero-init gradient trap while preserving Gaussian prior
        nn.init.normal_(self.geom_w2.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.geom_w2.bias)

        # Zero-init ONLY the final residual output projection -> exact step-0 identity!
        nn.init.zeros_(self.rel_out.weight)
        nn.init.zeros_(self.rel_out.bias)

    def compute_attention_bias(
        self,
        geom: Dict[str, torch.Tensor],
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Compute per-head geometric attention bias `[BT, num_heads, N, N]` in `out_dtype`.
        Chunks query dimension `N` through the sinusoidal MLP to keep peak activation memory low.
        """
        d_ray = geom["d_ray"]  # [BT, N, N] fp32
        cos_theta = geom["cos_theta"]
        sdf_ws = geom["sdf_ws"]
        min_z = geom["min_z"]
        same_view = geom["same_view"]  # [N, N] bool
        view_pair_id = geom["view_pair_id"]  # [N, N] long

        bt, n, _ = d_ray.shape
        device = d_ray.device
        w_dtype = self.geom_w1.weight.dtype

        # Multi-scale Gaussian kernel weights: [1, H, 1, 1]
        alpha = F.softplus(self.log_alpha.float()).view(1, self.num_heads, 1, 1)  # > 0
        # Pair offset lookup: [N, N, H] -> [1, H, N, N]
        pair_bias = self.view_pair_bias[view_pair_id].permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=out_dtype)

        # Chunk over query tokens to keep intermediate `[BT, chunk, N, 24]` tiny (~150 MB)
        mlp_bias_chunks = []
        step = max(1, self.bias_chunk_size)
        for q0 in range(0, n, step):
            q1 = min(n, q0 + step)
            sin_emb = encode_epipolar_sinusoidal(
                d_ray[:, q0:q1, :],
                cos_theta[:, q0:q1, :],
                sdf_ws[:, q0:q1, :],
                min_z[:, q0:q1, :],
                freqs=self.geom_freqs,
            ).to(w_dtype)
            # [BT, chunk, N, H] -> [BT, H, chunk, N]
            h = F.gelu(self.geom_w1(sin_emb), approximate="tanh")
            b_chunk = self.geom_w2(h).permute(0, 3, 1, 2).to(out_dtype)
            mlp_bias_chunks.append(b_chunk)
        mlp_bias = torch.cat(mlp_bias_chunks, dim=2)  # [BT, H, N, N]

        # Cross-view bias: -alpha_h * d_ray^2 + MLP + view_pair_offset
        d_sq = (d_ray * d_ray).unsqueeze(1)  # [BT, 1, N, N] fp32
        cross_bias = (-alpha * d_sq).to(out_dtype) + mlp_bias + pair_bias

        # Same-view pairs use their separate learned per-view/per-head bias only
        same_mask = same_view.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        attn_bias = torch.where(same_mask, pair_bias.expand_as(cross_bias), cross_bias)
        return attn_bias

    def _forward_impl(
        self,
        x_B_T_H_W_D: torch.Tensor,
        d_ray: torch.Tensor,
        cos_theta: torch.Tensor,
        sdf_ws: torch.Tensor,
        min_z: torch.Tensor,
        same_view: torch.Tensor,
        view_pair_id: torch.Tensor,
    ) -> torch.Tensor:
        b, t, h_tok, w_tok, d = x_B_T_H_W_D.shape
        n = h_tok * w_tok
        bt = b * t

        x_flat = x_B_T_H_W_D.reshape(bt, n, d)
        x_norm = self.pre_norm(x_flat.to(self.pre_norm.weight.dtype)).to(self.rel_q.weight.dtype)

        q = self.rel_q(x_norm).view(bt, n, self.num_heads, self.head_dim)
        k = self.rel_k(x_norm).view(bt, n, self.num_heads, self.head_dim)
        v = self.rel_v(x_norm).view(bt, n, self.num_heads, self.head_dim)

        q = self.q_rms(q.to(self.q_rms.weight.dtype)).to(v.dtype).transpose(1, 2)  # [BT, H, N, Dh]
        k = self.k_rms(k.to(self.k_rms.weight.dtype)).to(v.dtype).transpose(1, 2)  # [BT, H, N, Dh]
        v = v.transpose(1, 2)  # [BT, H, N, Dh]

        geom = {
            "d_ray": d_ray,
            "cos_theta": cos_theta,
            "sdf_ws": sdf_ws,
            "min_z": min_z,
            "same_view": same_view,
            "view_pair_id": view_pair_id,
        }
        attn_bias = self.compute_attention_bias(geom, out_dtype=q.dtype)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)  # [BT, H, N, Dh]
        attn_out = attn_out.transpose(1, 2).contiguous().view(bt, n, self.inner_dim)
        delta = self.rel_out(attn_out.to(self.rel_out.weight.dtype)).to(x_B_T_H_W_D.dtype).view(b, t, h_tok, w_tok, d)
        return x_B_T_H_W_D + delta

    def forward(self, x_B_T_H_W_D: torch.Tensor, geom: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.training and self.use_grad_checkpoint and x_B_T_H_W_D.requires_grad:
            return grad_checkpoint(
                self._forward_impl,
                x_B_T_H_W_D,
                geom["d_ray"],
                geom["cos_theta"],
                geom["sdf_ws"],
                geom["min_z"],
                geom["same_view"],
                geom["view_pair_id"],
                use_reentrant=False,
            )
        return self._forward_impl(
            x_B_T_H_W_D,
            geom["d_ray"],
            geom["cos_theta"],
            geom["sdf_ws"],
            geom["min_z"],
            geom["same_view"],
            geom["view_pair_id"],
        )
