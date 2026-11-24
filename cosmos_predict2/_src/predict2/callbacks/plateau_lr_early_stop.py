"""
Deterministic Validation Plateau LR Decay & Early Stopping Callback for DreamDojo.

Why deterministic validation?
Standard diffusion/rectified-flow validation samples random noise levels `sigma` per batch,
creating high run-to-run variance that triggers false LR decays or premature early stops.
This callback reuses the already-encoded latents (`output_batch["x0"]`, `output_batch["condition"]`)
during validation and evaluates velocity MSE on a fixed `sigmas` grid with fixed per-batch RNG
seeds (zero sampling noise), then:
  1. Keeps LR flat after warmup until deterministic val MSE plateaus for `lr_patience` evals.
  2. Multiplies LR by `lr_factor` (updating `scheduler.base_lrs`, `initial_lr`, and active `lr`)
     down to `min_lr`, with a `cooldown` window after each decay.
  3. Once LR has reached `min_lr`, triggers clean early stopping after `early_stop_patience`
     consecutive non-improving evals by setting `self.trainer._should_stop = True`.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.distributed as dist
import wandb

from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.utils.callback import Callback


class PlateauLRAndEarlyStopping(Callback):
    def __init__(
        self,
        sigmas: Sequence[float] = (0.2, 0.5, 0.8),
        max_eval_batches: int = 16,
        seed: int = 10000,
        lr_factor: float = 0.5,
        lr_patience: int = 3,
        cooldown: int = 2,
        min_lr: float = 5.0e-6,
        early_stop_patience: int = 3,
        min_delta: float = 2.0e-4,
        warmup_skip_iters: int = 250,
    ) -> None:
        super().__init__()
        self.sigmas = [float(s) for s in sigmas]
        self.max_eval_batches = int(max_eval_batches)
        self.seed = int(seed)
        self.lr_factor = float(lr_factor)
        self.lr_patience = int(lr_patience)
        self.cooldown = int(cooldown)
        self.min_lr = float(min_lr)
        self.early_stop_patience = int(early_stop_patience)
        self.min_delta = float(min_delta)
        self.warmup_skip_iters = int(warmup_skip_iters)

        self.best_loss: float = float("inf")
        self.best_iter: int = -1
        self.lr_bad_evals: int = 0
        self.es_bad_evals: int = 0
        self.cooldown_counter: int = 0
        self.num_decays: int = 0

        self._val_step_idx: int = 0
        self._det_loss_sum: torch.Tensor | None = None
        self._det_count: torch.Tensor | None = None

    def on_validation_start(
        self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        self._val_step_idx = 0
        self._det_loss_sum = torch.tensor(0.0, device="cuda", dtype=torch.float64)
        self._det_count = torch.tensor(0.0, device="cuda", dtype=torch.float64)

    @torch.no_grad()
    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self._val_step_idx >= self.max_eval_batches:
            self._val_step_idx += 1
            return

        if "x0" not in output_batch or "condition" not in output_batch:
            self._val_step_idx += 1
            return

        x0 = output_batch["x0"].detach()
        condition = output_batch["condition"]
        B, _, T_lat, _, _ = x0.shape
        # Enforce deterministic conditioning frame count during validation:
        # 2 latent frames (5 pixel frames) when state_t >= 5 and max_num_conditional_frames >= 2, else 1.
        max_cond = getattr(model.config, "max_num_conditional_frames", 1)
        val_num_cond = 2 if (T_lat >= 5 and max_cond >= 2) else 1
        if hasattr(condition, "set_video_condition"):
            condition = condition.set_video_condition(
                gt_frames=x0,
                random_min_num_conditional_frames=val_num_cond,
                random_max_num_conditional_frames=val_num_cond,
                num_conditional_frames=val_num_cond,
            )

        batch_mse_sum = 0.0
        for s_idx, sigma_val in enumerate(self.sigmas):
            gen = torch.Generator(device=x0.device)
            gen.manual_seed(self.seed + self._val_step_idx * 100 + s_idx)
            eps = torch.randn(x0.shape, dtype=torch.float32, device=x0.device, generator=gen)

            sigma_t = torch.full((B, 1), sigma_val, dtype=torch.float32, device=x0.device)
            timestep_t = sigma_t * 1000.0
            xt, vt = model.rectified_flow.get_interpolation(eps, x0, sigma_t)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vt_pred = model.denoise(
                    noise=eps,
                    xt_B_C_T_H_W=xt.to(**model.tensor_kwargs),
                    timesteps_B_T=timestep_t,
                    condition=condition,
                )
            # Evaluate on future latent frames (excluding clean conditioning frames)
            mse = torch.mean((vt_pred[:, :, val_num_cond:].float() - vt[:, :, val_num_cond:].float()) ** 2).item()
            batch_mse_sum += mse

        mean_batch_mse = batch_mse_sum / len(self.sigmas)
        assert self._det_loss_sum is not None and self._det_count is not None
        self._det_loss_sum += mean_batch_mse * B
        self._det_count += float(B)
        self._val_step_idx += 1

    def _get_base_lr(self) -> float:
        scheduler = getattr(self.trainer, "_scheduler", None)
        optimizer = getattr(self.trainer, "_optimizer", None)
        if scheduler is not None and hasattr(scheduler, "base_lrs") and scheduler.base_lrs:
            return float(max(scheduler.base_lrs))
        if optimizer is not None and optimizer.param_groups:
            return float(max(pg["lr"] for pg in optimizer.param_groups))
        return 0.0

    def _get_active_lr(self) -> float:
        optimizer = getattr(self.trainer, "_optimizer", None)
        if optimizer is not None and optimizer.param_groups:
            return float(max(pg["lr"] for pg in optimizer.param_groups))
        return 0.0

    def _decay_lr(self) -> tuple[float, float]:
        optimizer = getattr(self.trainer, "_optimizer", None)
        scheduler = getattr(self.trainer, "_scheduler", None)
        assert optimizer is not None and scheduler is not None, "Trainer optimizer/scheduler not attached"

        old_base = float(max(scheduler.base_lrs))
        new_base_lrs = []
        for i, (base_lr, pg) in enumerate(zip(scheduler.base_lrs, optimizer.param_groups)):
            cur_mult = (pg["lr"] / base_lr) if base_lr > 0 else 1.0
            new_base = max(self.min_lr, float(base_lr) * self.lr_factor)
            new_base_lrs.append(new_base)
            pg["initial_lr"] = new_base
            pg["lr"] = new_base * cur_mult
        scheduler.base_lrs = new_base_lrs
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = [pg["lr"] for pg in optimizer.param_groups]
        self.num_decays += 1
        return old_base, float(max(new_base_lrs))

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if self._det_loss_sum is None or self._det_count is None:
            return
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self._det_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self._det_count, op=dist.ReduceOp.SUM)
        count = self._det_count.item()
        if count <= 0:
            return
        det_val_mse = self._det_loss_sum.item() / count

        improved = det_val_mse < (self.best_loss - self.min_delta)
        action_msg = "baseline"

        if iteration == 0 or self.best_iter < 0:
            self.best_loss = det_val_mse
            self.best_iter = iteration
            self.lr_bad_evals = 0
            self.es_bad_evals = 0
            action_msg = "init_best"
        elif iteration < self.warmup_skip_iters:
            if improved:
                self.best_loss = det_val_mse
                self.best_iter = iteration
                action_msg = "warmup_new_best"
            else:
                action_msg = "warmup_skip"
        else:
            if improved:
                self.best_loss = det_val_mse
                self.best_iter = iteration
                self.lr_bad_evals = 0
                self.es_bad_evals = 0
                action_msg = "new_best"
            else:
                base_lr = self._get_base_lr()
                at_min_lr = base_lr <= self.min_lr * 1.01
                if self.cooldown_counter > 0:
                    self.cooldown_counter -= 1
                    action_msg = f"cooldown({self.cooldown_counter}_left)"
                elif not at_min_lr:
                    self.lr_bad_evals += 1
                    if self.lr_bad_evals >= self.lr_patience:
                        old_lr, new_lr = self._decay_lr()
                        self.lr_bad_evals = 0
                        self.es_bad_evals = 0
                        self.cooldown_counter = self.cooldown
                        action_msg = f"LR_DECAY({old_lr:.2e}->{new_lr:.2e})"
                    else:
                        action_msg = f"plateau_lr({self.lr_bad_evals}/{self.lr_patience})"
                else:
                    self.es_bad_evals += 1
                    if self.es_bad_evals >= self.early_stop_patience:
                        self.trainer._should_stop = True
                        action_msg = f"EARLY_STOP({self.es_bad_evals}/{self.early_stop_patience})"
                    else:
                        action_msg = f"plateau_es({self.es_bad_evals}/{self.early_stop_patience})"

        base_lr_now = self._get_base_lr()
        active_lr_now = self._get_active_lr()
        if distributed.is_rank0():
            log.info(
                f"[PlateauLR/ES] iter={iteration:5d} | det_val_mse={det_val_mse:.6f} "
                f"(best={self.best_loss:.6f} @ {self.best_iter}) | "
                f"base_lr={base_lr_now:.2e} active_lr={active_lr_now:.2e} | "
                f"decays={self.num_decays} | status={action_msg}"
            )
            if wandb.run is not None:
                wandb.log(
                    {
                        "val/det_mse": det_val_mse,
                        "val/best_det_mse": self.best_loss,
                        "val/base_lr": base_lr_now,
                        "val/active_lr": active_lr_now,
                        "val/lr_bad_evals": self.lr_bad_evals,
                        "val/es_bad_evals": self.es_bad_evals,
                        "val/num_lr_decays": self.num_decays,
                    },
                    step=iteration,
                )
