"""RiemannFM Lightning training module (spec §22, Stage 1 of v1.0.1).

Wires: manifold → flow → model → loss → optimizer.

Stage 1 scope — modality masking (per-node m_text / m_coord bits and
four-mode t distribution) is introduced in Stage 2. For now the loss
treats every real node as an unmasked anchor.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import lightning as L
import torch
from torch import Tensor

from riemannfm.optim.riemannian import build_optimizer, project_curvatures

if TYPE_CHECKING:
    from riemannfm.flow.joint_flow import RiemannFMJointFlow
    from riemannfm.losses.combined_loss import RiemannFMCombinedLoss
    from riemannfm.manifolds.product import RiemannFMProductManifold
    from riemannfm.models.pretrain_heads import RiemannFMPretrainHeads
    from riemannfm.models.riemannfm import RiemannFM

_VAL_T_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]


class RiemannFMPretrainModule(L.LightningModule):
    """Lightning module for RiemannFM pretraining (spec §22).

    Args:
        manifold: Product manifold M = H × S × E.
        model: RiemannFM model.
        flow: Joint flow matching module.
        loss_fn: Combined loss (parameter-free orchestrator).
        pretrain_heads: Pretrain-only parameters (entity_emb, mask_emb,
            optional W_p / W_p_c for L_align^R).
        C_R: Global relation text embeddings, shape ``(K, d_c)``.
        lr: Base learning rate (main group).
        curvature_lr: Curvature-group learning rate.
        relation_lr: Relation-group learning rate (``rel_emb`` +
            align projections). Defaults to ``lr / 3`` (spec §20.2).
        weight_decay: Weight decay for the main group.
        warmup_steps: Linear warmup steps.
        max_steps: Total training steps (for cosine decay).
        max_grad_norm: Gradient clipping norm.
        use_riemannian_optim: Use ``geoopt.RiemannianAdam``.
        lr_min_ratio: Cosine decay floor as a fraction of peak LR.
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        model: RiemannFM,
        flow: RiemannFMJointFlow,
        loss_fn: RiemannFMCombinedLoss,
        pretrain_heads: RiemannFMPretrainHeads,
        C_R: Tensor | None = None,
        lr: float = 1e-4,
        curvature_lr: float = 1e-5,
        relation_lr: float | None = None,
        weight_decay: float = 0.01,
        warmup_steps: int = 10000,
        max_steps: int = 500_000,
        max_grad_norm: float = 1.0,
        use_riemannian_optim: bool = True,
        lr_min_ratio: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["manifold", "model", "flow", "loss_fn", "pretrain_heads", "C_R"],
        )

        self.manifold = manifold
        self.model = torch.compile(model)
        self.flow = flow
        self.loss_fn = loss_fn
        self.pretrain_heads = pretrain_heads

        if C_R is not None:
            self.register_buffer("C_R", C_R)
        else:
            self.C_R = None

    def _get_manifold_coords(
        self, node_ids: Tensor, node_mask: Tensor,
    ) -> Tensor:
        """Look up entity embeddings and project onto the manifold."""
        safe_ids = node_ids.clamp(min=0)
        x_raw = self.pretrain_heads.entity_emb(safe_ids)

        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            x_1 = self.manifold.proj_manifold(x_raw.float())

        origin = self.manifold.origin(device=x_1.device, dtype=x_1.dtype)
        mask = node_mask.unsqueeze(-1)
        x_1 = torch.where(mask, x_1, origin)

        return x_1

    def _compute_relation_align_pairs(
        self,
    ) -> tuple[Tensor, Tensor] | tuple[None, None]:
        """Pre-project R and C_R for L_align^R (spec def 19.1)."""
        if (
            self.loss_fn.lambda_align_R <= 0
            or self.pretrain_heads.W_p is None
            or self.C_R is None
        ):
            return None, None
        R = self._raw_model().rel_emb
        z_R, z_C = self.pretrain_heads.project_relation_align(R, self.C_R)
        return z_R, z_C

    def _raw_model(self) -> RiemannFM:
        """Return the unwrapped RiemannFM (torch.compile wraps it)."""
        compiled = self.model
        inner = getattr(compiled, "_orig_mod", None)
        return inner if inner is not None else compiled  # type: ignore[return-value]

    def _apply_text_mask(
        self, node_text: Tensor, m_text: Tensor | None,
    ) -> Tensor:
        """Replace text with ``mask_emb`` at positions with m_text = 0."""
        if m_text is None or node_text.shape[-1] == 0:
            return node_text
        mask_vec = self.pretrain_heads.mask_emb.to(node_text.dtype)
        keep = m_text.unsqueeze(-1).to(node_text.dtype)
        return keep * node_text + (1.0 - keep) * mask_vec

    def _shared_step(
        self, batch: dict[str, Any], t_override: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        E_1 = batch["edge_types"]        # (B, N, N, K)
        node_text = batch["node_text"]   # (B, N, input_text_dim)
        node_mask = batch["node_mask"]   # (B, N)
        node_ids = batch["node_ids"]     # (B, N)
        m_text = batch.get("m_text")     # (B, N) bool or None
        m_coord = batch.get("m_coord")   # (B, N) bool or None
        mode_idx = batch.get("mode_idx") # (B,) long or None

        x_1 = self._get_manifold_coords(node_ids, node_mask)

        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            sample = self.flow.sample(
                x_1.float(), E_1.float(), node_mask,
                m_coord=m_coord, mode_idx=mode_idx,
            )

        # Validation override of the batch-level t (keeps the edge mask
        # consistent with cosine α(t)). Validation runs with modality
        # masking disabled by the datamodule, so m_coord/m_text are the
        # collator's all-ones defaults.
        if t_override is not None:
            from riemannfm.flow.continuous_flow import (
                geodesic_interpolation,
                vector_field_target,
            )
            from riemannfm.flow.discrete_flow import discrete_interpolation
            from riemannfm.flow.noise import sample_continuous_noise

            x_0 = sample_continuous_noise(
                self.manifold, *x_1.shape[:-1],
                device=x_1.device, dtype=x_1.dtype,
            )

            with torch.amp.autocast(device_type=self.device.type, enabled=False):
                sample.x_t = geodesic_interpolation(
                    self.manifold, x_0.float(), x_1.float(), t_override,
                ).to(x_1.dtype)
                sample.u_t = vector_field_target(
                    self.manifold, sample.x_t.float(), x_1.float(),
                    t_override, t_max=self.flow.t_max,
                ).to(x_1.dtype)

            sample.E_t, sample.mu_t = discrete_interpolation(
                E_1, t_override, schedule=self.flow.edge_schedule,
            )
            sample.t = t_override

        node_text_masked = self._apply_text_mask(node_text, m_text)

        V_hat, ell_ex, ell_type, _h = self.model(
            x_t=sample.x_t,
            E_t=sample.E_t,
            mu_t=sample.mu_t,
            t=sample.t,
            node_text=node_text_masked,
            node_mask=node_mask,
            C_R=self.C_R,
            node_pe=batch.get("node_pe"),
            m_text=m_text,
            m_coord=m_coord,
        )

        z_R, z_C = self._compute_relation_align_pairs()

        _f32 = torch.float32
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            total_loss, metrics = self.loss_fn(
                V_hat=V_hat if V_hat.dtype == _f32 else V_hat.float(),
                u_t=sample.u_t if sample.u_t.dtype == _f32 else sample.u_t.float(),
                x_t=sample.x_t if sample.x_t.dtype == _f32 else sample.x_t.float(),
                ell_ex=ell_ex,
                ell_type=ell_type,
                E_1=sample.E_1,
                mu_t=sample.mu_t,
                node_mask=node_mask,
                m_coord=m_coord,
                z_R=z_R,
                z_C=z_C,
            )

        return total_loss, metrics

    def training_step(
        self, batch: dict[str, Any], batch_idx: int,
    ) -> Tensor:
        total_loss, metrics = self._shared_step(batch)
        B = batch["edge_types"].shape[0]

        self.log("train/loss", metrics["loss/total"], prog_bar=True, sync_dist=True, batch_size=B, on_step=True, on_epoch=False)
        self.log("train/L_X", metrics["loss/X"], sync_dist=True, batch_size=B, on_step=True, on_epoch=False)
        self.log("train/L_ex", metrics["loss/ex"], sync_dist=True, batch_size=B, on_step=True, on_epoch=False)
        self.log("train/L_ty", metrics["loss/ty"], sync_dist=True, batch_size=B, on_step=True, on_epoch=False)
        self.log("train/L_align_R", metrics["loss/align_R"], sync_dist=True, batch_size=B, on_step=True, on_epoch=False)

        if self.manifold.hyperbolic is not None:
            self.log("curvature/kappa_h", self.manifold.hyperbolic.curvature.detach(), batch_size=B, on_step=True, on_epoch=False)
        if self.manifold.spherical is not None:
            self.log("curvature/kappa_s", self.manifold.spherical.curvature.detach(), batch_size=B, on_step=True, on_epoch=False)

        opts = self.optimizers()
        if opts is not None:
            opt = opts if isinstance(opts, torch.optim.Optimizer) else opts[0]  # type: ignore[index]
            for i, pg in enumerate(opt.param_groups):
                self.log(f"lr/pg{i}", pg["lr"], batch_size=B, on_step=True, on_epoch=False)

        return total_loss

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int,
    ) -> Tensor:
        """Multi-t validation averaging for stable metrics."""
        B = batch["edge_types"].shape[0]
        total_loss = torch.tensor(0.0, device=self.device)
        acc = {
            "loss/X": torch.tensor(0.0, device=self.device),
            "loss/ex": torch.tensor(0.0, device=self.device),
            "loss/ty": torch.tensor(0.0, device=self.device),
            "loss/align_R": torch.tensor(0.0, device=self.device),
        }

        for t_val in _VAL_T_VALUES:
            t_fixed = torch.full((B,), t_val, device=self.device)
            loss, metrics = self._shared_step(batch, t_override=t_fixed)
            total_loss = total_loss + loss
            for k in acc:
                acc[k] = acc[k] + metrics[k]

        n = len(_VAL_T_VALUES)
        avg_loss = total_loss / n

        self.log("val/loss", avg_loss, prog_bar=True, sync_dist=True, batch_size=B)
        self.log("val/L_X", acc["loss/X"] / n, sync_dist=True, batch_size=B)
        self.log("val/L_ex", acc["loss/ex"] / n, sync_dist=True, batch_size=B)
        self.log("val/L_ty", acc["loss/ty"] / n, sync_dist=True, batch_size=B)
        self.log("val/L_align_R", acc["loss/align_R"] / n, sync_dist=True, batch_size=B)

        return avg_loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Sanitise gradients and project curvatures."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        project_curvatures(self.manifold)

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        """Linear warmup + cosine annealing for all three param groups."""
        lr: float = self.hparams["lr"]
        curvature_lr: float = self.hparams["curvature_lr"]
        relation_lr: float | None = self.hparams.get("relation_lr")
        weight_decay: float = self.hparams["weight_decay"]
        use_riemannian_optim: bool = self.hparams["use_riemannian_optim"]
        warmup: int = self.hparams["warmup_steps"]
        total_steps: int = self.hparams["max_steps"]
        min_ratio: float = float(self.hparams.get("lr_min_ratio", 0.0))

        optimizer = build_optimizer(
            model=self,
            manifold=self.manifold,
            lr=lr,
            curvature_lr=curvature_lr,
            relation_lr=relation_lr,
            weight_decay=weight_decay,
            use_riemannian_optim=use_riemannian_optim,
        )

        def warmup_cosine(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            cos_term = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cos_term

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, [warmup_cosine, warmup_cosine, warmup_cosine],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

