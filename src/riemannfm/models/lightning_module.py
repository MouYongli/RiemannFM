"""RiemannFM Lightning training module.

Wires together: manifold -> flow -> model -> loss -> optimizer
into a single LightningModule for pretraining.
"""

from __future__ import annotations

import math
from typing import Any

import lightning as L
import torch
from torch import Tensor, nn

from riemannfm.flow.joint_flow import RiemannFMJointFlow
from riemannfm.losses.combined_loss import RiemannFMCombinedLoss
from riemannfm.manifolds.product import RiemannFMProductManifold
from riemannfm.models.riemannfm import RiemannFM
from riemannfm.optim.riemannian import build_optimizer, project_curvatures

# Fixed t values for multi-t validation averaging.
_VAL_T_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]


class RiemannFMPretrainModule(L.LightningModule):
    """Lightning module for RiemannFM pretraining (Algorithm 1).

    Owns:
      - Product manifold with learnable curvatures
      - Entity embedding table (nn.Embedding)
      - RiemannFM model (encoder + backbone + heads)
      - Joint flow matching (noise sampling + interpolation)
      - Combined loss function

    Args:
        manifold: Product manifold M = H x S x E.
        model: RiemannFM model.
        flow: Joint flow matching module.
        loss_fn: Combined loss function.
        num_entities: Total number of entities for learnable embeddings.
        lr: Base learning rate.
        curvature_lr: Curvature parameter learning rate.
        weight_decay: Weight decay.
        warmup_steps: Linear warmup steps.
        max_steps: Total training steps (for cosine annealing).
        max_grad_norm: Gradient clipping norm.
        use_riemannian_optim: Use geoopt RiemannianAdam.
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        model: RiemannFM,
        flow: RiemannFMJointFlow,
        loss_fn: RiemannFMCombinedLoss,
        num_entities: int,
        input_text_dim: int,
        C_R: Tensor | None = None,
        lr: float = 1e-4,
        curvature_lr: float = 1e-5,
        align_lr: float | None = None,
        weight_decay: float = 0.01,
        warmup_steps: int = 10000,
        max_steps: int = 500_000,
        max_grad_norm: float = 1.0,
        use_riemannian_optim: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["manifold", "model", "flow", "loss_fn", "C_R"],
        )

        self.manifold = manifold
        self.model = torch.compile(model)
        self.flow = flow
        self.loss_fn = loss_fn

        # Global relation text embeddings C_R (Def 3.6), shared across batches.
        # Registered as buffer so it moves with the model to GPU.
        if C_R is not None:
            self.register_buffer("C_R", C_R)
        else:
            self.C_R = None

        # Learnable entity embeddings -> manifold coordinates.
        # Initialize near the manifold origin with small noise.
        D = manifold.ambient_dim
        self.entity_emb = nn.Embedding(num_entities, D)
        nn.init.normal_(self.entity_emb.weight, std=0.01)
        # Set the origin-like initial values for hyperbolic/spherical time coords.
        with torch.no_grad():
            origin = manifold.origin(device=self.entity_emb.weight.device)  # (D,)
            self.entity_emb.weight.add_(origin)

        # Learnable [MASK] text token for MASK_C nodes (L_mask_c input).
        # Lives in text-embedding space (d_c), not manifold space:
        # M_c nodes keep their real geometric coordinates (t=1 in flow)
        # and only have their text replaced by this shared vector.
        self.mask_emb = nn.Parameter(torch.zeros(max(input_text_dim, 1)))
        if input_text_dim > 0:
            nn.init.normal_(self.mask_emb, std=0.02)

    def _get_manifold_coords(
        self, node_ids: Tensor, node_mask: Tensor,
    ) -> Tensor:
        """Look up entity embeddings and project onto manifold.

        Args:
            node_ids: Entity IDs, shape ``(B, N)``. -1 for virtual nodes.
            node_mask: Bool mask, shape ``(B, N)``.

        Returns:
            Manifold coordinates x_1, shape ``(B, N, D)``.
        """
        # Clamp -1 to 0 for embedding lookup, then mask later.
        safe_ids = node_ids.clamp(min=0)
        x_raw = self.entity_emb(safe_ids)  # (B, N, D)

        # Force float32 for manifold projection (numerical stability under AMP).
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            x_1 = self.manifold.proj_manifold(x_raw.float())

        # Zero out virtual node coordinates (use origin).
        origin = self.manifold.origin(device=x_1.device, dtype=x_1.dtype)
        mask = node_mask.unsqueeze(-1)  # (B, N, 1)
        x_1 = torch.where(mask, x_1, origin)

        return x_1

    def _apply_mask_semantics(
        self, node_text: Tensor, mask_type: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply text-side masking for MASK_C nodes (L_mask_c input).

        The collator already labelled each real node as REAL, MASK_C, or
        MASK_X (disjoint).  Here we only touch the text stream:

          - ``MASK_C`` nodes: c_i ← self.mask_emb (a single learnable
            vector).  The geometry is kept (collator sets ``t_node=1``
            so ``x_t = x_1`` for these nodes).
          - ``MASK_X`` nodes: c_i stays real.  Their geometry is masked
            by the flow module (``t_node=0`` forces ``x_t = x_0``).

        Args:
            node_text: Text embeddings, shape ``(B, N, d_c)``.
            mask_type: Node partition labels, shape ``(B, N)``.

        Returns:
            ``(node_text_masked, true_text_emb)`` where ``true_text_emb``
            is the pre-masking clone used as the L_mask_c target.
        """
        from riemannfm.data.collator import MASK_C

        true_text_emb = node_text.detach().clone()

        is_mc = mask_type == MASK_C
        if not is_mc.any():
            return node_text, true_text_emb

        node_text = node_text.clone()
        mask_vec = self.mask_emb.to(node_text.dtype)
        # Broadcast a single learnable vector over all M_c positions.
        node_text[is_mc] = mask_vec
        return node_text, true_text_emb

    def _shared_step(
        self, batch: dict[str, Any], t_override: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Shared computation for training and validation.

        Args:
            batch: Collated batch dict.
            t_override: If provided, use these fixed time steps instead
                of random sampling.  Shape ``(B,)``.

        Returns:
            (total_loss, metrics_dict).
        """
        E_1 = batch["edge_types"]       # (B, N, N, K)
        node_text = batch["node_text"]   # (B, N, input_text_dim)
        node_mask = batch["node_mask"]   # (B, N)
        node_ids = batch["node_ids"]     # (B, N)
        mask_type = batch.get("mask_type")  # (B, N) or None
        t_node = batch.get("t_node")     # (B, N) float32 or None

        # 1. Get data manifold coordinates from entity embeddings.
        x_1 = self._get_manifold_coords(node_ids, node_mask)

        # 1b. Apply M_c text replacement (collator already labelled nodes).
        #     M_x geometry masking happens inside flow via t_node=0.
        true_text_emb = None
        has_mask_loss = self.loss_fn.nu_mask_c > 0 or self.loss_fn.nu_mask_x > 0
        if mask_type is not None and has_mask_loss:
            node_text, true_text_emb = self._apply_mask_semantics(
                node_text, mask_type,
            )

        # 2. Flow matching: sample noise, time, interpolate.
        #    Force fp32 — geodesic interpolation is numerically sensitive.
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            sample = self.flow.sample(
                x_1.float(), E_1.float(), node_mask,
                t_node_override=t_node,
            )

        # Override t if provided (for multi-t validation).  Respects the
        # per-node mask labels: M_x stays at t=0, M_c stays at t=1.
        if t_override is not None:
            from riemannfm.flow.continuous_flow import (
                geodesic_interpolation,
                vector_field_target,
            )
            from riemannfm.flow.discrete_flow import discrete_interpolation
            from riemannfm.flow.noise import (
                sample_continuous_noise,
                sample_discrete_noise,
            )

            x_0 = sample_continuous_noise(self.manifold, *x_1.shape[:-1],
                                          device=x_1.device, dtype=x_1.dtype)
            E_0 = sample_discrete_noise(
                E_1, avg_edge_density=self.flow.avg_edge_density,
                rho_k=self.flow.rho_k,
            )

            # Expand scalar t_override to (B, N) and overlay mask labels.
            if t_node is not None:
                fill = t_override.unsqueeze(-1).expand_as(t_node)
                t_cont = torch.where(torch.isnan(t_node), fill, t_node.to(fill.dtype))
            else:
                t_cont = t_override

            with torch.amp.autocast(device_type=self.device.type, enabled=False):
                sample.x_t = geodesic_interpolation(
                    self.manifold, x_0.float(), x_1.float(), t_cont,
                ).to(x_1.dtype)
                sample.u_t = vector_field_target(
                    self.manifold, sample.x_t.float(), x_1.float(),
                    t_cont, t_max=self.flow.t_max,
                ).to(x_1.dtype)

            sample.E_t = discrete_interpolation(E_0, E_1, t_override)
            sample.t = t_cont

        # 3. Forward pass through model.
        V_hat, P_hat, h = self.model(
            x_t=sample.x_t,
            E_t=sample.E_t,
            t=sample.t,
            node_text=node_text,
            node_mask=node_mask,
            C_R=self.C_R,
            node_pe=batch.get("node_pe"),
        )

        # 4. Compute loss (fp32 for manifold norms; cast only what needs it).
        #    V_hat, u_t, x_t need fp32 for Riemannian tangent norm in L_cont.
        #    P_hat, E_1 stay in native dtype — BCE is numerically stable.
        _f32 = torch.float32
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            total_loss, metrics = self.loss_fn(
                V_hat=V_hat if V_hat.dtype == _f32 else V_hat.float(),
                u_t=sample.u_t if sample.u_t.dtype == _f32 else sample.u_t.float(),
                x_t=sample.x_t if sample.x_t.dtype == _f32 else sample.x_t.float(),
                P_hat=P_hat,
                E_1=sample.E_1,
                node_mask=node_mask,
                h=h if h.dtype == _f32 else h.float(),
                node_text=node_text,
                mask_type=mask_type,
                true_text_emb=true_text_emb,
            )

        return total_loss, metrics

    def training_step(
        self, batch: dict[str, Any], batch_idx: int,
    ) -> Tensor:
        """Single training step (Algorithm 1).

        Args:
            batch: Dict from collator.
            batch_idx: Batch index.

        Returns:
            Scalar loss.
        """
        total_loss, metrics = self._shared_step(batch)
        B = batch["edge_types"].shape[0]

        # Log with train/ prefix. on_step=True ensures CSVLogger records per step.
        self.log("train/loss", metrics["loss/total"], prog_bar=True, sync_dist=True, batch_size=B, on_step=True, on_epoch=False)
        self.log("train/L_cont", metrics["loss/cont"], sync_dist=True, batch_size=B, on_step=True, on_epoch=False)
        self.log("train/L_disc", metrics["loss/disc"], sync_dist=True, batch_size=B, on_step=True, on_epoch=False)
        self.log("train/L_align", metrics["loss/align"], sync_dist=True, batch_size=B, on_step=True, on_epoch=False)
        self.log("train/L_mask_c", metrics["loss/mask_c"], sync_dist=True, batch_size=B, on_step=True, on_epoch=False)
        self.log("train/L_mask_x", metrics["loss/mask_x"], sync_dist=True, batch_size=B, on_step=True, on_epoch=False)

        # Log curvatures.
        if self.manifold.hyperbolic is not None:
            self.log("curvature/kappa_h", self.manifold.hyperbolic.curvature.detach(), batch_size=B, on_step=True, on_epoch=False)
        if self.manifold.spherical is not None:
            self.log("curvature/kappa_s", self.manifold.spherical.curvature.detach(), batch_size=B, on_step=True, on_epoch=False)

        # Log learning rates on the same CSV row as training metrics.
        # LearningRateMonitor logs on separate rows, making CSV analysis hard.
        opts = self.optimizers()
        if opts is not None:
            opt = opts if isinstance(opts, torch.optim.Optimizer) else opts[0]  # type: ignore[index]
            for i, pg in enumerate(opt.param_groups):
                self.log(f"lr/pg{i}", pg["lr"], batch_size=B, on_step=True, on_epoch=False)

        return total_loss

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int,
    ) -> Tensor:
        """Validation step with multi-t averaging for stable metrics.

        Evaluates the model at multiple fixed time steps and averages
        the loss, providing more stable validation signals than a
        single random t sample.

        Args:
            batch: Dict from collator.
            batch_idx: Batch index.

        Returns:
            Average validation loss.
        """
        B = batch["edge_types"].shape[0]
        total_loss = torch.tensor(0.0, device=self.device)
        total_cont = torch.tensor(0.0, device=self.device)
        total_disc = torch.tensor(0.0, device=self.device)
        total_align = torch.tensor(0.0, device=self.device)
        total_mask_c = torch.tensor(0.0, device=self.device)
        total_mask_x = torch.tensor(0.0, device=self.device)

        for t_val in _VAL_T_VALUES:
            t_fixed = torch.full((B,), t_val, device=self.device)
            loss, metrics = self._shared_step(batch, t_override=t_fixed)
            total_loss = total_loss + loss
            total_cont = total_cont + metrics["loss/cont"]
            total_disc = total_disc + metrics["loss/disc"]
            total_align = total_align + metrics["loss/align"]
            total_mask_c = total_mask_c + metrics["loss/mask_c"]
            total_mask_x = total_mask_x + metrics["loss/mask_x"]

        n = len(_VAL_T_VALUES)
        avg_loss = total_loss / n

        self.log("val/loss", avg_loss, prog_bar=True, sync_dist=True, batch_size=B)
        self.log("val/L_cont", total_cont / n, sync_dist=True, batch_size=B)
        self.log("val/L_disc", total_disc / n, sync_dist=True, batch_size=B)
        self.log("val/L_align", total_align / n, sync_dist=True, batch_size=B)
        self.log("val/L_mask_c", total_mask_c / n, sync_dist=True, batch_size=B)
        self.log("val/L_mask_x", total_mask_x / n, sync_dist=True, batch_size=B)

        return avg_loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Sanitize gradients and project curvatures before optimizer step.

        Runs *before* Lightning's automatic ``gradient_clip_val`` clipping.
        ``clip_grad_norm_`` computes a single global norm across all
        parameters; if any parameter has a ``NaN``/``Inf`` gradient (e.g.
        from a degenerate manifold op), the global ``clip_coef`` becomes
        ``NaN`` and re-poisons every other parameter — including curvature,
        which has its own per-parameter sanitizing hook.  Replacing NaN/Inf
        with zero here keeps a single bad token from corrupting the entire
        step.

        Also re-projects curvatures into valid ranges (Algo 1, step 11):
          - kappa_h <= -eps (hyperbolic must be negative)
          - kappa_s >= +eps (spherical must be positive)
        """
        for p in self.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        project_curvatures(self.manifold)

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | int | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Per-group gradient clipping to protect alignment projections.

        Global clipping starves proj_g/proj_c (0.15% of params) when
        backbone gradients from L_cont dominate the global norm.
        Clip each group independently so alignment projections get
        their own gradient norm budget.
        """
        if not gradient_clip_val:
            return

        clip_val = float(gradient_clip_val)

        # Group 1: alignment projection parameters (proj_g, proj_c).
        align_params = [p for p in self.loss_fn.parameters() if p.grad is not None]
        if align_params:
            torch.nn.utils.clip_grad_norm_(align_params, clip_val)

        # Group 2: everything else (backbone, entity_emb, curvature).
        align_ids = {id(p) for p in self.loss_fn.parameters()}
        other_params = [
            p for p in self.parameters()
            if p.grad is not None and id(p) not in align_ids
        ]
        if other_params:
            torch.nn.utils.clip_grad_norm_(other_params, clip_val)

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        """Build optimizer with linear warmup + cosine annealing.

        Param groups pg0 (backbone) and pg1 (curvature) use linear
        warmup followed by cosine annealing.  pg2 (alignment
        projections) skips warmup and starts at full ``align_lr``
        with cosine-only decay, so L_align can learn from step 0.
        """
        lr: float = self.hparams["lr"]
        curvature_lr: float = self.hparams["curvature_lr"]
        align_lr: float | None = self.hparams.get("align_lr")
        weight_decay: float = self.hparams["weight_decay"]
        use_riemannian_optim: bool = self.hparams["use_riemannian_optim"]
        warmup: int = self.hparams["warmup_steps"]
        total_steps: int = self.hparams["max_steps"]

        optimizer = build_optimizer(
            model=self,
            manifold=self.manifold,
            lr=lr,
            curvature_lr=curvature_lr,
            align_lr=align_lr,
            weight_decay=weight_decay,
            use_riemannian_optim=use_riemannian_optim,
        )

        def warmup_cosine(step: int) -> float:
            """Linear warmup + cosine annealing (pg0, pg1)."""
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        def cosine_only(step: int) -> float:
            """Cosine annealing without warmup (pg2 — alignment)."""
            progress = step / max(total_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # Per-group lambdas: pg0, pg1 get warmup; pg2 skips warmup.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, [warmup_cosine, warmup_cosine, cosine_only],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @classmethod
    def from_config(
        cls,
        model_cfg: Any,
        manifold_cfg: Any,
        flow_cfg: Any,
        training_cfg: Any,
        ablation_cfg: Any,
        num_edge_types: int,
        num_entities: int,
        input_text_dim: int = 0,
        C_R: Tensor | None = None,
        max_steps: int = 500_000,
    ) -> RiemannFMPretrainModule:
        """Construct from Hydra configs.

        Args:
            input_text_dim: Actual text embedding dimension from disk
                (auto-detected by DataModule after setup).
        """
        manifold = RiemannFMProductManifold.from_config(manifold_cfg)

        model = RiemannFM(
            manifold=manifold,
            num_layers=model_cfg.num_layers,
            node_dim=model_cfg.node_dim,
            edge_dim=model_cfg.edge_dim,
            num_heads=model_cfg.num_heads,
            edge_heads=model_cfg.edge_heads,
            num_edge_types=num_edge_types,
            input_text_dim=input_text_dim,
            text_proj_dim=int(getattr(model_cfg, "text_proj_dim", 256)),
            use_text_cross_attn=bool(getattr(model_cfg, "use_text_cross_attn", False)),
            text_cross_attn_every=model_cfg.text_cross_attn_every,
            rel_emb_dim=model_cfg.rel_emb_dim,
            use_geodesic_kernel=ablation_cfg.use_geodesic_kernel,
            use_ath_norm=ablation_cfg.use_ath_norm,
            use_edge_self_update=ablation_cfg.use_edge_self_update,
            use_dual_stream_cross=ablation_cfg.use_dual_stream_cross,
            use_text_condition=ablation_cfg.use_text_condition,
            dropout=model_cfg.dropout,
        )

        flow = RiemannFMJointFlow.from_config(flow_cfg, manifold)

        loss_fn = RiemannFMCombinedLoss(
            manifold=manifold,
            lambda_disc=training_cfg.lambda_disc,
            mu_align=training_cfg.mu_align,
            nu_mask_c=float(getattr(training_cfg, "nu_mask_c", 0.0)),
            nu_mask_x=float(getattr(training_cfg, "nu_mask_x", 0.0)),
            neg_ratio=float(getattr(training_cfg, "neg_ratio", 1.0)),
            temperature=training_cfg.temperature,
            mask_c_temperature=float(
                getattr(training_cfg, "mask_c_temperature", 0.07),
            ),
            input_text_dim=input_text_dim,
            node_dim=model_cfg.node_dim,
            d_a=int(getattr(model_cfg, "text_proj_dim", 256)),
            max_align_nodes=int(getattr(training_cfg, "max_align_nodes", 128)),
        )

        return cls(
            manifold=manifold,
            model=model,
            flow=flow,
            loss_fn=loss_fn,
            num_entities=num_entities,
            input_text_dim=input_text_dim,
            C_R=C_R,
            lr=training_cfg.lr,
            curvature_lr=training_cfg.curvature_lr,
            align_lr=float(getattr(training_cfg, "align_lr", 0))
            or None,
            weight_decay=training_cfg.weight_decay,
            warmup_steps=training_cfg.warmup_steps,
            max_steps=max_steps,
            max_grad_norm=training_cfg.max_grad_norm,
            use_riemannian_optim=training_cfg.use_riemannian_optim,
        )
