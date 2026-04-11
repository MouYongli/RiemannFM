"""KGC fine-tuning Lightning module.

Loads a pretrained RiemannFM checkpoint, adds a task-specific head
(link prediction or relation prediction), and provides the training /
evaluation loops for KGC downstream tasks.

Supports two fine-tuning strategies controlled by config:
  - ``backbone_frozen=true``: freeze backbone + entity_emb, train head only
  - ``backbone_frozen=false``: backbone at ``backbone_lr``, head at ``head_lr``
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import lightning as L
import torch
from torch import Tensor, nn

from riemannfm.optim.riemannian import project_curvatures
from riemannfm.tasks.kgc import RiemannFMKGCLinkHead, RiemannFMKGCRelHead

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold
from riemannfm.tasks.metrics import (
    build_true_triples_index,
    compute_ranking_metrics,
    compute_relation_metrics,
    filtered_rank,
)

logger = logging.getLogger(__name__)


class RiemannFMKGCModule(L.LightningModule):
    """KGC fine-tuning module for link and relation prediction.

    Args:
        manifold: Product manifold M = H x S x E.
        entity_emb: Pretrained entity embedding table.
        task_head: Task-specific prediction head (LP or RP).
        task: Task name (``"kgc_lp"`` or ``"kgc_rp"``).
        num_relations: Number of relation types K.
        backbone_frozen: Whether to freeze backbone + entity_emb.
        backbone_lr: Learning rate for backbone parameters (ignored if frozen).
        head_lr: Learning rate for task head parameters.
        curvature_lr: Learning rate for curvature parameters.
        weight_decay: Weight decay.
        warmup_steps: Linear warmup steps.
        max_steps: Total training steps.
        max_grad_norm: Gradient clipping norm.
        use_riemannian_optim: Use geoopt RiemannianAdam.
        margin: Margin for MarginRankingLoss (LP only).
        train_triples: Training triples for filtered eval, shape ``(M, 3)``.
        val_triples: Validation triples for filtered eval, shape ``(M, 3)``.
        test_triples: Test triples for filtered eval, shape ``(M, 3)``.
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        entity_emb: nn.Embedding,
        task_head: RiemannFMKGCLinkHead | RiemannFMKGCRelHead,
        task: str = "kgc_lp",
        num_relations: int = 0,
        backbone_frozen: bool = False,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-3,
        curvature_lr: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 10000,
        max_grad_norm: float = 1.0,
        use_riemannian_optim: bool = True,
        margin: float = 1.0,
        train_triples: Tensor | None = None,
        val_triples: Tensor | None = None,
        test_triples: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "manifold", "entity_emb", "task_head",
                "train_triples", "val_triples", "test_triples",
            ],
        )

        self.manifold = manifold
        self.entity_emb = entity_emb
        self.task_head = task_head
        self.task = task

        # Freeze backbone if requested.
        if backbone_frozen:
            self.entity_emb.requires_grad_(False)
            self.manifold.requires_grad_(False)
            logger.info("Backbone frozen: entity_emb + manifold parameters locked")

        # Loss functions.
        if task == "kgc_lp":
            self.lp_loss = nn.MarginRankingLoss(margin=margin)
        elif task == "kgc_rp":
            self.rp_loss = nn.CrossEntropyLoss()

        # Build filtered ranking index from all known triples.
        all_triples = [t for t in [train_triples, val_triples, test_triples] if t is not None]
        if all_triples:
            self._hr2t, self._rt2h = build_true_triples_index(*all_triples)
        else:
            self._hr2t, self._rt2h = {}, {}

        # Accumulate ranks during validation/test.
        self._eval_ranks: list[int] = []
        self._eval_rel_logits: list[Tensor] = []
        self._eval_rel_targets: list[Tensor] = []

    def _get_entity_coords(self, ids: Tensor) -> Tensor:
        """Look up entity embeddings and project onto manifold.

        Args:
            ids: Entity IDs, shape ``(...)``.

        Returns:
            Manifold coordinates, shape ``(..., D)``.
        """
        x_raw = self.entity_emb(ids)
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            return self.manifold.proj_manifold(x_raw.float())

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int,
    ) -> Tensor:
        """Training step with negative sampling.

        For LP: MarginRankingLoss between positive and negative scores.
        For RP: CrossEntropyLoss on relation classification.
        """
        if self.task == "kgc_lp":
            return self._train_step_lp(batch)
        return self._train_step_rp(batch)

    def _train_step_lp(self, batch: dict[str, Tensor]) -> Tensor:
        """Link prediction training step."""
        # batch shapes: (B, 1+N) where index 0 is positive.
        heads = batch["head"]   # (B, 1+N)
        rels = batch["rel"]     # (B, 1+N)
        tails = batch["tail"]   # (B, 1+N)

        B, S = heads.shape

        # Flatten for embedding lookup.
        h_coords = self._get_entity_coords(heads.reshape(-1))  # (B*S, D)
        t_coords = self._get_entity_coords(tails.reshape(-1))  # (B*S, D)
        r_flat = rels.reshape(-1)  # (B*S,)

        assert isinstance(self.task_head, RiemannFMKGCLinkHead)
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            scores = self.task_head(h_coords.float(), r_flat, t_coords.float())  # (B*S,)

        scores = scores.reshape(B, S)  # (B, 1+N)

        # Positive scores vs negative scores.
        pos_scores = scores[:, 0]  # (B,)
        neg_scores = scores[:, 1:]  # (B, N)

        # MarginRankingLoss: pos_score should be higher than each neg_score.
        pos_expanded = pos_scores.unsqueeze(1).expand_as(neg_scores)  # (B, N)
        target = torch.ones_like(neg_scores)
        loss: Tensor = self.lp_loss(
            pos_expanded.reshape(-1),
            neg_scores.reshape(-1),
            target.reshape(-1),
        )

        self.log("train/loss", loss, prog_bar=True, sync_dist=True, batch_size=B)
        return loss

    def _train_step_rp(self, batch: dict[str, Tensor]) -> Tensor:
        """Relation prediction training step."""
        heads = batch["head"][:, 0]   # (B,) — positive only
        rels = batch["rel"][:, 0]     # (B,)
        tails = batch["tail"][:, 0]   # (B,)

        h_coords = self._get_entity_coords(heads)
        t_coords = self._get_entity_coords(tails)

        assert isinstance(self.task_head, RiemannFMKGCRelHead)
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            logits = self.task_head(h_coords.float(), t_coords.float())  # (B, K)

        loss: Tensor = self.rp_loss(logits, rels)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, batch_size=len(heads))
        return loss

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int,
    ) -> None:
        """Validation step: compute ranks for filtered evaluation."""
        self._eval_step(batch, prefix="val")

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int,
    ) -> None:
        """Test step: compute ranks for filtered evaluation."""
        self._eval_step(batch, prefix="test")

    @torch.no_grad()
    def _eval_step(
        self, batch: dict[str, Tensor], prefix: str,
    ) -> None:
        """Shared evaluation step for val/test."""
        heads = batch["head"]  # (B,)
        rels = batch["rel"]    # (B,)
        tails = batch["tail"]  # (B,)

        if self.task == "kgc_rp":
            h_coords = self._get_entity_coords(heads)
            t_coords = self._get_entity_coords(tails)
            assert isinstance(self.task_head, RiemannFMKGCRelHead)
            with torch.amp.autocast(device_type=self.device.type, enabled=False):
                logits = self.task_head(h_coords.float(), t_coords.float())
            self._eval_rel_logits.append(logits.cpu())
            self._eval_rel_targets.append(rels.cpu())
            return

        # LP: filtered ranking over all entities.
        assert isinstance(self.task_head, RiemannFMKGCLinkHead)

        # Get all entity embeddings.
        all_ids = torch.arange(self.entity_emb.num_embeddings, device=self.device)
        all_coords = self._get_entity_coords(all_ids)  # (E, D)

        for i in range(len(heads)):
            h_i = int(heads[i].item())
            r_i = int(rels[i].item())
            t_i = int(tails[i].item())

            h_coord = self._get_entity_coords(heads[i:i+1])  # (1, D)
            t_coord = self._get_entity_coords(tails[i:i+1])  # (1, D)

            with torch.amp.autocast(device_type=self.device.type, enabled=False):
                # Tail prediction: (h, r, ?) — rank all entities as tail.
                tail_scores = self.task_head.score_all_tails(
                    h_coord.float(), rels[i:i+1], all_coords.float(),
                ).squeeze(0)  # (E,)
                true_tails = self._hr2t.get((h_i, r_i), torch.tensor([], dtype=torch.long))
                tail_rank = filtered_rank(tail_scores, t_i, true_tails.to(self.device))

                # Head prediction: (?, r, t) — rank all entities as head.
                head_scores = self.task_head.score_all_heads(
                    t_coord.float(), rels[i:i+1], all_coords.float(),
                ).squeeze(0)  # (E,)
                true_heads = self._rt2h.get((r_i, t_i), torch.tensor([], dtype=torch.long))
                head_rank = filtered_rank(head_scores, h_i, true_heads.to(self.device))

            self._eval_ranks.extend([tail_rank, head_rank])

    def on_validation_epoch_end(self) -> None:
        self._aggregate_eval_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._aggregate_eval_metrics("test")

    def _aggregate_eval_metrics(self, prefix: str) -> None:
        """Aggregate and log evaluation metrics."""
        if self.task == "kgc_rp" and self._eval_rel_logits:
            all_logits = torch.cat(self._eval_rel_logits)
            all_targets = torch.cat(self._eval_rel_targets)
            metrics = compute_relation_metrics(all_logits, all_targets)
            for key, value in metrics.items():
                self.log(f"{prefix}/{key}", value, prog_bar=(key == "accuracy"), sync_dist=True)
            self._eval_rel_logits.clear()
            self._eval_rel_targets.clear()
            return

        if self._eval_ranks:
            ranks = torch.tensor(self._eval_ranks, dtype=torch.long)
            metrics = compute_ranking_metrics(ranks)
            for key, value in metrics.items():
                self.log(f"{prefix}/{key}", value, prog_bar=(key == "mrr"), sync_dist=True)
            logger.info(
                "%s metrics: MRR=%.4f, Hits@1=%.4f, Hits@3=%.4f, Hits@10=%.4f",
                prefix, metrics["mrr"], metrics["hits@1"],
                metrics["hits@3"], metrics["hits@10"],
            )
            self._eval_ranks.clear()

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Sanitize gradients and project curvatures."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if not self.hparams["backbone_frozen"]:
            project_curvatures(self.manifold)

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        """Build optimizer with separate param groups for backbone and head."""
        head_lr: float = self.hparams["head_lr"]
        backbone_lr: float = self.hparams["backbone_lr"]
        curvature_lr: float = self.hparams["curvature_lr"]
        weight_decay: float = self.hparams["weight_decay"]
        warmup: int = self.hparams["warmup_steps"]
        total_steps: int = self.hparams["max_steps"]
        use_riemannian: bool = self.hparams["use_riemannian_optim"]

        # Head parameters always trained.
        head_params = list(self.task_head.parameters())

        param_groups = [
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        ]

        # Backbone parameters if not frozen.
        if not self.hparams["backbone_frozen"]:
            # Curvature params.
            curvature_ids = set()
            curv_params = []
            for name, p in self.manifold.named_parameters():
                if p.requires_grad and ("curvature" in name or "kappa" in name):
                    curvature_ids.add(id(p))
                    curv_params.append(p)

            # Entity emb + manifold (non-curvature).
            backbone_params = []
            for p in self.entity_emb.parameters():
                if p.requires_grad:
                    backbone_params.append(p)
            for p in self.manifold.parameters():
                if p.requires_grad and id(p) not in curvature_ids:
                    backbone_params.append(p)

            if backbone_params:
                param_groups.append(
                    {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
                )
            if curv_params:
                param_groups.append(
                    {"params": curv_params, "lr": curvature_lr, "weight_decay": 0.0},
                )

        if use_riemannian:
            import geoopt
            optimizer: torch.optim.Optimizer = geoopt.optim.RiemannianAdam(param_groups)
        else:
            optimizer = torch.optim.AdamW(param_groups)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    @classmethod
    def from_pretrained(
        cls,
        ckpt_path: str,
        task: str = "kgc_lp",
        num_relations: int = 0,
        scoring: str = "manifold_dist",
        hidden_dim: int = 512,
        train_triples: Tensor | None = None,
        val_triples: Tensor | None = None,
        test_triples: Tensor | None = None,
        **kwargs: Any,
    ) -> RiemannFMKGCModule:
        """Construct from a pretrained checkpoint.

        Args:
            ckpt_path: Path to pretrained RiemannFMPretrainModule checkpoint.
            task: ``"kgc_lp"`` or ``"kgc_rp"``.
            num_relations: Number of relation types K.
            scoring: LP scoring mode (``"manifold_dist"`` or ``"bilinear"``).
            hidden_dim: RP head hidden dimension.
            train_triples: For building filtered eval index.
            val_triples: For building filtered eval index.
            test_triples: For building filtered eval index.
            **kwargs: Passed to constructor (backbone_frozen, head_lr, etc.).
        """
        from riemannfm.models.lightning_module import RiemannFMPretrainModule

        logger.info("Loading pretrained checkpoint: %s", ckpt_path)
        pretrained = RiemannFMPretrainModule.load_from_checkpoint(
            ckpt_path, map_location="cpu",
        )

        manifold = pretrained.manifold
        entity_emb = pretrained.entity_emb
        ambient_dim = manifold.ambient_dim

        # Build task head.
        if task == "kgc_lp":
            task_head: RiemannFMKGCLinkHead | RiemannFMKGCRelHead = RiemannFMKGCLinkHead(
                manifold=manifold,
                ambient_dim=ambient_dim,
                num_relations=num_relations,
                scoring=scoring,
            )
        elif task == "kgc_rp":
            task_head = RiemannFMKGCRelHead(
                ambient_dim=ambient_dim,
                num_relations=num_relations,
                hidden_dim=hidden_dim,
            )
        else:
            msg = f"Unknown task: {task}"
            raise ValueError(msg)

        return cls(
            manifold=manifold,
            entity_emb=entity_emb,
            task_head=task_head,
            task=task,
            num_relations=num_relations,
            train_triples=train_triples,
            val_triples=val_triples,
            test_triples=test_triples,
            **kwargs,
        )
