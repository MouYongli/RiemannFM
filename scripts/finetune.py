"""RieDFM-G fine-tuning script for downstream tasks.

Usage:
    python scripts/finetune.py data=fb15k237 training=finetune
    python scripts/finetune.py data=wn18rr training=finetune training.lr=1e-5
"""

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../src/riedfm/configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Fine-tuning configuration:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Build manifold
    from riedfm.manifolds.product import ProductManifold

    manifold = ProductManifold(
        dim_hyperbolic=cfg.manifold.dim_hyperbolic,
        dim_spherical=cfg.manifold.dim_spherical,
        dim_euclidean=cfg.manifold.dim_euclidean,
        init_curvature_h=cfg.manifold.init_curvature_h,
        init_curvature_s=cfg.manifold.init_curvature_s,
        learn_curvature=cfg.manifold.learn_curvature,
    ).to(device)

    # Build model and load pretrained backbone
    from riedfm.models.riedfm_g import RieDFMG

    model = RieDFMG(
        manifold=manifold,
        num_layers=cfg.model.num_layers,
        node_dim=cfg.model.node_dim,
        edge_dim=cfg.model.edge_dim,
        num_heads=cfg.model.num_heads,
        num_edge_types=cfg.data.num_edge_types,
        text_dim=cfg.model.text_dim,
        text_cross_attn_every=cfg.model.text_cross_attn_every,
        dropout=cfg.model.dropout,
    ).to(device)

    # Load pretrained weights if specified
    pretrained_path = cfg.get("pretrained_checkpoint", None)
    if pretrained_path:
        from riedfm.utils.checkpoint import load_backbone

        load_backbone(pretrained_path, model, strict=False, device=device)
        logger.info(f"Loaded pretrained backbone from {pretrained_path}")

    # Optionally freeze backbone layers
    freeze_layers = cfg.training.get("freeze_backbone_layers", 0)
    if freeze_layers > 0:
        for i, block in enumerate(model.backbone.blocks[:freeze_layers]):
            for param in block.parameters():
                param.requires_grad = False
        logger.info(f"Froze first {freeze_layers} backbone layers")

    # Build optimizer and loss (same structure as pretrain)
    from riedfm.utils.riemannian_optim import build_optimizer

    optimizer = build_optimizer(
        model,
        lr=cfg.training.lr,
        curvature_lr=cfg.training.curvature_lr,
        weight_decay=cfg.training.weight_decay,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {num_params:,}")

    # TODO: Add dataset-specific fine-tuning loop
    # This follows the same pattern as pretrain.py but with:
    # - Different dataset (FB15k-237, WN18RR, etc.)
    # - Task-specific evaluation metrics
    # - Potentially different loss weighting

    logger.info("Fine-tuning script ready. Implement task-specific training loop.")


if __name__ == "__main__":
    main()
