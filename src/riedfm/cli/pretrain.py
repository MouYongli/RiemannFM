"""RieDFM-G pretraining script.

Usage:
    python -m riedfm.cli.pretrain
    python -m riedfm.cli.pretrain model=red_former_large data=wikidata_5m
    python -m riedfm.cli.pretrain training.batch_size=128 training.lr=2e-4
"""

import logging
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build manifold
    from riedfm.manifolds.product import RieDFMProductManifold

    manifold = RieDFMProductManifold(
        dim_hyperbolic=cfg.manifold.dim_hyperbolic,
        dim_spherical=cfg.manifold.dim_spherical,
        dim_euclidean=cfg.manifold.dim_euclidean,
        init_curvature_h=cfg.manifold.init_curvature_h,
        init_curvature_s=cfg.manifold.init_curvature_s,
        learn_curvature=cfg.manifold.learn_curvature,
    ).to(device)

    # Build model
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
        avg_edge_density=cfg.flow.avg_edge_density,
        dropout=cfg.model.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Build optimizer
    from riedfm.utils.riemannian_optim import build_optimizer

    optimizer = build_optimizer(
        model,
        lr=cfg.training.lr,
        curvature_lr=cfg.training.curvature_lr,
        weight_decay=cfg.training.weight_decay,
        use_riemannian=cfg.training.use_riemannian_optim,
    )

    # Build loss
    from riedfm.losses.combined_loss import RieDFMCombinedLoss

    criterion = RieDFMCombinedLoss(
        manifold=manifold,
        num_edge_types=cfg.data.num_edge_types,
        text_dim=cfg.model.text_dim,
        lambda_disc=cfg.training.lambda_disc,
        mu_align=cfg.training.mu_align,
        temperature=cfg.training.temperature,
    ).to(device)

    # Build dataset
    from riedfm.data.collator import RieDFMGraphCollator
    from riedfm.data.wikidata_dataset import RieDFMWikiDataDataset

    dataset = RieDFMWikiDataDataset(
        data_dir=cfg.data.data_dir,
        manifold=manifold,
        max_nodes=cfg.data.max_nodes,
        num_edge_types=cfg.data.num_edge_types,
    )
    collator = RieDFMGraphCollator(manifold=manifold, max_nodes=cfg.data.max_nodes)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.get("num_workers", 4),
        collate_fn=collator,
    )

    # Setup logging
    from riedfm.utils.logging import log_manifold_stats, log_metrics, setup_wandb

    wandb_run = setup_wandb(OmegaConf.to_container(cfg, resolve=True))  # noqa: F841

    # Training loop
    global_step = 0
    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_losses = {"loss": 0.0, "loss_continuous": 0.0, "loss_discrete": 0.0}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            x = batch["x"].to(device)
            edge_types = batch["edge_types"].to(device)
            depth = batch["depth"].to(device)
            B = batch["batch_size"]

            # Process each graph in the batch independently
            # (for simplicity; batched processing can be optimized later)
            batch_loss = torch.tensor(0.0, device=device)
            for b in range(B):
                n = batch["num_real_nodes"][b].item()
                outputs = model(
                    x_1=x[b, :n],
                    e_1=edge_types[b, :n, :n],
                    depth=depth[b, :n],
                )
                losses = criterion(outputs)
                batch_loss = batch_loss + losses["loss"] / B

            # Gradient accumulation
            batch_loss = batch_loss / cfg.training.gradient_accumulation_steps
            batch_loss.backward()

            if (batch_idx + 1) % cfg.training.gradient_accumulation_steps == 0:
                from riedfm.utils.riemannian_optim import clip_riemannian_grad

                clip_riemannian_grad(model, cfg.training.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % 100 == 0:
                    log_metrics(
                        {"loss": batch_loss.item() * cfg.training.gradient_accumulation_steps},
                        global_step,
                        prefix="train/",
                    )
                    log_manifold_stats(model, global_step)

            epoch_losses["loss"] += batch_loss.item() * cfg.training.gradient_accumulation_steps
            num_batches += 1
            pbar.set_postfix(loss=f"{batch_loss.item() * cfg.training.gradient_accumulation_steps:.4f}")

        # Epoch summary
        avg_loss = epoch_losses["loss"] / max(num_batches, 1)
        logger.info(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            from riedfm.utils.checkpoint import save_checkpoint

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                step=global_step,
                path=f"{cfg.output_dir}/checkpoint_epoch{epoch + 1}.pt",
                config=OmegaConf.to_container(cfg, resolve=True),
            )

    logger.info("Pretraining complete!")


if __name__ == "__main__":
    main()
