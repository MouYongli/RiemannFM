"""RiemannFM pretraining script.

Single-GPU:
    python -m riedfm.cli.pretrain
    python -m riedfm.cli.pretrain model=red_former_large data=wikidata_5m

Multi-GPU (DDP via torchrun):
    torchrun --nproc_per_node=4 -m riedfm.cli.pretrain training.batch_size=64
"""

import logging
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

logger = logging.getLogger(__name__)


def set_seed(seed: int, rank: int = 0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    from riemannfm.utils.distributed import (
        cleanup_distributed,
        get_rank,
        get_world_size,
        is_distributed,
        is_main_process,
        setup_distributed,
        unwrap_ddp,
        wrap_ddp,
    )

    # Setup distributed / device
    device = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()

    if is_main_process():
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        logger.info(f"World size: {world_size}, Device: {device}")

    set_seed(cfg.seed, rank)

    # ─── Build manifold ───────────────────────────────────────────────
    from riemannfm.manifolds.product import RiemannFMProductManifold

    manifold = RiemannFMProductManifold(
        dim_hyperbolic=cfg.manifold.dim_hyperbolic,
        dim_spherical=cfg.manifold.dim_spherical,
        dim_euclidean=cfg.manifold.dim_euclidean,
        init_curvature_h=cfg.manifold.init_curvature_h,
        init_curvature_s=cfg.manifold.init_curvature_s,
        learn_curvature=cfg.manifold.learn_curvature,
    ).to(device)

    # ─── Build model ──────────────────────────────────────────────────
    from riemannfm.models.riemannfm import RiemannFM

    model = RiemannFM(
        manifold=manifold,
        num_layers=cfg.model.num_layers,
        node_dim=cfg.model.node_dim,
        edge_dim=cfg.model.edge_dim,
        num_heads=cfg.model.num_heads,
        num_edge_types=cfg.data.num_edge_types,
        text_dim=cfg.model.get("text_dim", 0),
        text_cross_attn_every=cfg.model.get("text_cross_attn_every", 3),
        avg_edge_density=cfg.flow.avg_edge_density,
        dropout=cfg.model.dropout,
    ).to(device)

    ddp_model = wrap_ddp(model, device)

    num_params = sum(p.numel() for p in ddp_model.parameters() if p.requires_grad)
    if is_main_process():
        logger.info(f"Model parameters: {num_params:,}")

    # ─── Optimizer ────────────────────────────────────────────────────
    from riemannfm.optim.riemannian import build_optimizer

    optimizer = build_optimizer(
        unwrap_ddp(ddp_model),
        lr=cfg.training.lr,
        curvature_lr=cfg.training.curvature_lr,
        weight_decay=cfg.training.weight_decay,
        use_riemannian=cfg.training.get("use_riemannian_optim", True),
    )

    # ─── Loss ─────────────────────────────────────────────────────────
    from riemannfm.losses.combined_loss import RiemannFMCombinedLoss

    criterion = RiemannFMCombinedLoss(
        manifold=manifold,
        num_edge_types=cfg.data.num_edge_types,
        text_dim=cfg.model.get("text_dim", 0),
        lambda_disc=cfg.training.lambda_disc,
        mu_align=cfg.training.mu_align,
        temperature=cfg.training.get("temperature", 0.07),
    ).to(device)

    # ─── Dataset ──────────────────────────────────────────────────────
    from riemannfm.data.collator import RiemannFMGraphCollator
    from riemannfm.data.wikidata_dataset import RiemannFMWikiDataDataset

    dataset = RiemannFMWikiDataDataset(
        data_dir=cfg.data.data_dir,
        manifold=manifold,
        max_nodes=cfg.data.max_nodes,
        num_edge_types=cfg.data.num_edge_types,
    )

    sampler: DistributedSampler | None = DistributedSampler(dataset, shuffle=True) if is_distributed() else None
    collator = RiemannFMGraphCollator(manifold=manifold, max_nodes=cfg.data.max_nodes)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(sampler is None),
        num_workers=cfg.data.get("num_workers", 4),
        collate_fn=collator,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
    )

    # ─── Mixed precision ──────────────────────────────────────────────
    use_amp = cfg.training.get("mixed_precision", "none") != "none"
    amp_dtype = torch.bfloat16 if cfg.training.get("mixed_precision", "none") == "bf16" else torch.float16
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # ─── Logging ──────────────────────────────────────────────────────
    from riemannfm.utils.logging import log_manifold_stats, log_metrics, setup_wandb

    if is_main_process():
        setup_wandb(OmegaConf.to_container(cfg, resolve=True))

    # ─── Training loop ────────────────────────────────────────────────
    accum_steps = cfg.training.gradient_accumulation_steps
    global_step = 0

    for epoch in range(cfg.training.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        ddp_model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs}", disable=not is_main_process())

        for batch_idx, batch in enumerate(pbar):
            # ─── Move to device ──────────────────────────────────
            x = batch["x"].to(device, non_blocking=True)
            edge_types = batch["edge_types"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            num_real = batch["num_real_nodes"].to(device, non_blocking=True)
            B = x.shape[0]

            # ─── Forward: process each graph in the padded batch ─
            # TODO: Implement true batched forward pass in RiemannFM
            # For now, iterate over graphs but accumulate loss properly
            batch_loss = torch.tensor(0.0, device=device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                for b in range(B):
                    n = num_real[b].item()
                    if n < 2:
                        continue
                    outputs = unwrap_ddp(ddp_model)(
                        x_1=x[b, :n],
                        e_1=edge_types[b, :n, :n],
                        depth=depth[b, :n],
                    )
                    losses = criterion(outputs)
                    batch_loss = batch_loss + losses["loss"] / B

            # ─── Backward ────────────────────────────────────────
            scaled_loss = batch_loss / accum_steps
            scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                from riemannfm.optim.riemannian import clip_riemannian_grad

                scaler.unscale_(optimizer)
                clip_riemannian_grad(unwrap_ddp(ddp_model), cfg.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Logging
                if is_main_process() and global_step % cfg.training.get("log_every", 100) == 0:
                    log_metrics(
                        {"loss": batch_loss.item()},
                        global_step,
                        prefix="train/",
                    )
                    log_manifold_stats(unwrap_ddp(ddp_model), global_step)

            epoch_loss += batch_loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

        # ─── Epoch summary ───────────────────────────────────────────
        avg_loss = epoch_loss / max(num_batches, 1)
        if is_main_process():
            logger.info(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

        # ─── Checkpoint ──────────────────────────────────────────────
        save_every = cfg.training.get("save_every", 10)
        if is_main_process() and (epoch + 1) % save_every == 0:
            from riemannfm.utils.checkpoint import save_checkpoint

            save_checkpoint(
                model=unwrap_ddp(ddp_model),
                optimizer=optimizer,
                epoch=epoch + 1,
                step=global_step,
                path=f"{cfg.output_dir}/checkpoint_epoch{epoch + 1}.pt",
                config=OmegaConf.to_container(cfg, resolve=True),
            )

    if is_main_process():
        logger.info("Pretraining complete!")
    cleanup_distributed()


if __name__ == "__main__":
    main()
