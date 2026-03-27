"""RieDFM-G fine-tuning script for link prediction on KG benchmarks.

Usage:
    python -m riedfm.cli.finetune data=fb15k237 training=finetune
    python -m riedfm.cli.finetune data=wn18rr training=finetune training.lr=1e-5
    python -m riedfm.cli.finetune data=fb15k237 pretrained_checkpoint=outputs/pretrain/checkpoint.pt
"""

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Fine-tuning configuration:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ─── Build manifold ───────────────────────────────────────────────
    from riedfm.manifolds.product import RieDFMProductManifold

    manifold = RieDFMProductManifold(
        dim_hyperbolic=cfg.manifold.dim_hyperbolic,
        dim_spherical=cfg.manifold.dim_spherical,
        dim_euclidean=cfg.manifold.dim_euclidean,
        init_curvature_h=cfg.manifold.init_curvature_h,
        init_curvature_s=cfg.manifold.init_curvature_s,
        learn_curvature=cfg.manifold.learn_curvature,
    ).to(device)

    # ─── Load KG benchmark dataset ────────────────────────────────────
    from riedfm.data.kg_datasets import RieDFMKGDataset

    train_dataset = RieDFMKGDataset(
        data_dir=cfg.data.data_dir,
        manifold=manifold,
        split="train",
        mode="triple",
    )
    val_dataset = RieDFMKGDataset(
        data_dir=cfg.data.data_dir,
        manifold=manifold,
        split="valid",
        mode="triple",
    )

    if train_dataset.num_entities == 0:
        logger.error(f"No data found in {cfg.data.data_dir}. Check data_dir path.")
        return

    logger.info(
        f"Dataset: {train_dataset.num_entities} entities, "
        f"{train_dataset.num_relations} relations, "
        f"{len(train_dataset)} train triples"
    )

    def triple_collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        return {
            "head": torch.tensor([b["head"] for b in batch]),
            "relation": torch.tensor([b["relation"] for b in batch]),
            "tail": torch.tensor([b["tail"] for b in batch]),
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.get("num_workers", 4),
        collate_fn=triple_collate,
        drop_last=True,
    )

    # ─── Build link prediction head ───────────────────────────────────
    from riedfm.models.downstream.link_prediction import RieDFMLinkPredictionHead

    lp_head = RieDFMLinkPredictionHead(
        manifold=manifold,
        num_entities=train_dataset.num_entities,
        num_relations=train_dataset.num_relations,
    ).to(device)

    # Optionally load pretrained backbone weights into entity embeddings
    pretrained_path = cfg.get("pretrained_checkpoint", None)
    if pretrained_path:
        from riedfm.utils.checkpoint import load_checkpoint

        load_checkpoint(pretrained_path, device=device)
        logger.info(f"Loaded pretrained checkpoint from {pretrained_path}")

    # ─── Optimizer + Scheduler ────────────────────────────────────────
    from riedfm.utils.riemannian_optim import build_optimizer
    from riedfm.utils.scheduler import build_cosine_scheduler

    optimizer = build_optimizer(
        lp_head,
        lr=cfg.training.lr,
        curvature_lr=cfg.training.curvature_lr,
        weight_decay=cfg.training.weight_decay,
    )

    accum_steps = cfg.training.gradient_accumulation_steps
    total_steps = cfg.training.epochs * len(train_loader) // accum_steps
    scheduler = build_cosine_scheduler(
        optimizer,
        warmup_steps=cfg.training.warmup_steps,
        total_steps=total_steps,
    )

    num_params = sum(p.numel() for p in lp_head.parameters() if p.requires_grad)
    logger.info(f"Link prediction head parameters: {num_params:,}")

    # ─── Training loop ────────────────────────────────────────────────
    from riedfm.utils.logging import log_metrics, setup_wandb

    setup_wandb(OmegaConf.to_container(cfg, resolve=True))

    eval_every = cfg.training.get("eval_every", 5)
    best_mrr = 0.0
    global_step = 0

    for epoch in range(cfg.training.epochs):
        lp_head.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs}")
        for batch_idx, batch in enumerate(pbar):
            h = batch["head"].to(device)
            r = batch["relation"].to(device)
            t = batch["tail"].to(device)

            loss = lp_head.link_prediction_loss(h, r, t)
            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(lp_head.parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                # Re-project entity embeddings onto manifold
                lp_head.project_embeddings()
                global_step += 1

                if global_step % 100 == 0:
                    log_metrics(
                        {"loss": loss.item(), "lr": float(scheduler.get_last_lr()[0])},
                        global_step,
                        prefix="finetune/",
                    )

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

        # ─── Validation ──────────────────────────────────────────
        if (epoch + 1) % eval_every == 0:
            from riedfm.utils.metrics import evaluate_link_prediction

            lp_head.eval()
            results = evaluate_link_prediction(
                lp_head,
                val_dataset,
                device,
                batch_size=cfg.training.get("eval_batch_size", 64),
            )

            logger.info(
                f"Validation: MRR={results['mrr']:.4f} "
                f"H@1={results['hits_at_1']:.4f} "
                f"H@3={results['hits_at_3']:.4f} "
                f"H@10={results['hits_at_10']:.4f}"
            )
            log_metrics(results, global_step, prefix="val/")

            # Save best model
            if results["mrr"] > best_mrr:
                best_mrr = results["mrr"]
                from riedfm.utils.checkpoint import save_checkpoint

                save_checkpoint(
                    model=lp_head,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    step=global_step,
                    path=f"{cfg.output_dir}/best_model.pt",
                    config=OmegaConf.to_container(cfg, resolve=True),
                    best_mrr=best_mrr,
                )
                logger.info(f"New best MRR: {best_mrr:.4f}, saved to {cfg.output_dir}/best_model.pt")

    logger.info(f"Fine-tuning complete! Best MRR: {best_mrr:.4f}")


if __name__ == "__main__":
    main()
