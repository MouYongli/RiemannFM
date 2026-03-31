"""RiemannFM evaluation script.

Link prediction:
    python -m riedfm.cli.evaluate checkpoint=outputs/best_model.pt data=fb15k237

Graph generation:
    python -m riedfm.cli.evaluate checkpoint=outputs/pretrain/checkpoint.pt eval.mode=generation
"""

import json
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Evaluation configuration:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ─── Build manifold ───────────────────────────────────────────────
    from riemannfm.manifolds.product import RiemannFMProductManifold

    manifold = RiemannFMProductManifold(
        dim_hyperbolic=cfg.manifold.dim_hyperbolic,
        dim_spherical=cfg.manifold.dim_spherical,
        dim_euclidean=cfg.manifold.dim_euclidean,
    ).to(device)

    results: dict[str, object] = {}
    eval_mode = cfg.eval.get("mode", "link_prediction")

    # ─── Link Prediction Evaluation ───────────────────────────────────
    if eval_mode in ("link_prediction", "all"):
        logger.info("Running link prediction evaluation...")
        results.update(_evaluate_link_prediction(cfg, manifold, device))

    # ─── Graph Generation Evaluation ──────────────────────────────────
    if eval_mode in ("generation", "all"):
        logger.info("Running graph generation evaluation...")
        results.update(_evaluate_generation(cfg, manifold, device))

    # ─── Save results ─────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "eval_results.json"

    # Convert non-serializable types
    serializable = {k: float(v) if isinstance(v, int | float) else v for k, v in results.items()}
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    for k, v in results.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    logger.info("Evaluation complete!")


def _evaluate_link_prediction(
    cfg: DictConfig,
    manifold: object,
    device: torch.device,
) -> dict[str, float]:
    """Run link prediction evaluation on test set."""
    from riemannfm.data.kg_datasets import RiemannFMKGDataset
    from riemannfm.tasks.kgc_lp import RiemannFMLinkPredictionHead
    from riemannfm.utils.metrics import evaluate_link_prediction

    test_dataset = RiemannFMKGDataset(
        data_dir=cfg.data.data_dir,
        manifold=manifold,  # type: ignore[arg-type]
        split="test",
        mode="triple",
    )

    if test_dataset.num_entities == 0:
        logger.error("No test data found.")
        return {}

    # Build and load link prediction head
    lp_head = RiemannFMLinkPredictionHead(
        manifold=manifold,  # type: ignore[arg-type]
        num_entities=test_dataset.num_entities,
        num_relations=test_dataset.num_relations,
    ).to(device)

    # Load checkpoint
    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            lp_head.load_state_dict(checkpoint["model_state_dict"], strict=False)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warning("Checkpoint has no model_state_dict, using random weights")

    lp_head.eval()

    results = evaluate_link_prediction(
        lp_head,
        test_dataset,
        device,
        batch_size=cfg.eval.get("batch_size", 64),
    )

    return {f"lp_{k}": v for k, v in results.items()}


def _evaluate_generation(
    cfg: DictConfig,
    manifold: object,
    device: torch.device,
) -> dict[str, float]:
    """Run graph generation evaluation."""
    from riemannfm.models.riemannfm import RiemannFM
    from riemannfm.utils.metrics import clustering_mmd, compute_vun, degree_mmd, spectral_mmd

    # Build model
    model = RiemannFM(
        manifold=manifold,  # type: ignore[arg-type]
        num_layers=cfg.model.num_layers,
        node_dim=cfg.model.node_dim,
        edge_dim=cfg.model.edge_dim,
        num_heads=cfg.model.num_heads,
        num_edge_types=cfg.data.num_edge_types,
        dropout=0.0,
    ).to(device)

    # Load checkpoint
    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    num_samples = cfg.eval.get("num_generation_samples", 100)
    num_steps = cfg.eval.get("num_inference_steps", 100)
    max_nodes = cfg.data.get("max_nodes", 32)

    # Generate graphs
    logger.info(f"Generating {num_samples} graphs...")
    gen_adjs = []
    with torch.no_grad():
        for i in range(num_samples):
            _x, e = model.generate(num_nodes=max_nodes, num_steps=num_steps, device=device)
            gen_adjs.append(e.cpu().numpy())
            if (i + 1) % 50 == 0:
                logger.info(f"  Generated {i + 1}/{num_samples}")

    # Load reference graphs from dataset
    from riemannfm.data.kg_datasets import RiemannFMKGDataset

    ref_dataset = RiemannFMKGDataset(
        data_dir=cfg.data.data_dir,
        manifold=manifold,  # type: ignore[arg-type]
        split="test",
        mode="subgraph",
        max_nodes=max_nodes,
    )

    num_ref = min(len(ref_dataset), num_samples)
    ref_adjs = []
    for i in range(num_ref):
        sample = ref_dataset[i]
        if hasattr(sample, "edge_types"):
            ref_adjs.append(sample.edge_types.numpy())

    if not ref_adjs:
        logger.warning("No reference graphs available. Skipping generation metrics.")
        return {}

    # Compute metrics
    results: dict[str, float] = {}
    results["gen_degree_mmd"] = degree_mmd(gen_adjs, ref_adjs)
    results["gen_clustering_mmd"] = clustering_mmd(gen_adjs, ref_adjs)
    results["gen_spectral_mmd"] = spectral_mmd(gen_adjs, ref_adjs)

    vun = compute_vun(gen_adjs, ref_adjs)
    results["gen_validity"] = vun["validity"]
    results["gen_uniqueness"] = vun["uniqueness"]
    results["gen_novelty"] = vun["novelty"]

    return results


if __name__ == "__main__":
    main()
