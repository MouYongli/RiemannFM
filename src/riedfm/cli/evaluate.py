"""RieDFM-G evaluation script.

Usage:
    python -m riedfm.cli.evaluate checkpoint=path/to/checkpoint.pt data=fb15k237
"""

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Evaluation configuration:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Build manifold and model
    from riedfm.manifolds.product import RieDFMProductManifold

    manifold = RieDFMProductManifold(
        dim_hyperbolic=cfg.manifold.dim_hyperbolic,
        dim_spherical=cfg.manifold.dim_spherical,
        dim_euclidean=cfg.manifold.dim_euclidean,
    ).to(device)

    from riedfm.models.riedfm_g import RieDFMG

    model = RieDFMG(
        manifold=manifold,
        num_layers=cfg.model.num_layers,
        node_dim=cfg.model.node_dim,
        edge_dim=cfg.model.edge_dim,
        num_heads=cfg.model.num_heads,
        num_edge_types=cfg.data.num_edge_types,
        text_dim=cfg.model.text_dim,
        dropout=0.0,  # No dropout during eval
    ).to(device)

    # Load checkpoint
    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path:
        from riedfm.utils.checkpoint import load_checkpoint

        load_checkpoint(checkpoint_path, model=model, device=device)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    model.eval()

    # Generate graphs
    logger.info(f"Generating {cfg.eval.num_generation_samples} graphs...")
    generated_graphs = []
    with torch.no_grad():
        for i in range(cfg.eval.num_generation_samples):
            x, e = model.generate(
                num_nodes=cfg.data.max_nodes,
                num_steps=cfg.eval.num_inference_steps,
                device=device,
            )
            generated_graphs.append((x.cpu(), e.cpu()))

            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{cfg.eval.num_generation_samples} graphs")

    # Compute metrics
    from riedfm.utils.metrics import compute_vun

    adj_matrices = [e.numpy() for _, e in generated_graphs]
    vun = compute_vun(adj_matrices, adj_matrices[:10])  # Placeholder ref graphs
    logger.info(f"V.U.N. metrics: {vun}")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
