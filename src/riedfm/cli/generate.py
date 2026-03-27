"""RieDFM-G graph generation / inference script.

Usage:
    python -m riedfm.cli.generate --checkpoint path/to/checkpoint.pt
    python -m riedfm.cli.generate --checkpoint path/to/checkpoint.pt --num_nodes 32 --num_steps 200
"""

import argparse
import logging

import torch

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate graphs with RieDFM-G")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_nodes", type=int, default=32, help="Number of nodes to generate")
    parser.add_argument("--num_steps", type=int, default=100, help="ODE integration steps")
    parser.add_argument("--num_graphs", type=int, default=10, help="Number of graphs to generate")
    parser.add_argument("--output", type=str, default="generated_graphs.pt", help="Output path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--text_query", type=str, default=None, help="Optional text condition")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Build model from checkpoint config
    from riedfm.manifolds.product import RieDFMProductManifold

    manifold_cfg = config.get("manifold", {})
    manifold = RieDFMProductManifold(
        dim_hyperbolic=manifold_cfg.get("dim_hyperbolic", 32),
        dim_spherical=manifold_cfg.get("dim_spherical", 32),
        dim_euclidean=manifold_cfg.get("dim_euclidean", 32),
    ).to(device)

    from riedfm.models.riedfm_g import RieDFMG

    model_cfg = config.get("model", {})
    model = RieDFMG(
        manifold=manifold,
        num_layers=model_cfg.get("num_layers", 12),
        node_dim=model_cfg.get("node_dim", 768),
        edge_dim=model_cfg.get("edge_dim", 256),
        num_heads=model_cfg.get("num_heads", 12),
        num_edge_types=config.get("data", {}).get("num_edge_types", 1201),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Generate
    results = []
    for i in range(args.num_graphs):
        x, e = model.generate(
            num_nodes=args.num_nodes,
            num_steps=args.num_steps,
            device=device,
        )

        # Remove virtual nodes
        from riedfm.utils.virtual_nodes import remove_virtual_nodes

        x_real, e_real = remove_virtual_nodes(x, e, manifold)
        results.append({"x": x_real.cpu(), "edge_types": e_real.cpu()})
        logger.info(
            f"Generated graph {i + 1}/{args.num_graphs}: {x_real.shape[0]} nodes, {(e_real > 0).sum().item()} edges"
        )

    # Save
    torch.save(results, args.output)
    logger.info(f"Saved {len(results)} graphs to {args.output}")


if __name__ == "__main__":
    main()
