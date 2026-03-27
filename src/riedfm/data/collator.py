"""Batch collation for graph data with virtual node padding.

Pads graphs in a batch to the same size using virtual (absorbing) nodes,
and stacks them into batched tensors.
"""

import torch

from riedfm.data.graph_data import RieDFMGraphData
from riedfm.manifolds.product import RieDFMProductManifold


class RieDFMGraphCollator:
    """Collates a list of RieDFMGraphData into a padded batch.

    Pads all graphs to max_nodes using virtual nodes:
    - Virtual node coordinates are set to the manifold origin
    - Virtual edges are type 0 (no edge)
    - A mask distinguishes real from virtual nodes

    Args:
        manifold: Product manifold for computing the origin point.
        max_nodes: Maximum number of nodes (pad to this size).
                   If None, pads to the maximum in the batch.
    """

    def __init__(self, manifold: RieDFMProductManifold, max_nodes: int | None = None):
        self.manifold = manifold
        self.max_nodes = max_nodes

    def __call__(self, batch: list[RieDFMGraphData]) -> dict[str, object]:
        """Collate a batch of RieDFMGraphData.

        Args:
            batch: List of RieDFMGraphData samples.

        Returns:
            Dictionary with batched tensors:
                - x: (B, N_max, D) node coordinates
                - edge_types: (B, N_max, N_max) edge type matrices
                - node_mask: (B, N_max) boolean mask (True = real node)
                - depth: (B, N_max) hierarchy depths
                - text_input_ids: (B, N_max, T) if available
                - text_attention_mask: (B, N_max, T) if available
                - batch_size: int
                - num_real_nodes: (B,) number of real nodes per graph
        """
        B = len(batch)
        N_max = self.max_nodes or max(g.num_nodes for g in batch)
        device = batch[0].device

        # Get manifold origin for virtual nodes
        origin = self.manifold.origin(device)  # (D,)

        # Initialize padded tensors
        x = origin.unsqueeze(0).unsqueeze(0).expand(B, N_max, -1).clone()
        edge_types = torch.zeros(B, N_max, N_max, dtype=torch.long, device=device)
        node_mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)
        depth = torch.zeros(B, N_max, dtype=torch.long, device=device)
        num_real = torch.zeros(B, dtype=torch.long, device=device)

        for i, g in enumerate(batch):
            n = min(g.num_nodes, N_max)
            x[i, :n] = g.x[:n]
            edge_types[i, :n, :n] = g.edge_types[:n, :n]
            node_mask[i, :n] = True
            num_real[i] = n
            if g.depth is not None:
                depth[i, :n] = g.depth[:n]

        result = {
            "x": x,
            "edge_types": edge_types,
            "node_mask": node_mask,
            "depth": depth,
            "batch_size": B,
            "num_real_nodes": num_real,
        }

        # Handle text features
        has_text = all(g.text_input_ids is not None for g in batch)
        if has_text:
            T = max(g.text_input_ids.shape[-1] for g in batch)
            text_ids = torch.zeros(B, N_max, T, dtype=torch.long, device=device)
            text_mask = torch.zeros(B, N_max, T, dtype=torch.long, device=device)
            for i, g in enumerate(batch):
                n = min(g.num_nodes, N_max)
                t = g.text_input_ids.shape[-1]
                text_ids[i, :n, :t] = g.text_input_ids[:n]
                text_mask[i, :n, :t] = g.text_attention_mask[:n]
            result["text_input_ids"] = text_ids
            result["text_attention_mask"] = text_mask

        return result
