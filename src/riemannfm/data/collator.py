"""Batch collation for graph data with virtual node padding.

Pads graphs in a batch to N_max and stacks into batched tensors.
Aligned with math spec Definitions 3.7-3.8:
- Virtual node edges: E[i,j] = 0_K for all virtual nodes       [Def 3.7]
- Virtual node text: c_i = 0_{d_c}                              [Def 3.7]
- Node mask: m_i = 1 for real nodes, 0 for virtual              [Def 3.8]

Note: Virtual node *coordinates* (anchor points on the manifold) are NOT
set here — that is a model-level concern handled during flow matching.
"""

import torch

from riemannfm.data.graph import RiemannFMGraphData


class RiemannFMGraphCollator:
    """Collates a list of RiemannFMGraphData into a padded batch.

    All graphs are padded to max_nodes (N_max) with virtual nodes that have
    zero edges, zero text embeddings, and mask=0.

    Args:
        max_nodes: Fixed N_max to pad all graphs to.
            If None, pads to the maximum in the batch.
        num_edge_types: Total number of relation types K.
            If None, inferred from the first sample.
    """

    def __init__(
        self,
        max_nodes: int | None = None,
        num_edge_types: int | None = None,
    ):
        self.max_nodes = max_nodes
        self.num_edge_types = num_edge_types

    def __call__(self, batch: list[RiemannFMGraphData]) -> dict[str, object]:
        """Collate a batch of RiemannFMGraphData into padded tensors.

        Args:
            batch: List of RiemannFMGraphData samples.

        Returns:
            Dictionary with batched tensors:
                - edge_types:      (B, N_max, N_max, K)  float32  multi-hot
                - node_text:       (B, N_max, d_c)       float32  text embeddings
                - node_mask:       (B, N_max)             bool     real vs virtual
                - node_ids:        (B, N_max)             long     entity IDs (-1 = virtual)
                - num_real_nodes:  (B,)                   long     real node counts
                - batch_size:      int
        """
        B = len(batch)
        N_max = self.max_nodes or max(g.total_nodes for g in batch)
        K = self.num_edge_types or batch[0].n_relations
        d_c = batch[0].node_text.shape[-1]

        # Initialize padded tensors (zeros = virtual node defaults per Def 3.7)
        edge_types = torch.zeros(B, N_max, N_max, K, dtype=torch.float32)
        node_text = torch.zeros(B, N_max, d_c, dtype=torch.float32)
        node_mask = torch.zeros(B, N_max, dtype=torch.bool)
        node_ids = torch.full((B, N_max), -1, dtype=torch.long)
        num_real = torch.zeros(B, dtype=torch.long)

        for i, g in enumerate(batch):
            n = min(g.total_nodes, N_max)
            edge_types[i, :n, :n, :K] = g.edge_types[:n, :n, :K]
            node_text[i, :n] = g.node_text[:n]
            node_mask[i, :n] = g.node_mask[:n]
            node_ids[i, :n] = g.node_ids[:n]
            num_real[i] = min(g.num_nodes, N_max)

        return {
            "edge_types": edge_types,
            "node_text": node_text,
            "node_mask": node_mask,
            "node_ids": node_ids,
            "num_real_nodes": num_real,
            "batch_size": B,
        }
