"""Batch collation for graph data with virtual node padding.

Pads graphs in a batch to N_max and stacks into batched tensors.
Aligned with math spec Definitions 3.7-3.8:
- Virtual node edges: E[i,j] = 0_K for all virtual nodes       [Def 3.7]
- Virtual node text: c_i = 0_{d_c}                              [Def 3.7]
- Node mask: m_i = 1 for real nodes, 0 for virtual              [Def 3.8]

Masked node prediction (MLM-style pretraining for KGC):
- A fraction of real nodes are randomly masked each batch.
- BERT-style strategy: 80% replaced with [MASK], 10% replaced with
  a random entity, 10% kept unchanged — all are prediction targets.
- Masking is applied at collation time so the model receives
  ``mask_type`` alongside the standard batch tensors.

Note: Virtual node *coordinates* (anchor points on the manifold) are NOT
set here — that is a model-level concern handled during flow matching.
"""

import torch

from riemannfm.data.graph import RiemannFMGraphData

# Mask type constants used in mask_type tensor.
MASK_REAL: int = 0      # real node, unmasked
MASK_MASKED: int = 1    # masked node (prediction target)
MASK_VIRTUAL: int = -1  # virtual (padding) node


class RiemannFMGraphCollator:
    """Collates a list of RiemannFMGraphData into a padded batch.

    All graphs are padded to max_nodes (N_max) with virtual nodes that have
    zero edges, zero text embeddings, and mask=0.

    Args:
        max_nodes: Fixed N_max to pad all graphs to.
            If None, pads to the maximum in the batch.
        num_edge_types: Total number of relation types K.
            If None, inferred from the first sample.
        mask_ratio: Fraction of real nodes to mask per graph (0 to disable).
    """

    def __init__(
        self,
        max_nodes: int | None = None,
        num_edge_types: int | None = None,
        mask_ratio: float = 0.0,
    ):
        self.max_nodes = max_nodes
        self.num_edge_types = num_edge_types
        self.mask_ratio = mask_ratio

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
                - mask_type:       (B, N_max)             long     0=real, 1=masked, -1=virtual
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
        mask_type = torch.full((B, N_max), MASK_VIRTUAL, dtype=torch.long)

        for i, g in enumerate(batch):
            n = min(g.total_nodes, N_max)
            edge_types[i, :n, :n, :K] = g.edge_types[:n, :n, :K]
            node_text[i, :n] = g.node_text[:n]
            node_mask[i, :n] = g.node_mask[:n]
            node_ids[i, :n] = g.node_ids[:n]
            nr = min(g.num_nodes, N_max)
            num_real[i] = nr
            # Real nodes default to MASK_REAL (0); virtual stays MASK_VIRTUAL (-1).
            mask_type[i, :nr] = MASK_REAL

        # Apply masked node prediction if enabled.
        if self.mask_ratio > 0:
            _apply_node_masking(mask_type, num_real, self.mask_ratio)

        return {
            "edge_types": edge_types,
            "node_text": node_text,
            "node_mask": node_mask,
            "node_ids": node_ids,
            "num_real_nodes": num_real,
            "mask_type": mask_type,
            "batch_size": B,
        }


def _apply_node_masking(
    mask_type: torch.Tensor,
    num_real: torch.Tensor,
    mask_ratio: float,
) -> None:
    """Mark a fraction of real nodes as masked (BERT 80/10/10 strategy).

    Modifies ``mask_type`` in-place.  For each graph in the batch:
      - Select ``ceil(mask_ratio * num_real)`` real nodes at random
      - 80% get ``MASK_MASKED`` (will be replaced with [MASK] embedding)
      - 10% get ``MASK_MASKED`` (will be replaced with random entity)
      - 10% get ``MASK_MASKED`` (kept unchanged — still prediction targets)

    All selected nodes share the same ``MASK_MASKED`` label; the 80/10/10
    replacement strategy is applied later in the lightning module where
    entity embeddings are available.

    Args:
        mask_type: Tensor of shape ``(B, N_max)`` to modify in-place.
        num_real: Tensor of shape ``(B,)`` with real node counts.
        mask_ratio: Fraction of real nodes to mask.
    """
    B = mask_type.shape[0]
    for i in range(B):
        nr = int(num_real[i].item())
        if nr < 2:
            continue
        # Always keep at least 1 unmasked node for context.
        n_mask = max(1, min(int(torch.ceil(torch.tensor(mask_ratio * nr)).item()), nr - 1))
        perm = torch.randperm(nr, device=mask_type.device)[:n_mask]
        mask_type[i, perm] = MASK_MASKED
