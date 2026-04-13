"""Batch collation for graph data with virtual node padding.

Pads graphs in a batch to N_max and stacks into batched tensors.
Aligned with math spec Definitions 3.7-3.8:
- Virtual node edges: E[i,j] = 0_K for all virtual nodes       [Def 3.7]
- Virtual node text: c_i = 0_{d_c}                              [Def 3.7]
- Node mask: m_i = 1 for real nodes, 0 for virtual              [Def 3.8]

Masked node prediction (Def 6.9a — node three-way partition):
- Each real node is labelled REAL, MASK_C, or MASK_X (disjoint).
- ``MASK_C``: text masked, geometry kept.  Target = true text (L_mask_c).
- ``MASK_X``: geometry masked (x forced to noise at t=0), text kept.
  Target = vector field ``u_t`` at t=0 (L_mask_x, shares L_cont machinery).
- ``REAL``:  both real — participates in L_cont and L_align.

The collator emits ``t_node`` per position: ``0`` for M_x, ``1`` for M_c,
``NaN`` for REAL/VIRTUAL (filled later with batch-sampled t in the
lightning module).  This makes collation the single source of truth for
mask semantics; downstream modules consume ``mask_type`` + ``t_node``.

Note: Virtual node *coordinates* (anchor points on the manifold) are NOT
set here — that is a model-level concern handled during flow matching.
"""

import torch

from riemannfm.data.graph import RiemannFMGraphData

# Mask type constants used in mask_type tensor.
MASK_REAL: int = 0      # real node, unmasked — participates in L_cont / L_align
MASK_C: int = 1         # text masked, x kept — participates in L_mask_c
MASK_X: int = 2         # geometry masked (x=noise at t=0), text kept — L_mask_x
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
        mask_ratio_c: Fraction of real nodes assigned to MASK_C (text mask).
        mask_ratio_x: Fraction of real nodes assigned to MASK_X (geom mask).
    """

    def __init__(
        self,
        max_nodes: int | None = None,
        num_edge_types: int | None = None,
        mask_ratio_c: float = 0.0,
        mask_ratio_x: float = 0.0,
    ):
        self.max_nodes = max_nodes
        self.num_edge_types = num_edge_types
        self.mask_ratio_c = mask_ratio_c
        self.mask_ratio_x = mask_ratio_x

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
                - mask_type:       (B, N_max)             long     {-1, 0, 1, 2}
                - t_node:          (B, N_max)             float32  per-node time label
                                                                   (NaN = use batch-t)
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
        t_node = torch.full((B, N_max), float("nan"), dtype=torch.float32)

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

        # Apply masked node partition if enabled.
        if self.mask_ratio_c > 0 or self.mask_ratio_x > 0:
            _apply_node_masking(
                mask_type, t_node, num_real,
                self.mask_ratio_c, self.mask_ratio_x,
            )

        return {
            "edge_types": edge_types,
            "node_text": node_text,
            "node_mask": node_mask,
            "node_ids": node_ids,
            "num_real_nodes": num_real,
            "mask_type": mask_type,
            "t_node": t_node,
            "batch_size": B,
        }


def _apply_node_masking(
    mask_type: torch.Tensor,
    t_node: torch.Tensor,
    num_real: torch.Tensor,
    mask_ratio_c: float,
    mask_ratio_x: float,
) -> None:
    """Partition real nodes into disjoint REAL / MASK_C / MASK_X subsets.

    For each graph:
      - Draw ``n_c = ceil(p_c * nr)`` nodes → MASK_C (t_node = 1.0)
      - From the remainder, draw ``n_x = ceil(p_x * nr)`` → MASK_X (t_node = 0.0)
      - At least one real node is kept as REAL anchor (|U| >= 1)

    Modifies ``mask_type`` and ``t_node`` in-place.

    Args:
        mask_type: Tensor of shape ``(B, N_max)`` to modify in-place.
        t_node: Tensor of shape ``(B, N_max)`` to modify in-place.
            M_x positions written to ``0.0``; M_c positions to ``1.0``.
        num_real: Tensor of shape ``(B,)`` with real node counts.
        mask_ratio_c: Fraction of real nodes to label MASK_C.
        mask_ratio_x: Fraction of real nodes to label MASK_X.
    """
    B = mask_type.shape[0]
    for i in range(B):
        nr = int(num_real[i].item())
        if nr < 2:
            continue
        # Compute target sizes with at-least-one-REAL anchor constraint.
        max_masked = nr - 1
        n_c = min(int(torch.ceil(torch.tensor(mask_ratio_c * nr)).item()), max_masked)
        n_x = min(
            int(torch.ceil(torch.tensor(mask_ratio_x * nr)).item()),
            max_masked - n_c,
        )
        n_c = max(0, n_c)
        n_x = max(0, n_x)

        if n_c == 0 and n_x == 0:
            continue

        perm = torch.randperm(nr, device=mask_type.device)
        c_idx = perm[:n_c]
        x_idx = perm[n_c:n_c + n_x]
        if n_c > 0:
            mask_type[i, c_idx] = MASK_C
            t_node[i, c_idx] = 1.0
        if n_x > 0:
            mask_type[i, x_idx] = MASK_X
            t_node[i, x_idx] = 0.0
