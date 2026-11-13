"""Graph data container aligned with math spec Definitions 3.4-3.8.

A subgraph sample is the 5-tuple (X, E, C_V, C_R, m):
- X:   node coordinates on product manifold, shape (N, D)       [Def 3.3]
- E:   multi-hot edge type tensor, shape (N, N, K)              [Def 3.4]
- C_V: node text condition matrix, shape (N, d_c)               [Def 3.6]
- C_R: relation text condition matrix, shape (K, d_c)           [Def 3.6]
- m:   node mask, shape (N,)                                    [Def 3.8]

At the data-pipeline level we do NOT produce manifold coordinates X;
those are created by the flow-matching training loop (noise prior + interpolation).
The relation text matrix (spec C_R) is global and shared across all subgraphs --
loaded once by the DataModule (exposed as ``DataModule.relation_text``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor


@dataclass(slots=True)
class RiemannFMGraphData:
    """A single knowledge (sub)graph sample.

    Attributes:
        edge_types: Multi-hot edge type tensor E in {0,1}^{N x N x K}.
            E[i,j,k] = 1 iff relation r_k exists from node i to node j.
        node_text: Precomputed node text embeddings C_V in R^{N x d_c}.
        node_mask: Boolean mask m in {0,1}^N. 1 = real node, 0 = virtual.
        num_nodes: Number of real (non-virtual) nodes |V_G|.
        node_ids: Global entity integer IDs, shape (N,). Used for indexing
            learnable entity embeddings. -1 for virtual nodes.
        num_edge_types: Total number of relation types K.
    """

    edge_types: Tensor      # (N, N, K), binary
    node_text: Tensor        # (N, d_c), float
    node_mask: Tensor        # (N,), bool
    num_nodes: int
    node_ids: Tensor         # (N,), long
    num_edge_types: int

    @property
    def num_real_nodes(self) -> int:
        """Number of real (non-virtual) nodes."""
        return int(self.node_mask.sum().item())

    @property
    def num_edges(self) -> int:
        """Number of (i, j) pairs with at least one edge type."""
        return int((self.edge_types.sum(dim=-1) > 0).sum().item())

    @property
    def total_nodes(self) -> int:
        """Total number of nodes including virtual (= N_max after padding)."""
        return self.edge_types.shape[0]

    @property
    def n_relations(self) -> int:
        """Number of relation types K."""
        return self.edge_types.shape[2]

    @property
    def device(self) -> torch.device:
        return self.edge_types.device

    def to(self, device: torch.device) -> RiemannFMGraphData:
        """Move all tensors to device."""
        return RiemannFMGraphData(
            edge_types=self.edge_types.to(device),
            node_text=self.node_text.to(device),
            node_mask=self.node_mask.to(device),
            num_nodes=self.num_nodes,
            node_ids=self.node_ids.to(device),
            num_edge_types=self.num_edge_types,
        )
