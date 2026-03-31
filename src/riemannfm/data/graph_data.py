"""Graph data container with manifold coordinates and edge types.

Lightweight data class for representing knowledge (sub)graphs with:
- Node coordinates on the product manifold
- Edge type matrix
- Text features (entity labels/descriptions)
- Hierarchy depth information
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class RiemannFMGraphData:
    """A single knowledge (sub)graph sample.

    Attributes:
        x: Node coordinates on the product manifold, shape (N, total_dim).
        edge_types: Edge type matrix, shape (N, N). Values in {0, ..., K}.
        num_nodes: Number of real (non-virtual) nodes.
        depth: Hierarchy depth per node, shape (N,). Optional.
        text_input_ids: Tokenized text for each node, shape (N, max_text_len). Optional.
        text_attention_mask: Text attention mask, shape (N, max_text_len). Optional.
        node_labels: Entity labels as strings. Optional.
        relation_labels: Relation labels indexed by type. Optional.
        entity_ids: WikiData entity IDs (e.g., "Q42"). Optional.
    """

    x: Tensor
    edge_types: Tensor
    num_nodes: int
    depth: Tensor | None = None
    text_input_ids: Tensor | None = None
    text_attention_mask: Tensor | None = None
    node_labels: list[str] = field(default_factory=list)
    relation_labels: dict[int, str] = field(default_factory=dict)
    entity_ids: list[str] = field(default_factory=list)

    @property
    def num_edges(self) -> int:
        """Number of non-zero edges."""
        return int((self.edge_types > 0).sum().item())

    @property
    def device(self) -> torch.device:
        return self.x.device

    def to(self, device: torch.device) -> "RiemannFMGraphData":
        """Move all tensors to device."""
        return RiemannFMGraphData(
            x=self.x.to(device),
            edge_types=self.edge_types.to(device),
            num_nodes=self.num_nodes,
            depth=self.depth.to(device) if self.depth is not None else None,
            text_input_ids=self.text_input_ids.to(device) if self.text_input_ids is not None else None,
            text_attention_mask=self.text_attention_mask.to(device) if self.text_attention_mask is not None else None,
            node_labels=self.node_labels,
            relation_labels=self.relation_labels,
            entity_ids=self.entity_ids,
        )
