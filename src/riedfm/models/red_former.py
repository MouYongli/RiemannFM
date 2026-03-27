"""RED-Former: Riemannian Equivariant Dual-Stream Transformer.

The backbone network for RieDFM-G. Stacks multiple REDFormerBlocks
and handles input embedding, output decoding, and text conditioning.
"""

import torch
import torch.nn as nn
from torch import Tensor

from riedfm.layers.ath_norm import TimestepEmbedding
from riedfm.layers.vector_field_head import ContinuousVectorFieldHead, DiscreteEdgeTypeHead
from riedfm.manifolds.product import ProductManifold
from riedfm.models.red_former_block import REDFormerBlock


class REDFormer(nn.Module):
    """Riemannian Equivariant Dual-Stream Transformer.

    Takes noisy node coordinates and edge types, plus timestep and optional
    text condition, and predicts the velocity field and edge type probabilities.

    Args:
        manifold: Product manifold for node coordinates.
        num_layers: Number of RED-Former blocks.
        node_dim: Node hidden feature dimension.
        edge_dim: Edge hidden feature dimension.
        num_heads: Number of attention heads.
        edge_heads: Number of edge attention heads.
        num_edge_types: Total number of edge types (K+1).
        text_dim: Text encoder hidden dim (0 to disable text conditioning).
        text_cross_attn_every: Inject text cross-attention every N layers.
        dropout: Dropout rate.
        use_mrope: Whether to use Manifold RoPE.
    """

    def __init__(
        self,
        manifold: ProductManifold,
        num_layers: int = 12,
        node_dim: int = 768,
        edge_dim: int = 256,
        num_heads: int = 12,
        edge_heads: int = 4,
        num_edge_types: int = 1201,
        text_dim: int = 1024,
        text_cross_attn_every: int = 3,
        dropout: float = 0.1,
        use_mrope: bool = True,
    ):
        super().__init__()
        self.manifold = manifold
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_edge_types = num_edge_types

        # Time embedding
        time_embed_dim = node_dim // 3  # Compact time embedding
        self.time_embed = TimestepEmbedding(time_embed_dim)

        # Input projections
        # Node: project manifold coordinates to hidden dim
        self.node_input_proj = nn.Linear(manifold.total_dim, node_dim)
        # Edge: embed discrete edge type + project to hidden dim
        self.edge_type_embed = nn.Embedding(num_edge_types, edge_dim)

        # Stack of RED-Former blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            # Include text cross-attention every `text_cross_attn_every` layers
            layer_text_dim = text_dim if (i % text_cross_attn_every == 0) else 0
            self.blocks.append(
                REDFormerBlock(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    num_heads=num_heads,
                    edge_heads=edge_heads,
                    time_embed_dim=time_embed_dim,
                    text_dim=layer_text_dim,
                    dropout=dropout,
                    use_mrope=use_mrope,
                )
            )

        # Output heads
        self.vector_field_head = ContinuousVectorFieldHead(node_dim, manifold)
        self.edge_type_head = DiscreteEdgeTypeHead(edge_dim, num_edge_types)

        # Final layer norms
        self.final_node_norm = nn.LayerNorm(node_dim)
        self.final_edge_norm = nn.LayerNorm(edge_dim)

    def forward(
        self,
        x_t: Tensor,
        e_t: Tensor,
        t: Tensor,
        text_embeds: Tensor | None = None,
        text_mask: Tensor | None = None,
        depth: Tensor | None = None,
        node_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass: predict velocity field and edge type probabilities.

        Args:
            x_t: Current noisy node coordinates, shape (N, total_dim).
            e_t: Current noisy edge types, shape (N, N), long tensor.
            t: Current timestep, scalar or shape (1,).
            text_embeds: Optional text token embeddings, shape (T, text_dim).
            text_mask: Optional text attention mask, shape (T,).
            depth: Optional hierarchy depth per node, shape (N,).
            node_mask: Optional node attention mask, shape (N, N).

        Returns:
            (v_pred, p_pred):
                v_pred: Predicted tangent vectors, shape (N, total_dim).
                p_pred: Edge type logits, shape (N, N, num_edge_types).
        """
        N = x_t.shape[0]

        # Compute time embedding
        t_embed = self.time_embed(t)  # (1, time_embed_dim) or (time_embed_dim,)
        if t_embed.dim() == 1:
            t_embed = t_embed.unsqueeze(0)

        # Input projections
        h_v = self.node_input_proj(x_t)  # (N, node_dim)
        h_e = self.edge_type_embed(e_t.long())  # (N, N, edge_dim)

        # Edge mask: non-diagonal entries
        edge_mask = ~torch.eye(N, dtype=torch.bool, device=x_t.device)

        # Process through blocks
        for block in self.blocks:
            h_v, h_e = block(
                h_v=h_v,
                h_e=h_e,
                manifold=self.manifold,
                positions=x_t,
                t_embed=t_embed,
                depth=depth,
                text_embeds=text_embeds,
                text_mask=text_mask,
                edge_mask=edge_mask,
                node_mask=node_mask,
            )

        # Final norms
        h_v = self.final_node_norm(h_v)
        h_e = self.final_edge_norm(h_e)

        # Output heads
        v_pred = self.vector_field_head(h_v, x_t)
        p_pred = self.edge_type_head(h_e)

        return v_pred, p_pred
