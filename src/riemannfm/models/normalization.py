"""Adaptive Time-Hierarchy Normalization (ATH-Norm).

Conditions LayerNorm affine parameters on both:
- Flow matching timestep t (controls denoising stage)
- Hierarchy depth (controls coarse-to-fine generation order)

Early in generation (small t): high-level nodes are active, low-level frozen.
Late in generation (large t): low-level nodes become active for fine detail.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class RiemannFMTimestepEmbedding(nn.Module):
    """Sinusoidal + MLP embedding for scalar timestep t in [0, 1].

    Uses sinusoidal positional encoding followed by a 2-layer MLP,
    similar to the time embedding in diffusion models.
    """

    def __init__(self, embed_dim: int, max_period: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        # Frequency bands for sinusoidal encoding
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: Timestep values in [0, 1], shape (...).

        Returns:
            Time embeddings, shape (..., embed_dim).
        """
        # Ensure t has at least 1 dimension
        if t.dim() == 0:
            t = t.unsqueeze(0)
        freqs: Tensor = self.freqs  # type: ignore[assignment]
        args = t.unsqueeze(-1) * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # Handle odd embed_dim
        if self.embed_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return Tensor(self.mlp(embedding))


class RiemannFMATHNorm(nn.Module):
    """Adaptive Time-Hierarchy Normalization.

    Extends AdaLN (Adaptive Layer Normalization) by conditioning on both
    timestep and hierarchy depth:

        gamma_i, beta_i = MLP(t_embed, depth_i)
        output_i = gamma_i * LayerNorm(h_i) + beta_i

    This enables coarse-to-fine generation: high-level (shallow depth)
    nodes activate early, low-level (deep) nodes activate later.

    Args:
        hidden_dim: Feature dimension to normalize.
        time_embed_dim: Dimension of timestep embedding.
        max_depth: Maximum hierarchy depth (for depth embedding).
    """

    def __init__(self, hidden_dim: int, time_embed_dim: int, max_depth: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # Depth embedding: learnable lookup table
        self.depth_embed = nn.Embedding(max_depth, time_embed_dim)
        # MLP that produces (gamma, beta) from (time_embed + depth_embed)
        self.adaLN_mlp = nn.Sequential(
            nn.Linear(time_embed_dim * 2, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # gamma and beta
        )
        # Initialize to identity (gamma=1, beta=0)
        last_layer = self.adaLN_mlp[-1]
        assert isinstance(last_layer, nn.Linear)
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def forward(
        self,
        h: Tensor,
        t_embed: Tensor,
        depth: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            h: Node features, shape (N, hidden_dim).
            t_embed: Timestep embeddings, shape (N, time_embed_dim) or (1, time_embed_dim).
            depth: Hierarchy depth indices, shape (N,). If None, uses depth=0 for all.

        Returns:
            Normalized features, shape (N, hidden_dim).
        """
        N = h.shape[0]
        device = h.device

        if depth is None:
            depth = torch.zeros(N, dtype=torch.long, device=device)
        depth = depth.clamp(max=self.depth_embed.num_embeddings - 1)

        d_embed = self.depth_embed(depth)  # (N, time_embed_dim)

        # Flatten t_embed to 2D (N, time_embed_dim) and broadcast if needed
        while t_embed.dim() > 2:
            t_embed = t_embed.squeeze(0)
        if t_embed.shape[0] == 1 and N > 1:
            t_embed = t_embed.expand(N, -1)

        # Concatenate time and depth embeddings
        cond = torch.cat([t_embed, d_embed], dim=-1)  # (N, 2 * time_embed_dim)
        params = self.adaLN_mlp(cond)  # (N, 2 * hidden_dim)

        gamma, beta = params.chunk(2, dim=-1)
        gamma = 1.0 + gamma  # Center around 1

        return Tensor(gamma * self.norm(h) + beta)
