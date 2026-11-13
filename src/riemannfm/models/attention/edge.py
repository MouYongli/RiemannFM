"""Edge self-update and edge-to-attention bias (spec §12.3, §14).

``RiemannFMEdgeBias``: per-head bias for A_V (spec §12.3).

``RiemannFMEdgeSelfUpdate`` (sub-module C, spec §14): refines edge
hidden states using the **current-layer** node hidden pair plus the
geometric direction ``π^T(log_xi(xj))``.  Edge hidden dimension is
``d_b``; node hidden dimension is ``d_v``.  ``π^T`` mirrors the node
encoder's ``π`` — drops the Lorentz time tangent, per-block-LN on
spherical and Euclidean tangent slices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


class RiemannFMEdgeBias(nn.Module):
    """Project edge features into per-head attention bias (spec §12.3).

    Args:
        edge_dim: Edge hidden dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, edge_dim: int, num_heads: int) -> None:
        super().__init__()
        self.proj = nn.Linear(edge_dim, num_heads, bias=False)

    def forward(self, g: Tensor) -> Tensor:
        """Compute edge bias for attention.

        Args:
            g: Edge hidden states, shape ``(B, N, N, edge_dim)``.

        Returns:
            Per-head bias, shape ``(B, num_heads, N, N)``.
        """
        result: Tensor = self.proj(g).permute(0, 3, 1, 2)
        return result


class RiemannFMEdgeSelfUpdate(nn.Module):
    """Edge self-update C via MLP on [h_i ‖ h_j ‖ π^T(log_xi(xj)) ‖ h_ij_bar]
    (spec def 14.1).

    ``h_ij_bar = ATH-Norm(h_ij^{(l-1)})`` is applied externally by the
    block; this module consumes it directly.

    Args:
        manifold: Product manifold (used for ``log_map``).
        node_dim: Node hidden dim ``d_v``.
        edge_dim: Edge hidden dim ``d_b``.
        dim_h_ambient: Lorentz ambient dim (d_h + 1); 0 disables H.
        dim_s_ambient: Sphere ambient dim (d_s + 1); 0 disables S.
        dim_e: Euclidean dim; 0 disables E.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        node_dim: int,
        edge_dim: int,
        dim_h_ambient: int,
        dim_s_ambient: int,
        dim_e: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.dim_h_ambient = dim_h_ambient
        self.dim_s_ambient = dim_s_ambient
        self.dim_e = dim_e

        # π^T LN layers mirror the node encoder's π LN layers: LN on the
        # S tangent (d_s + 1 dims, keep rotational symmetry) and LN on
        # the E tangent (d_e dims). H tangent drops the Lorentz time
        # dim (tangent freedom is d_h).
        self.ln_s = nn.LayerNorm(dim_s_ambient) if dim_s_ambient > 0 else None
        self.ln_e = nn.LayerNorm(dim_e) if dim_e > 0 else None

        drop_h = 1 if dim_h_ambient > 0 else 0
        tangent_readout_dim = (
            (dim_h_ambient - drop_h)
            + dim_s_ambient
            + dim_e
        )

        in_dim = 2 * node_dim + tangent_readout_dim + edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, edge_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim, edge_dim),
        )

    def _proj_tangent_readout(self, v: Tensor) -> Tensor:
        """π^T: drop Lorentz time tangent, LN spherical & Euclidean slices."""
        h_end = self.dim_h_ambient
        s_end = h_end + self.dim_s_ambient
        parts: list[Tensor] = []
        if self.dim_h_ambient > 0:
            parts.append(v[..., 1:h_end])  # drop tangent "time" dim
        if self.dim_s_ambient > 0:
            assert self.ln_s is not None
            parts.append(self.ln_s(v[..., h_end:s_end]))
        if self.dim_e > 0:
            assert self.ln_e is not None
            parts.append(self.ln_e(v[..., s_end:]))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        h_E_bar: Tensor,
        h_V: Tensor,
        x: Tensor,
    ) -> Tensor:
        """Compute Δh_E for residual update (spec def 14.1).

        Args:
            h_E_bar: Pre-normalized edge hidden, shape ``(B, N, N, d_b)``.
            h_V: **Current-layer** node hidden (post-A_V, per §14.2),
                shape ``(B, N, d_v)``.
            x: Manifold coordinates, shape ``(B, N, D)``.

        Returns:
            Edge residual Δh_E, shape ``(B, N, N, d_b)``.
        """
        B, N, _, _ = h_E_bar.shape

        # Pairwise log map: v[b, i, j] = log_{x_i}(x_j) in ambient coords.
        xi = x.unsqueeze(2).expand(B, N, N, -1)
        xj = x.unsqueeze(1).expand(B, N, N, -1)
        v_ij = self.manifold.log_map(xi.reshape(B * N * N, -1), xj.reshape(B * N * N, -1))
        v_ij = v_ij.reshape(B, N, N, -1)
        pi_v = self._proj_tangent_readout(v_ij)

        h_i = h_V.unsqueeze(2).expand(B, N, N, -1)
        h_j = h_V.unsqueeze(1).expand(B, N, N, -1)
        combined = torch.cat([h_i, h_j, pi_v, h_E_bar], dim=-1)

        out: Tensor = self.mlp(combined)
        return out
