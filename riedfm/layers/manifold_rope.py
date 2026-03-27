"""Manifold Rotary Position Encoding (M-RoPE).

Extends standard RoPE to manifold-valued node positions.
Instead of using sequential position indices, uses pairwise geodesic
distances on the product manifold to define rotation angles:

    theta_ij^(k) = omega_k * d_M(x_i, x_j)

The rotation only depends on the relative geometric relationship between
nodes, automatically preserving permutation equivariance.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from riedfm.manifolds.product import ProductManifold


class ManifoldRoPE(nn.Module):
    """Manifold-aware Rotary Position Encoding.

    For each pair of nodes (i, j), computes rotation angles based on
    their geodesic distance on the product manifold. These rotations are
    applied to query/key vectors before computing attention.

    Args:
        head_dim: Dimension per attention head (must be even).
        num_freq: Number of frequency bands (default: head_dim // 2).
        max_dist: Maximum distance for frequency normalization.
    """

    def __init__(
        self,
        head_dim: int,
        num_freq: int | None = None,
        max_dist: float = 10.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_freq = num_freq or head_dim // 2
        self.max_dist = max_dist

        # Learnable frequency parameters (geometric series)
        # Initialize similar to standard RoPE
        freqs = 1.0 / (10000 ** (torch.arange(0, self.num_freq).float() / self.num_freq))
        self.log_freqs = nn.Parameter(freqs.log())

    @property
    def freqs(self) -> Tensor:
        return self.log_freqs.exp()

    def compute_rotation_angles(
        self,
        dist_matrix: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute cos/sin rotation matrices from pairwise distances.

        Args:
            dist_matrix: Pairwise geodesic distances, shape (N, N).

        Returns:
            (cos_angles, sin_angles), each shape (N, N, num_freq).
        """
        # dist_matrix: (N, N) -> (N, N, 1) * (num_freq,) -> (N, N, num_freq)
        angles = dist_matrix.unsqueeze(-1) * self.freqs.to(dist_matrix.device)
        return torch.cos(angles), torch.sin(angles)

    def apply_rotary(
        self,
        q: Tensor,
        k: Tensor,
        cos_angles: Tensor,
        sin_angles: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply rotary position encoding to query and key tensors.

        Uses the pairwise rotation approach: the rotation for attention(i, j)
        depends on the distance between nodes i and j.

        For efficiency, we apply the rotation as:
            q_rot_i = q_i  (queries unchanged)
            k_rot_ij = R(theta_ij) * k_j  (keys rotated per-pair)

        But for computational tractability with N^2 pairs, we approximate:
        each node gets a rotation based on its distance to the origin
        (or mean position), and the relative rotation emerges from the
        difference. This is the standard RoPE trick generalized to manifolds.

        Args:
            q: Queries, shape (N, num_heads, head_dim).
            k: Keys, shape (N, num_heads, head_dim).
            cos_angles: Cosine of rotation angles, shape (N, N, num_freq).
            sin_angles: Sine of rotation angles, shape (N, N, num_freq).

        Returns:
            (q_rotated, k_rotated), same shapes as input.
        """
        # Split into pairs for rotation: (even, odd) indices
        d2 = self.head_dim // 2
        q1, q2 = q[..., :d2], q[..., d2 : 2 * d2]
        k1, k2 = k[..., :d2], k[..., d2 : 2 * d2]

        # For the per-node approximation, use the mean distance as position
        # cos_node, sin_node: (N, num_freq) from diagonal or row-mean
        cos_node = cos_angles.diagonal(dim1=0, dim2=1).T  # (N, num_freq)
        sin_node = sin_angles.mean(dim=1)  # (N, num_freq) approximate

        # Pad or truncate to match d2
        if cos_node.shape[-1] < d2:
            pad = d2 - cos_node.shape[-1]
            cos_node = torch.cat([cos_node, torch.ones(*cos_node.shape[:-1], pad, device=cos_node.device)], dim=-1)
            sin_node = torch.cat([sin_node, torch.zeros(*sin_node.shape[:-1], pad, device=sin_node.device)], dim=-1)
        elif cos_node.shape[-1] > d2:
            cos_node = cos_node[..., :d2]
            sin_node = sin_node[..., :d2]

        # Expand for num_heads: (N, 1, d2)
        cos_n = cos_node.unsqueeze(1)
        sin_n = sin_node.unsqueeze(1)

        # Apply rotation
        q_rot = torch.cat([q1 * cos_n - q2 * sin_n, q1 * sin_n + q2 * cos_n], dim=-1)
        k_rot = torch.cat([k1 * cos_n - k2 * sin_n, k1 * sin_n + k2 * cos_n], dim=-1)

        # If head_dim is odd, keep the last element unchanged
        if self.head_dim % 2 == 1:
            q_rot = torch.cat([q_rot, q[..., -1:]], dim=-1)
            k_rot = torch.cat([k_rot, k[..., -1:]], dim=-1)

        return q_rot, k_rot

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        manifold: ProductManifold,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute and apply M-RoPE.

        Args:
            q: Queries, shape (N, num_heads, head_dim).
            k: Keys, shape (N, num_heads, head_dim).
            manifold: Product manifold for distance computation.
            positions: Node positions on manifold, shape (N, total_dim).

        Returns:
            (q_rotated, k_rotated).
        """
        # Compute pairwise distances
        N = positions.shape[0]
        # dist_matrix: (N, N)
        pos_i = positions.unsqueeze(1).expand(-1, N, -1)
        pos_j = positions.unsqueeze(0).expand(N, -1, -1)
        dist_matrix = manifold.dist(pos_i, pos_j)

        cos_angles, sin_angles = self.compute_rotation_angles(dist_matrix)
        return self.apply_rotary(q, k, cos_angles, sin_angles)
