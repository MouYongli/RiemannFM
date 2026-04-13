"""Joint continuous-discrete flow matching (Algorithm 1).

Combines geodesic interpolation on the product manifold with
discrete edge-type interpolation via shared Bernoulli masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from riemannfm.flow.continuous_flow import (
    geodesic_interpolation,
    sample_time,
    vector_field_target,
)
from riemannfm.flow.discrete_flow import discrete_interpolation
from riemannfm.flow.noise import sample_continuous_noise, sample_discrete_noise

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


@dataclass(slots=True)
class FlowMatchingSample:
    """Output of the joint flow matching interpolation.

    Contains all tensors needed for the training step:
    model input (x_t, E_t, t) and targets (u_t, E_1).

    Attributes:
        x_t: Interpolated manifold coordinates, shape ``(B, N, D)``.
        u_t: Vector field target (tangent at x_t), shape ``(B, N, D)``.
        E_t: Interpolated edge types, shape ``(B, N, N, K)``.
        E_1: Target edge types (data), shape ``(B, N, N, K)``.
        t: Time steps.  Shape ``(B,)`` when no per-node override is
            supplied, or ``(B, N)`` when the collator assigned per-node
            mask labels (M_x=0 / M_c=1) through ``t_node_override``.
        node_mask: Bool mask, shape ``(B, N)``.
    """

    x_t: Tensor
    u_t: Tensor
    E_t: Tensor
    E_1: Tensor
    t: Tensor
    node_mask: Tensor


class RiemannFMJointFlow:
    """Joint continuous-discrete flow matching.

    Orchestrates noise sampling, time sampling, interpolation, and
    vector field target computation for training.

    Supports ``disable_continuous`` and ``disable_discrete`` flags from
    the flow config for ablation experiments.

    Args:
        manifold: Product manifold M = H x S x E.
        avg_edge_density: Average edge density rho for Bernoulli noise.
        t_max: Maximum time clamp for VF target singularity.
        disable_continuous: If True, skip continuous flow (x_t = x_1).
        disable_discrete: If True, skip discrete flow (E_t = E_1).
        radius_h: Geodesic radius for hyperbolic noise.
        sigma_e: Std for Euclidean noise.
        time_distribution: ``"uniform"`` or ``"logit_normal"``.
        logit_normal_mu: Mean of normal in logit space (logit_normal only).
        logit_normal_sigma: Std of normal in logit space (logit_normal only).
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        avg_edge_density: float = 0.05,
        rho_k: Tensor | None = None,
        t_max: float = 0.999,
        disable_continuous: bool = False,
        disable_discrete: bool = False,
        radius_h: float = 5.0,
        sigma_e: float = 1.0,
        time_distribution: str = "uniform",
        logit_normal_mu: float = 0.0,
        logit_normal_sigma: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.manifold = manifold
        self.avg_edge_density = avg_edge_density
        self.rho_k = rho_k
        self.t_max = t_max
        self.disable_continuous = disable_continuous
        self.disable_discrete = disable_discrete
        self.radius_h = radius_h
        self.sigma_e = sigma_e
        self.time_distribution = time_distribution
        self.logit_normal_mu = logit_normal_mu
        self.logit_normal_sigma = logit_normal_sigma

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        manifold: RiemannFMProductManifold,
    ) -> RiemannFMJointFlow:
        """Construct from Hydra config."""
        return cls(
            manifold=manifold,
            avg_edge_density=getattr(cfg, "avg_edge_density", 0.05),
            t_max=getattr(cfg, "t_max", 0.999),
            disable_continuous=getattr(cfg, "disable_continuous", False),
            disable_discrete=getattr(cfg, "disable_discrete", False),
            time_distribution=getattr(cfg, "time_distribution", "uniform"),
            logit_normal_mu=getattr(cfg, "logit_normal_mu", 0.0),
            logit_normal_sigma=getattr(cfg, "logit_normal_sigma", 1.0),
        )

    def sample(
        self,
        x_1: Tensor,
        E_1: Tensor,
        node_mask: Tensor,
        generator: torch.Generator | None = None,
        t_node_override: Tensor | None = None,
    ) -> FlowMatchingSample:
        """Generate a flow matching training sample.

        Given data points (x_1, E_1), samples noise, time, and computes
        the interpolated state and vector field target.

        When ``t_node_override`` is provided (from the collator's masking
        pipeline), per-node time labels replace the batch-level scalar for
        continuous interpolation: M_x positions are forced to ``t=0`` and
        M_c positions to ``t=1``.  Discrete (edge) flow continues to use
        the batch-level scalar because edges span node pairs.

        Args:
            x_1: Data manifold coordinates, shape ``(B, N, D)``.
            E_1: Data edge types, shape ``(B, N, N, K)``.
            node_mask: Bool mask, shape ``(B, N)``.
            generator: Optional RNG.
            t_node_override: Optional per-node time, shape ``(B, N)``.
                ``NaN`` entries are filled with the sampled batch-``t``.

        Returns:
            FlowMatchingSample with all training tensors.
        """
        B = x_1.shape[0]
        device = x_1.device
        dtype = x_1.dtype

        # Sample batch-level time (used for discrete flow and as the
        # fill value for per-node NaN sentinels).
        t_batch = sample_time(
            B, device=device, dtype=dtype, generator=generator,
            distribution=self.time_distribution,
            logit_normal_mu=self.logit_normal_mu,
            logit_normal_sigma=self.logit_normal_sigma,
        )

        # Construct the continuous-flow time tensor.  Per-node when the
        # collator supplied mask labels; otherwise the batch scalar.
        if t_node_override is not None:
            fill = t_batch.unsqueeze(-1).expand_as(t_node_override)
            t_cont = torch.where(
                torch.isnan(t_node_override), fill, t_node_override.to(dtype),
            )
        else:
            t_cont = t_batch

        # Continuous flow.
        if self.disable_continuous:
            x_t = x_1.clone()
            u_t = torch.zeros_like(x_1)
        else:
            x_0 = sample_continuous_noise(
                self.manifold, *x_1.shape[:-1],
                device=device, dtype=dtype, generator=generator,
                radius_h=self.radius_h, sigma_e=self.sigma_e,
            )
            x_t = geodesic_interpolation(self.manifold, x_0, x_1, t_cont)
            u_t = vector_field_target(
                self.manifold, x_t, x_1, t_cont, t_max=self.t_max,
            )

        # Discrete flow uses the batch-level scalar — edges are per-pair,
        # not per-node, so per-node masking does not apply to them.
        if self.disable_discrete:
            E_t = E_1.clone()
        else:
            E_0 = sample_discrete_noise(
                E_1, avg_edge_density=self.avg_edge_density,
                rho_k=self.rho_k, generator=generator,
            )
            E_t = discrete_interpolation(E_0, E_1, t_batch, generator=generator)

        return FlowMatchingSample(
            x_t=x_t,
            u_t=u_t,
            E_t=E_t,
            E_1=E_1,
            t=t_cont,
            node_mask=node_mask,
        )
