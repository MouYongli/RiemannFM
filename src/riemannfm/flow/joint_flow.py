"""Joint continuous-discrete flow matching (spec §6, §9).

Combines geodesic interpolation on the product manifold with masked
discrete-flow edge interpolation. Both streams share the batch-level
time ``t`` (spec §6.2, §6.7).

Modality masking (spec §9) influences two things here:

  - ``mode_idx``: selects the per-subgraph ``p_t`` distribution. Text-mask
    mode uses ``Beta(beta_a_text_mask, beta_b_text_mask)`` (default 5/1,
    spec §9.6) so the text-only self-sufficiency check lands near t = 1
    where the geometry is nearly clean.
  - ``m_coord``: nodes with m_coord = 0 are held at the prior sample
    (``x_t = x_0`` for all t, spec §9.3) and carry a zero target
    vector field — they are gated out of L_X by the loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from riemannfm.data.collator import MODE_TEXT_MASK
from riemannfm.flow.continuous_flow import (
    geodesic_interpolation,
    sample_time,
    vector_field_target,
)
from riemannfm.flow.discrete_flow import discrete_interpolation
from riemannfm.flow.noise import sample_continuous_noise

if TYPE_CHECKING:
    from riemannfm.manifolds.product import RiemannFMProductManifold


@dataclass(slots=True)
class FlowMatchingSample:
    """Output of joint flow matching interpolation.

    Attributes:
        x_t: Interpolated manifold coordinates, shape ``(B, N, D)``.
        u_t: Target vector field at ``x_t``, shape ``(B, N, D)``.
        E_t: Edge value tensor (data at unmasked positions, zeros at
            masked), shape ``(B, N, N, K)``.
        mu_t: Mask indicator (1 = masked, 0 = decoded),
            shape ``(B, N, N)``.
        E_1: Target edge types (data), shape ``(B, N, N, K)``.
        t: Batch-level time, shape ``(B,)``.
        node_mask: Real-vs-virtual bool mask, shape ``(B, N)``.
    """

    x_t: Tensor
    u_t: Tensor
    E_t: Tensor
    mu_t: Tensor
    E_1: Tensor
    t: Tensor
    node_mask: Tensor


class RiemannFMJointFlow:
    """Joint continuous-discrete flow matching (spec §6, §7, §8, §9).

    Args:
        manifold: Product manifold M = H × S × E.
        t_max: Max clamp for 1/(1-t) singularity in the vector field.
        disable_continuous: Ablation — skip continuous flow (x_t = x_1).
        disable_discrete: Ablation — skip discrete flow (E_t = E_1,
            mu_t = 0).
        radius_h: Hyperbolic noise radius.
        sigma_e: Euclidean noise std.
        time_distribution: Base ``p_t``: ``"uniform"``, ``"logit_normal"``,
            or ``"beta"`` (spec §7.5).
        logit_normal_mu / logit_normal_sigma: Logit-Normal params.
        beta_a / beta_b: Beta params for the base distribution.
        beta_a_text_mask / beta_b_text_mask: Beta params used in the
            text-mask mode only (spec §9.6 default ``Beta(5, 1)``).
        edge_schedule: α(t) schedule kind (spec §8.2), cosine by default.
    """

    def __init__(
        self,
        manifold: RiemannFMProductManifold,
        t_max: float = 0.999,
        disable_continuous: bool = False,
        disable_discrete: bool = False,
        radius_h: float = 5.0,
        sigma_e: float = 1.0,
        time_distribution: str = "logit_normal",
        logit_normal_mu: float = 0.0,
        logit_normal_sigma: float = 1.0,
        beta_a: float = 1.0,
        beta_b: float = 1.0,
        beta_a_text_mask: float = 5.0,
        beta_b_text_mask: float = 1.0,
        edge_schedule: str = "cosine",
        **kwargs: Any,
    ) -> None:
        self.manifold = manifold
        self.t_max = t_max
        self.disable_continuous = disable_continuous
        self.disable_discrete = disable_discrete
        self.radius_h = radius_h
        self.sigma_e = sigma_e
        self.time_distribution = time_distribution
        self.logit_normal_mu = logit_normal_mu
        self.logit_normal_sigma = logit_normal_sigma
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.beta_a_text_mask = beta_a_text_mask
        self.beta_b_text_mask = beta_b_text_mask
        self.edge_schedule = edge_schedule

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        manifold: RiemannFMProductManifold,
    ) -> RiemannFMJointFlow:
        """Construct from Hydra config."""
        return cls(
            manifold=manifold,
            t_max=getattr(cfg, "t_max", 0.999),
            disable_continuous=getattr(cfg, "disable_continuous", False),
            disable_discrete=getattr(cfg, "disable_discrete", False),
            time_distribution=getattr(cfg, "time_distribution", "logit_normal"),
            logit_normal_mu=getattr(cfg, "logit_normal_mu", 0.0),
            logit_normal_sigma=getattr(cfg, "logit_normal_sigma", 1.0),
            beta_a=getattr(cfg, "beta_a", 1.0),
            beta_b=getattr(cfg, "beta_b", 1.0),
            beta_a_text_mask=getattr(cfg, "beta_a_text_mask", 5.0),
            beta_b_text_mask=getattr(cfg, "beta_b_text_mask", 1.0),
            edge_schedule=getattr(cfg, "edge_schedule", "cosine"),
        )

    def _sample_time_per_mode(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
        mode_idx: Tensor | None,
    ) -> Tensor:
        """Per-sample ``t`` routed by ``mode_idx`` (spec §9.6).

        Text-mask subgraphs draw ``t ~ Beta(beta_a_text_mask,
        beta_b_text_mask)``; everything else uses the configured base
        ``time_distribution``. The two tensors are sampled in parallel
        and selected pointwise so the result is a clean per-sample ``t``.
        """
        t_base = sample_time(
            batch_size, device=device, dtype=dtype, generator=generator,
            distribution=self.time_distribution,
            logit_normal_mu=self.logit_normal_mu,
            logit_normal_sigma=self.logit_normal_sigma,
            beta_a=self.beta_a,
            beta_b=self.beta_b,
        )
        if mode_idx is None or not torch.any(mode_idx == MODE_TEXT_MASK):
            return t_base
        t_text_mask = sample_time(
            batch_size, device=device, dtype=dtype, generator=generator,
            distribution="beta",
            beta_a=self.beta_a_text_mask,
            beta_b=self.beta_b_text_mask,
        )
        return torch.where(mode_idx == MODE_TEXT_MASK, t_text_mask, t_base)

    def sample(
        self,
        x_1: Tensor,
        E_1: Tensor,
        node_mask: Tensor,
        m_coord: Tensor | None = None,
        mode_idx: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> FlowMatchingSample:
        """Generate a flow matching training sample (spec §22.1).

        Args:
            x_1: Data manifold coordinates, shape ``(B, N, D)``.
            E_1: Data edge types, shape ``(B, N, N, K)``.
            node_mask: Real-vs-virtual bool mask, shape ``(B, N)``.
            m_coord: Modality-mask coord gate, shape ``(B, N)`` (1 = use
                data coord, 0 = hold at prior sample). ``None`` means
                every real node is unmasked.
            mode_idx: Per-subgraph modality mode, shape ``(B,)``. Used to
                pick the correct ``p_t`` distribution. ``None`` falls
                back to the base distribution.
            generator: Optional RNG.
        """
        B = x_1.shape[0]
        device = x_1.device
        dtype = x_1.dtype

        t = self._sample_time_per_mode(B, device, dtype, generator, mode_idx)

        if self.disable_continuous:
            x_t = x_1.clone()
            u_t = torch.zeros_like(x_1)
        else:
            x_0 = sample_continuous_noise(
                self.manifold, *x_1.shape[:-1],
                device=device, dtype=dtype, generator=generator,
                radius_h=self.radius_h, sigma_e=self.sigma_e,
            )
            x_t = geodesic_interpolation(self.manifold, x_0, x_1, t)
            u_t = vector_field_target(
                self.manifold, x_t, x_1, t, t_max=self.t_max,
            )
            if m_coord is not None:
                # Spec §9.3: coord-masked nodes stay at the prior sample
                # independent of t, and carry no flow-matching signal
                # (L_X gates them out too).
                keep = m_coord.to(dtype=x_t.dtype).unsqueeze(-1)  # (B, N, 1)
                x_t = keep * x_t + (1.0 - keep) * x_0
                u_t = keep * u_t

        if self.disable_discrete:
            E_t = E_1.clone()
            mu_t = torch.zeros(
                *E_1.shape[:-1], device=device, dtype=dtype,
            )
        else:
            E_t, mu_t = discrete_interpolation(
                E_1, t, schedule=self.edge_schedule, generator=generator,
            )

        return FlowMatchingSample(
            x_t=x_t,
            u_t=u_t,
            E_t=E_t,
            mu_t=mu_t,
            E_1=E_1,
            t=t,
            node_mask=node_mask,
        )
