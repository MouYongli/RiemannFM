"""Batch collation with virtual-node padding + modality masking (§3.7-3.8, §9).

Pads graphs in a batch to ``N_max`` with virtual nodes (zero edges, zero
text, ``node_mask = False``). Virtual-node manifold coordinates are a
model-level concern handled during flow matching.

Modality masking (spec §9) is applied per subgraph at batch time:

  - ``MODE_FULL``       : m_text = 1, m_coord = 1 everywhere (standard).
  - ``MODE_TEXT_MASK``  : fraction ``rho_tm`` of real nodes get m_text = 0
                          (their text is replaced by ``mask_emb`` at the
                          lightning module). m_coord stays 1.
  - ``MODE_COORD_MASK`` : fraction ``rho_cm`` of real nodes get m_coord = 0
                          (their x_t is held at the prior sample for all
                          t, and they are excluded from L_X). m_text stays 1.

Mode is sampled per-subgraph from ``mode_probs`` (default
``(0.70, 0.15, 0.15)``). The emitted ``mode_idx`` tensor lets the flow
module pick the correct per-sample ``p_t`` (Beta(5, 1) for text-mask,
default otherwise — spec §9.6).
"""

import torch

from riemannfm.data.graph import RiemannFMGraphData

MODE_FULL: int = 0
MODE_TEXT_MASK: int = 1
MODE_COORD_MASK: int = 2


class RiemannFMGraphCollator:
    """Collate a list of RiemannFMGraphData into a padded batch.

    Args:
        max_nodes: Fixed N_max padding. ``None`` uses per-batch max.
        num_edge_types: K. ``None`` infers from the first sample.
        rwpe_k: Steps for random-walk PE (``0`` disables).
        mode_probs: Probabilities over ``(full, text_mask, coord_mask)``.
            Must be non-negative and sum to 1. Default ``(1, 0, 0)``
            disables modality masking — the caller (datamodule /
            training config) is expected to pass spec defaults
            ``(0.70, 0.15, 0.15)`` when training.
        rho_tm: Fraction of real nodes whose text is masked in text-mask
            mode (spec §9.2). Default 0.30.
        rho_cm: Fraction of real nodes whose coord is masked in
            coord-mask mode. Default 0.15.
    """

    def __init__(
        self,
        max_nodes: int | None = None,
        num_edge_types: int | None = None,
        rwpe_k: int = 0,
        mode_probs: tuple[float, float, float] = (1.0, 0.0, 0.0),
        rho_tm: float = 0.30,
        rho_cm: float = 0.15,
    ):
        if len(mode_probs) != 3:
            msg = f"mode_probs must be (p_full, p_tm, p_cm), got {mode_probs!r}"
            raise ValueError(msg)
        if any(p < 0 for p in mode_probs):
            msg = f"mode_probs must be non-negative, got {mode_probs!r}"
            raise ValueError(msg)
        total = sum(mode_probs)
        if total <= 0:
            msg = f"mode_probs must sum to a positive number, got {mode_probs!r}"
            raise ValueError(msg)
        if not (0.0 <= rho_tm <= 1.0):
            msg = f"rho_tm must be in [0, 1], got {rho_tm}"
            raise ValueError(msg)
        if not (0.0 <= rho_cm <= 1.0):
            msg = f"rho_cm must be in [0, 1], got {rho_cm}"
            raise ValueError(msg)

        self.max_nodes = max_nodes
        self.num_edge_types = num_edge_types
        self.rwpe_k = rwpe_k
        self.mode_probs = torch.tensor(
            [p / total for p in mode_probs], dtype=torch.float32,
        )
        self.rho_tm = rho_tm
        self.rho_cm = rho_cm

    def __call__(self, batch: list[RiemannFMGraphData]) -> dict[str, object]:
        """Collate into padded tensors.

        Returns:
            dict with:
              - edge_types     (B, N_max, N_max, K)  float32
              - node_text      (B, N_max, d_c)       float32
              - node_mask      (B, N_max)            bool
              - node_ids       (B, N_max)            long    -1 = virtual
              - num_real_nodes (B,)                  long
              - m_text         (B, N_max)            bool    1 = keep text
              - m_coord        (B, N_max)            bool    1 = keep coord
              - mode_idx       (B,)                  long    {0, 1, 2}
              - batch_size     int
              - node_pe        (B, N_max, rwpe_k)    float32 (if rwpe_k > 0)
        """
        B = len(batch)
        N_max = self.max_nodes or max(g.total_nodes for g in batch)
        K = self.num_edge_types or batch[0].n_relations
        d_c = batch[0].node_text.shape[-1]

        edge_types = torch.zeros(B, N_max, N_max, K, dtype=torch.float32)
        node_text = torch.zeros(B, N_max, d_c, dtype=torch.float32)
        node_mask = torch.zeros(B, N_max, dtype=torch.bool)
        node_ids = torch.full((B, N_max), -1, dtype=torch.long)
        num_real = torch.zeros(B, dtype=torch.long)

        for i, g in enumerate(batch):
            n = min(g.total_nodes, N_max)
            edge_types[i, :n, :n, :K] = g.edge_types[:n, :n, :K]
            node_text[i, :n] = g.node_text[:n]
            node_mask[i, :n] = g.node_mask[:n]
            node_ids[i, :n] = g.node_ids[:n]
            num_real[i] = min(g.num_nodes, N_max)

        mode_idx = torch.multinomial(
            self.mode_probs, num_samples=B, replacement=True,
        ).to(torch.long)
        m_text, m_coord = _apply_modality_masks(
            mode_idx, num_real, N_max, self.rho_tm, self.rho_cm,
        )

        out: dict[str, object] = {
            "edge_types": edge_types,
            "node_text": node_text,
            "node_mask": node_mask,
            "node_ids": node_ids,
            "num_real_nodes": num_real,
            "m_text": m_text,
            "m_coord": m_coord,
            "mode_idx": mode_idx,
            "batch_size": B,
        }

        if self.rwpe_k > 0:
            out["node_pe"] = _compute_rwpe(edge_types, node_mask, self.rwpe_k)

        return out


def _apply_modality_masks(
    mode_idx: torch.Tensor,
    num_real: torch.Tensor,
    N_max: int,
    rho_tm: float,
    rho_cm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-subgraph modality masks (spec §9.3).

    Virtual-node positions keep the default of 1 so they don't
    accidentally look like masked content downstream; the real gating
    against virtual nodes uses ``node_mask``.
    """
    B = mode_idx.shape[0]
    m_text = torch.ones(B, N_max, dtype=torch.bool)
    m_coord = torch.ones(B, N_max, dtype=torch.bool)

    for i in range(B):
        nr = int(num_real[i].item())
        mode = int(mode_idx[i].item())

        if mode == MODE_TEXT_MASK and nr > 0:
            k = max(1, round(rho_tm * nr))
            k = min(k, nr)
            idx = torch.randperm(nr)[:k]
            m_text[i, idx] = False
        elif mode == MODE_COORD_MASK and nr > 0:
            k = max(1, round(rho_cm * nr))
            k = min(k, nr)
            idx = torch.randperm(nr)[:k]
            m_coord[i, idx] = False

    return m_text, m_coord


def _compute_rwpe(
    edge_types: torch.Tensor,
    node_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Random-walk positional encoding (spec §10.4)."""
    B, N, _, _ = edge_types.shape
    A = (edge_types.sum(dim=-1) > 0).float()
    A = A * node_mask.unsqueeze(1).float() * node_mask.unsqueeze(2).float()

    deg = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
    P = A / deg

    pe = torch.zeros(B, N, k, dtype=torch.float32)
    P_power = P.clone()
    for step in range(k):
        pe[..., step] = torch.diagonal(P_power, dim1=-2, dim2=-1)
        if step < k - 1:
            P_power = P_power @ P
    pe = pe * node_mask.unsqueeze(-1).float()
    return pe
