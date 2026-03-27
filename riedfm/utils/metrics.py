"""Evaluation metrics for graph generation and KG tasks.

Includes:
- Graph structural metrics (MMD for degree, clustering, spectral)
- V.U.N. (Validity, Uniqueness, Novelty)
- Link prediction metrics (MRR, Hits@K)
"""

import numpy as np
import torch
from torch import Tensor


def degree_mmd(gen_graphs: list, ref_graphs: list, sigma: float = 1.0) -> float:
    """Maximum Mean Discrepancy between degree distributions.

    Args:
        gen_graphs: List of generated adjacency matrices (as numpy or torch).
        ref_graphs: List of reference adjacency matrices.
        sigma: RBF kernel bandwidth.

    Returns:
        MMD value (lower is better).
    """
    gen_dists = [_degree_histogram(g) for g in gen_graphs]
    ref_dists = [_degree_histogram(g) for g in ref_graphs]
    return _gaussian_mmd(gen_dists, ref_dists, sigma)


def clustering_mmd(gen_graphs: list, ref_graphs: list, sigma: float = 0.1) -> float:
    """MMD between clustering coefficient distributions."""
    import networkx as nx

    def get_clustering(adj):
        G = nx.from_numpy_array(np.array(adj > 0, dtype=float))
        coeffs = list(nx.clustering(G).values())
        return np.histogram(coeffs, bins=20, range=(0, 1), density=True)[0]

    gen_dists = [get_clustering(g) for g in gen_graphs]
    ref_dists = [get_clustering(g) for g in ref_graphs]
    return _gaussian_mmd(gen_dists, ref_dists, sigma)


def spectral_mmd(gen_graphs: list, ref_graphs: list, sigma: float = 1.0) -> float:
    """MMD between graph Laplacian eigenvalue distributions."""

    def get_spectrum(adj):
        adj_np = np.array(adj > 0, dtype=float)
        degree = np.diag(adj_np.sum(axis=1))
        laplacian = degree - adj_np
        eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
        return np.histogram(eigenvalues, bins=20, density=True)[0]

    gen_dists = [get_spectrum(g) for g in gen_graphs]
    ref_dists = [get_spectrum(g) for g in ref_graphs]
    return _gaussian_mmd(gen_dists, ref_dists, sigma)


def compute_vun(
    gen_graphs: list,
    ref_graphs: list,
    validity_fn=None,
) -> dict[str, float]:
    """Compute Validity, Uniqueness, Novelty metrics.

    Args:
        gen_graphs: List of generated graphs (adjacency matrices).
        ref_graphs: List of reference/training graphs.
        validity_fn: Optional function to check if a graph is valid.

    Returns:
        Dictionary with "validity", "uniqueness", "novelty" values in [0, 1].
    """
    n_gen = len(gen_graphs)
    if n_gen == 0:
        return {"validity": 0.0, "uniqueness": 0.0, "novelty": 0.0}

    # Validity
    if validity_fn is not None:
        n_valid = sum(1 for g in gen_graphs if validity_fn(g))
    else:
        # Default: connected graph with at least one edge
        n_valid = sum(1 for g in gen_graphs if np.array(g > 0).sum() > 0)
    validity = n_valid / n_gen

    # Uniqueness (via graph hash)
    hashes = set()
    for g in gen_graphs:
        h = _graph_hash(g)
        hashes.add(h)
    uniqueness = len(hashes) / n_gen

    # Novelty (not in training set)
    ref_hashes = {_graph_hash(g) for g in ref_graphs}
    n_novel = sum(1 for g in gen_graphs if _graph_hash(g) not in ref_hashes)
    novelty = n_novel / n_gen

    return {"validity": validity, "uniqueness": uniqueness, "novelty": novelty}


def mean_reciprocal_rank(ranks: Tensor) -> float:
    """Compute MRR from a tensor of ranks."""
    return (1.0 / ranks.float()).mean().item()


def hits_at_k(ranks: Tensor, k: int) -> float:
    """Compute Hits@K from a tensor of ranks."""
    return (ranks <= k).float().mean().item()


# --- Helper functions ---


def _degree_histogram(adj, num_bins: int = 50) -> np.ndarray:
    """Compute degree histogram from adjacency matrix."""
    adj_np = np.array(adj > 0, dtype=float)
    degrees = adj_np.sum(axis=1)
    hist, _ = np.histogram(degrees, bins=num_bins, density=True)
    return hist


def _gaussian_mmd(x_dists: list, y_dists: list, sigma: float) -> float:
    """Compute Gaussian MMD between two sets of distributions."""
    x = np.array(x_dists)
    y = np.array(y_dists)

    def rbf(a, b):
        diff = a[:, None] - b[None, :]
        sq_dist = (diff**2).sum(axis=-1)
        return np.exp(-sq_dist / (2 * sigma**2))

    xx = rbf(x, x).mean()
    yy = rbf(y, y).mean()
    xy = rbf(x, y).mean()
    return float(xx + yy - 2 * xy)


def _graph_hash(adj) -> str:
    """Simple hash of a graph based on sorted degree sequence."""
    adj_np = np.array(adj > 0, dtype=float)
    degrees = sorted(adj_np.sum(axis=1).tolist())
    return str(degrees)
