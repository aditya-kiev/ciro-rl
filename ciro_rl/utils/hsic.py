"""HSIC estimators (RFF approximation and exact V-statistic).

Migrated wholesale from causal-cir's `losses/hsic.py` (same RFF-vs-exact dual
path) and renamed to match the CIRO paper notation. Two public API layers:

  * per-store helpers : pairwise-coordinate HSIC over a single (n, d) matrix,
    used by the CIRO / marginal-HSIC penalties;
  * cross-store helpers: cross-variable HSIC between two matrices (for
    inspection / debugging).

References to the CIRO paper math:
  * ``pairwise_hsic_rff``   -> CIRO Appendix A (RFF approximation, D features)
  * ``pairwise_hsic_exact`` -> CIRO Appendix A.2 (exact biased V-statistic)

All estimators are biased (V-statistics), consistent with the paper's
formulation of HSIC as 1/n^2 tr(K H L H).
"""

from __future__ import annotations

from typing import List, Optional

import torch

__all__ = [
    "centering_matrix",
    "biased_hsic_vstat",
    "rbf_kernel_1d",
    "median_heuristic",
    "pairwise_hsic_exact",
    "pairwise_hsic_rff",
    "cross_pairwise_hsic_exact",
    "cross_pairwise_hsic_rff",
]


def upper_triu_mask(m: torch.Tensor) -> torch.Tensor:
    """Bool mask for the upper triangle (j > i) of a square matrix."""
    return torch.triu(torch.ones_like(m, dtype=torch.bool), diagonal=1)


def median_heuristic(x: torch.Tensor) -> float:
    """Median pairwise L2 distance of a vector -> RBF bandwidth.

    If an explicit bandwidth is None the kernels use this value per coordinate,
    matching the paper's median-heuristic default.
    """
    d = torch.cdist(x.reshape(-1, 1), x.reshape(-1, 1))
    vals = d[upper_triu_mask(d)]
    return float(vals.detach().median().clamp(min=1e-6))


def centering_matrix(n: int, device: torch.device) -> torch.Tensor:
    """H = I_n - (1/n) 1 1^T (the centering matrix, Algorithm 1 line 6)."""
    return torch.eye(n, device=device) - (1.0 / n) * torch.ones(n, n, device=device)


def biased_hsic_vstat(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Biased HSIC V-statistic 1/n^2 tr(K H L H) (CIRO Appendix A.2)."""
    n = K.shape[0]
    H = centering_matrix(n, K.device)
    return torch.trace(K @ H @ L @ H) / (n * n)


def rbf_kernel_1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """RBF kernel matrix (n, n) for the 1-D column x, bandwidth sigma."""
    sq = (x.unsqueeze(0) - x.unsqueeze(1)) ** 2
    return torch.exp(-sq / (2.0 * sigma * sigma))


def _column_sigmas(R: torch.Tensor, sigma: Optional[float]) -> List[float]:
    """Per-column bandwidths: constant ``sigma`` or a median heuristic each."""
    d = R.shape[1]
    if sigma is not None:
        sig = float(sigma)
        return [sig] * d
    return [median_heuristic(R[:, j]) for j in range(d)]


def pairwise_hsic_exact(R: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    """Sum of per-coordinate-pair exact HSIC over the (N, d) matrix R.

    HSIC_exact = sum_{i,j} 1/n^2 tr(K_i H K_j H), with K_i the RBF gram of
    coordinate i (sigma_j the median heuristic of column j unless given).
    """
    R = R.float()
    n, d = R.shape
    sigmas = _column_sigmas(R, sigma)
    kernels = [rbf_kernel_1d(R[:, j], sigmas[j]) for j in range(d)]
    total = torch.zeros((), dtype=R.dtype)
    for i in range(d):
        for k in range(i, d):
            total = total + biased_hsic_vstat(kernels[i], kernels[k])
    return total


def pairwise_hsic_rff(R: torch.Tensor, sigma: Optional[float] = None,
                      n_features: int = 128, seed: int = 0) -> torch.Tensor:
    """RFF-approximated pairwise HSIC for (N, d) R (training-time default).

    Each coordinate is mapped through D random Fourier features
    phi_j = sqrt(2/m) cos( x_j/sigma_j . w + b ), and the empirical kernel
    K_j = phi_j phi_j^T / m approximates the coordinate's RBF gram, where m is
    an internal Monte-Carlo multiplicity (reduces estimator variance for the
    same nominal ``n_features``).
    """
    R = R.float()
    n, d = R.shape
    sigmas = _column_sigmas(R, sigma)
    g = torch.Generator().manual_seed(seed)
    m_total = int(n_features) * 4  # internal averaging multiplicity
    phis = []
    for j in range(d):
        w = torch.randn(m_total, generator=g)
        b = torch.rand(m_total, generator=g) * (2.0 * torch.pi)
        arr = (R[:, j].reshape(-1, 1) / sigmas[j]) * w.unsqueeze(0) + b.unsqueeze(0)
        phis.append(torch.cos(arr) * (2.0 / m_total) ** 0.5)  # (n, m)
    total = torch.zeros((), dtype=R.dtype)
    for i in range(d):
        Ki = phis[i] @ phis[i].t()
        for k in range(i, d):
            Kk = phis[k] @ phis[k].t()
            total = total + biased_hsic_vstat(Ki, Kk)
    return total


def cross_pairwise_hsic_exact(X: torch.Tensor, Y: torch.Tensor,
                              sigma: Optional[float] = None) -> torch.Tensor:
    """Cross-variable HSIC between representation X (n, dx) and target Y (n, dy).

    Sum over coordinates of 1/n^2 tr(K_i^X H K_j^Y H); used for inspecting
    dependence between two feature sets (e.g. encoder vs decoder latents).
    """
    X, Y = X.float(), Y.float()
    sx = _column_sigmas(X, sigma)
    sy = _column_sigmas(Y, sigma)
    KX = [rbf_kernel_1d(X[:, j], sx[j]) for j in range(X.shape[1])]
    KY = [rbf_kernel_1d(Y[:, j], sy[j]) for j in range(Y.shape[1])]
    total = torch.zeros((), dtype=X.dtype)
    for kx in KX:
        for ky in KY:
            total = total + biased_hsic_vstat(kx, ky)
    return total


def cross_pairwise_hsic_rff(X: torch.Tensor, Y: torch.Tensor,
                            sigma: Optional[float] = None,
                            n_features: int = 128, seed: int = 0) -> torch.Tensor:
    """RFF-approximated cross-coordinate HSIC between X (n, dx) and Y (n, dy)."""
    X, Y = X.float(), Y.float()
    g = torch.Generator().manual_seed(seed)
    m_total = int(n_features) * 4
    sx = _column_sigmas(X, sigma)
    sy = _column_sigmas(Y, sigma)

    def _phi(M: torch.Tensor, sigmas: List[float]) -> List[torch.Tensor]:
        out = []
        for j in range(M.shape[1]):
            w = torch.randn(m_total, generator=g)
            b = torch.rand(m_total, generator=g) * (2.0 * torch.pi)
            arr = (M[:, j].reshape(-1, 1) / sigmas[j]) * w.unsqueeze(0) + b.unsqueeze(0)
            out.append(torch.cos(arr) * (2.0 / m_total) ** 0.5)
        return out

    phix, phiy = _phi(X, sx), _phi(Y, sy)
    total = torch.zeros((), dtype=X.dtype)
    for px in phix:
        Kx = px @ px.T
        for py in phiy:
            Ky = py @ py.T
            total = total + biased_hsic_vstat(Kx, Ky)
    return total