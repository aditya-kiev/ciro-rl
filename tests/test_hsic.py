"""Unit tests for the HSIC estimators.

Requirement (repo spec): the RFF approximation must agree with the exact
V-statistic estimator on a small synthetic batch to within a stated tolerance.
"""

from __future__ import annotations

import torch

from ciro_rl.utils.hsic import (pairwise_hsic_exact, pairwise_hsic_rff,
                                centering_matrix, biased_hsic_vstat)


def test_exact_and_rff_agree_within_tolerance():
    torch.manual_seed(0)
    R = torch.randn(48, 6)
    exact = pairwise_hsic_exact(R)
    rff = pairwise_hsic_rff(R, n_features=128)
    # RFF is an unbiased estimator: for 128 features this typically agrees to a
    # few percent. We assert a generous 30% relative bound (robust, low flake).
    scale = max(float(exact), 1e-6)
    assert abs(float(exact) - float(rff)) <= 0.30 * scale, \
        f"exact={exact:.5f} rff={rff:.5f} disagree"


def test_hsic_independent_coordinates_is_smaller():
    torch.manual_seed(1)
    n = 64
    # Independent Gaussian columns
    ind = torch.randn(n, 4)
    # A clearly-dependent coordinate (x1 duplicated + noise)
    dep = ind.clone()
    dep[:, 1] = ind[:, 0] + 0.05 * torch.randn(n)
    h_ind = pairwise_hsic_exact(ind).item()
    h_dep = pairwise_hsic_exact(dep).item()
    assert h_dep > h_ind, f"dep={h_dep} should exceed ind={h_ind}"


def test_centering_matrix():
    n = 5
    H = centering_matrix(n, torch.device("cpu"))
    assert H.shape == (n, n)
    assert torch.allclose(H @ H, H, atol=1e-6)  # idempotent
    assert torch.allclose(H.sum(0), torch.zeros(n), atol=1e-6)


def test_biased_vstat_sanity():
    torch.manual_seed(2)
    K = torch.randn(8, 8)
    L = torch.randn(8, 8)
    v = biased_hsic_vstat(K, L)
    assert v.dim() == 0
    assert torch.isfinite(v)
