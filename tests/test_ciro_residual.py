"""Unit tests for the CIRO per-pair residual construction."""

from __future__ import annotations

import torch

from ciro_rl.methods.ciro import CIROLoss


def test_residual_per_pair_mean_subtraction():
    torch.manual_seed(0)
    n, d = 16, 8
    zq = torch.randn(n, d)
    zk = torch.randn(n, d)
    loss = CIROLoss()
    R = loss.residuals(zq, zk)
    assert R.shape == (2 * n, d)
    # residual of the two views of the same anchor sum to zero per anchor
    # (tolerance is loose: (hq-m) + (hk-m) only vanishes up to fp rounding)
    rq = R[:n]
    rk = R[n:]
    assert torch.allclose(rq + rk, torch.zeros_like(rq), atol=1e-6), "r_q + r_k != 0"
    # and each residual equals view - pair mean
    m = (zq + zk) / 2.0
    assert torch.allclose(rq, zq - m, atol=1e-6)
    assert torch.allclose(rk, zk - m, atol=1e-6)


def test_residual_shape_mismatch_raises():
    zq = torch.randn(8, 4)
    zk = torch.randn(7, 4)
    loss = CIROLoss()
    try:
        loss.residuals(zq, zk)
        assert False, "expected a batch-size mismatch error"
    except RuntimeError:
        pass


def test_ciro_loss_matches_curl_when_lambda_zero():
    torch.manual_seed(1)
    z = torch.randn(20, 8)
    from ciro_rl.methods.curl import curl_contrastive_loss
    W = torch.eye(8)
    l = CIROLoss(lambda_=0.0, hsic_mode="exact")
    out = l(z, z, z, z, W, placement="encoder")
    c = curl_contrastive_loss(z, z, W)
    assert abs(float(out["total"]) - float(c)) < 1e-6