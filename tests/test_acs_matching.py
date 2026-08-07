"""Unit tests for the ACS Hungarian-matching logic.

Requirement: on a batch with a known ground-truth causal-latent-to-coordinate
mapping, the matching must recover it.
"""

from __future__ import annotations

import numpy as np

from ciro_rl.diagnostics.acs import (hungarian_match, concentration,
                                     average_causal_sensitivity)


def test_hungarian_recovers_known_mapping():
    rng = np.random.RandomState(3)
    n_c, d = 4, 6
    perm = [2, 0, 5, 3]                      # ground-truth causal->coord mapping
    S = np.zeros((n_c, d))
    for i, c in enumerate(perm):
        S[i, c] = 1.0 + 0.1 * rng.rand()
        S[i] += 0.05 * rng.rand(d)           # small noise on other coords
    mapping = hungarian_match(S)
    assert list(mapping) == perm, f"mapping {mapping} != {perm}"


def test_concentration_isolated_coordinates():
    S = np.eye(3)
    mapping = hungarian_match(S)
    conc = concentration(S, mapping)
    assert np.allclose(conc, 1.0, atol=1e-6)


def test_concentration_edge_no_positive_rows():
    S = np.zeros((2, 4))
    mapping = hungarian_match(S)
    conc = concentration(S, mapping)
    assert np.allclose(conc, 0.0, atol=1e-8)


def test_acs_returns_expected_keys():
    from ciro_rl.envs.scm_mdp import get_scm_mdp
    import torch

    class LinEnc(torch.nn.Module):
        def forward(self, x):
            b = x.shape[0]
            torch.manual_seed(max(x.sum().abs().item() * 0 + b, 0) or 1)
            return torch.randn(b, 8)

    scm = get_scm_mdp("independent", latent_dim=8, seed=7)
    enc = LinEnc()
    res = average_causal_sensitivity(scm, enc, n_samples=8, h=0, device="cpu")
    assert {"acs", "sensitivity_matrix", "matching", "concentration"} <= set(res)
