"""Regression test: the momentum key encoder must change SLOWLY.

The old CURL implementation computed key = m*key + (1-m)*query with m = 0.05,
i.e. it moved the key ~95% toward the query encoder on every single update,
defeating the purpose of a momentum/averaging target entirely. With the fix the
update is key = (1-m)*key + m*query, a small step toward the query each time.

This test pins the closed-form EMA value and asserts the "retains most of its
pre-update weights" property so the bug cannot silently regress.
"""

from __future__ import annotations

import torch

from ciro_rl.methods.curl import CURLModel

MOMENTUM = 0.05
N_UPDATES = 4


def test_momentum_key_retains_most_weights():
    torch.manual_seed(0)
    model = CURLModel(d_rep=16)
    # Drive the key encoder to a known value (all ones) and the query encoder to
    # a far-away value (all zeros): a correctly-averaging key must stay near its
    # own old weights instead of collapsing onto the query.
    for p in model.key.parameters():
        p.data.fill_(1.0)
    for p in model.query.parameters():
        p.data.fill_(0.0)

    for _ in range(N_UPDATES):
        model.update_momentum_key(MOMENTUM)

    expected = (1.0 - MOMENTUM) ** N_UPDATES
    for p in model.key.parameters():
        # key must track the closed-form EMA value (0.95^4 ~= 0.815).
        assert torch.allclose(p, torch.full_like(p, expected), atol=1e-5)
        # and retain most of its pre-update magnitude (0.5 is far below 0.815
        # but far above what the swapped-coefficient bug produced, 0.05^4).
        assert float(p.detach().abs().min()) > 0.5


def test_momentum_key_moves_slowly_toward_query():
    torch.manual_seed(1)
    model = CURLModel(d_rep=16)
    for p in model.key.parameters():
        p.data.fill_(1.0)
    for p in model.query.parameters():
        p.data.fill_(0.0)

    model.update_momentum_key(MOMENTUM)

    for p in model.key.parameters():
        # after a single update the key is still ~95% its old self...
        assert float(p.detach().mean()) > 0.8
        # ...and nowhere near the query encoder's value.
        assert float(p.detach().abs().max()) > 0.5
