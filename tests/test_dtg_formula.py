"""Unit tests for the DTG formula and its edge cases.

Requirement: percentage-drop arithmetic + edge cases (e.g. an ID score of 0).
"""

from __future__ import annotations

import math

import pytest

from ciro_rl.diagnostics.dtg import dtg_formula


def test_dtg_normal_positive():
    # 90% -> 72% is a 20% relative drop.
    assert dtg_formula(0.90, 0.72) == pytest.approx(20.0)


def test_dtg_no_drop():
    assert dtg_formula(0.80, 0.80) == pytest.approx(0.0)


def test_dtg_eval_better_is_negative():
    # eval pool easier -> negative drop
    assert dtg_formula(0.70, 0.95) == pytest.approx(-35.7142857, abs=1e-3)


def test_dtg_id_zero_returns_nan():
    # ID score of 0 -> undefined; must be NaN, not a misleading number.
    assert dtg_formula(0.0, 0.0) != dtg_formula(0.0, 0.0)  # NaN != NaN


def test_dtg_id_negative_returns_nan():
    value = dtg_formula(-0.5, 0.4)
    assert value != value  # NaN


def test_dtg_returns_float():
    assert isinstance(dtg_formula(0.9, 0.6), float)
