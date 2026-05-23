"""portfolio_eval.py 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from portfolio_eval import (
    PerformanceMetrics,
    _max_drawdown,
    compare,
    evaluate,
    render_report,
)


def _make_rets(rng_seed: int = 0, T: int = 252, N: int = 5,
               mean: float = 0.0005, std: float = 0.01) -> np.ndarray:
    rng = np.random.RandomState(rng_seed)
    return rng.normal(mean, std, (T, N))


# --- evaluate ----------------------------------------------------------------

def test_evaluate_returns_metrics_object():
    rets = _make_rets()
    w = np.full(5, 0.2)
    m = evaluate("equal", rets, w)
    assert isinstance(m, PerformanceMetrics)
    assert m.name == "equal"
    assert m.n_periods == 252


def test_evaluate_weights_shape_must_match():
    rets = _make_rets(N=5)
    with pytest.raises(ValueError):
        evaluate("bad", rets, np.full(3, 1 / 3))


def test_evaluate_rejects_1d_returns():
    with pytest.raises(ValueError):
        evaluate("bad", np.array([0.01, 0.02]), np.array([1.0]))


def test_evaluate_metrics_are_finite():
    rets = _make_rets()
    w = np.full(5, 0.2)
    m = evaluate("equal", rets, w)
    for v in [m.cum_return, m.annualized_return, m.annualized_volatility,
              m.sharpe, m.sortino, m.max_drawdown, m.calmar]:
        assert np.isfinite(v)


def test_evaluate_positive_drift_positive_sharpe():
    """均值正的资产 + 等权 → Sharpe 应 > 0。"""
    rets = _make_rets(mean=0.002, std=0.01)
    w = np.full(5, 0.2)
    m = evaluate("equal", rets, w)
    assert m.sharpe > 0
    assert m.annualized_return > 0


def test_evaluate_negative_drift_negative_sharpe():
    rets = _make_rets(mean=-0.002, std=0.01)
    w = np.full(5, 0.2)
    m = evaluate("equal", rets, w)
    assert m.sharpe < 0


def test_evaluate_zero_volatility():
    """常数收益率（std≈0）应该不崩 sharpe=0。"""
    rets = np.full((100, 3), 0.001)
    w = np.array([0.5, 0.3, 0.2])
    m = evaluate("const", rets, w)
    assert m.annualized_volatility < 1e-10
    assert m.sharpe == 0


def test_evaluate_max_drawdown_is_negative_or_zero():
    rets = _make_rets()
    w = np.full(5, 0.2)
    m = evaluate("equal", rets, w)
    assert m.max_drawdown <= 0


def test_evaluate_to_dict_is_json_serializable():
    import json
    m = evaluate("x", _make_rets(), np.full(5, 0.2))
    d = m.to_dict()
    s = json.dumps(d)
    assert "x" in s
    assert "sharpe" in d


# --- _max_drawdown ------------------------------------------------------------

def test_max_drawdown_monotone_up_is_zero():
    eq = np.array([1.0, 1.1, 1.2, 1.3])
    assert _max_drawdown(eq) == 0.0


def test_max_drawdown_simple_dip():
    eq = np.array([1.0, 1.2, 0.9, 1.0])
    # peak=1.2, trough=0.9, dd = (0.9-1.2)/1.2 = -0.25
    assert np.isclose(_max_drawdown(eq), -0.25)


def test_max_drawdown_empty_series():
    assert _max_drawdown(np.array([])) == 0.0


# --- compare -----------------------------------------------------------------

def test_compare_returns_list_for_each():
    rets = _make_rets()
    weights_by_name = {
        "equal": np.full(5, 0.2),
        "concentrated": np.array([1.0, 0, 0, 0, 0]),
    }
    out = compare(rets, weights_by_name)
    assert len(out) == 2
    assert {m.name for m in out} == {"equal", "concentrated"}


def test_compare_diversification_lowers_volatility():
    """等权应比集中持仓波动更低（在独立收益假设下）。"""
    rets = _make_rets(T=500)
    out = compare(rets, {
        "equal": np.full(5, 0.2),
        "concentrated": np.array([1.0, 0, 0, 0, 0]),
    })
    by_name = {m.name: m for m in out}
    assert by_name["equal"].annualized_volatility < by_name["concentrated"].annualized_volatility


# --- render_report ------------------------------------------------------------

def test_render_report_includes_all_strategies():
    rets = _make_rets()
    metrics = compare(rets, {
        "alpha": np.full(5, 0.2),
        "beta": np.array([0.5, 0.5, 0, 0, 0]),
    })
    out = render_report(metrics)
    assert "alpha" in out
    assert "beta" in out
    assert "Sharpe" in out


def test_render_report_has_header_and_separator():
    metrics = compare(_make_rets(), {"x": np.full(5, 0.2)})
    out = render_report(metrics)
    lines = out.split("\n")
    assert "Strategy" in lines[0]
    assert lines[1].startswith("-")
