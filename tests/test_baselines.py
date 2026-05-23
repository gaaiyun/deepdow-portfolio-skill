"""baselines.py 测试 —— 纯 numpy，不依赖网络 / torch / deepdow。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from baselines import (
    BASELINES,
    equal_weight,
    min_variance,
    momentum_topk,
    risk_parity,
)


# --- equal_weight -------------------------------------------------------------

def test_equal_weight_sums_to_one():
    rets = np.random.RandomState(0).randn(100, 5) * 0.01
    w = equal_weight(rets)
    assert w.shape == (5,)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, np.full(5, 0.2))


def test_equal_weight_rejects_1d():
    with pytest.raises(ValueError):
        equal_weight(np.array([0.01, 0.02, 0.03]))


# --- momentum_topk ------------------------------------------------------------

def test_momentum_topk_picks_top_3():
    rng = np.random.RandomState(0)
    rets = rng.randn(100, 10) * 0.01
    # 把第 7, 3, 5 列偏向正收益，其它偏向 0
    rets[:, 7] += 0.005
    rets[:, 3] += 0.004
    rets[:, 5] += 0.003

    w = momentum_topk(rets, top_k=3)
    assert w.shape == (10,)
    assert np.isclose(w.sum(), 1.0)
    nonzero_idx = np.where(w > 0)[0]
    assert set(nonzero_idx.tolist()) == {7, 3, 5}
    # 均权重
    assert np.allclose(w[nonzero_idx], 1 / 3)


def test_momentum_topk_lookback_window():
    rets = np.zeros((100, 5))
    # 只在最后 20 期让 asset 2 涨
    rets[-20:, 2] = 0.01
    w = momentum_topk(rets, top_k=1, lookback=20)
    assert w[2] == 1.0


def test_momentum_topk_clamps_top_k_to_n():
    """top_k > n_assets 时不应该崩。"""
    rets = np.random.RandomState(0).randn(50, 3) * 0.01
    w = momentum_topk(rets, top_k=100)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w > 0)


# --- min_variance ------------------------------------------------------------

def test_min_variance_sums_to_one():
    rng = np.random.RandomState(1)
    rets = rng.randn(200, 5) * 0.01
    w = min_variance(rets)
    assert np.isclose(w.sum(), 1.0)


def test_min_variance_no_short_by_default():
    rng = np.random.RandomState(1)
    rets = rng.randn(200, 5) * 0.01
    w = min_variance(rets)
    assert np.all(w >= -1e-12)


def test_min_variance_concentrates_in_low_vol():
    """低波动资产应拿到更高权重。"""
    rng = np.random.RandomState(2)
    rets = np.column_stack([
        rng.randn(300) * 0.005,   # 低波动
        rng.randn(300) * 0.05,    # 高波动
        rng.randn(300) * 0.03,
    ])
    w = min_variance(rets, shrinkage=0.3)
    assert w[0] > w[1], "低波动资产权重应大于高波动"


def test_min_variance_fallback_when_singular():
    """协方差奇异时回退到等权，不崩。"""
    rets = np.ones((50, 3)) * 0.001  # 完全相关
    w = min_variance(rets)
    assert np.isclose(w.sum(), 1.0)
    assert w.shape == (3,)


def test_min_variance_too_few_samples_returns_equal_weight():
    rets = np.array([[0.01, 0.02, 0.03]])  # 单行
    w = min_variance(rets)
    assert np.allclose(w, np.full(3, 1 / 3))


# --- risk_parity --------------------------------------------------------------

def test_risk_parity_sums_to_one():
    rng = np.random.RandomState(3)
    rets = rng.randn(300, 5) * 0.02
    w = risk_parity(rets)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= 0)


def test_risk_parity_equal_risk_contribution():
    """每只资产对组合方差的贡献应大致相等。"""
    rng = np.random.RandomState(4)
    rets = rng.randn(500, 4) * 0.02
    rets[:, 1] *= 2  # 让 asset 1 波动是别的 2x
    w = risk_parity(rets)
    cov = np.cov(rets, rowvar=False)
    rc = w * (cov @ w)
    rc_normalized = rc / rc.sum()
    # 每个 risk contribution 应接近 1/n（容忍 5%）
    assert np.allclose(rc_normalized, np.full(4, 0.25), atol=0.05)


def test_risk_parity_high_vol_gets_lower_weight():
    rng = np.random.RandomState(5)
    rets = np.column_stack([
        rng.randn(400) * 0.01,
        rng.randn(400) * 0.05,
    ])
    w = risk_parity(rets)
    assert w[0] > w[1], "低波动应拿更大权重（risk parity）"


# --- 注册表 -------------------------------------------------------------------

def test_baselines_registry_has_all_four():
    assert set(BASELINES.keys()) == {"equal_weight", "momentum_topk",
                                      "min_variance", "risk_parity"}


def test_all_baselines_return_valid_weights():
    """所有注册的 baseline 都能在标准输入上跑通，输出合法权重。"""
    rng = np.random.RandomState(6)
    rets = rng.randn(200, 5) * 0.02
    for name, alloc in BASELINES.items():
        w = alloc(rets)
        assert w.shape == (5,), f"{name} 输出形状错"
        assert np.isclose(w.sum(), 1.0, atol=1e-6), f"{name} 权重不归一"
        assert np.all(w >= -1e-9), f"{name} 出负权重"
