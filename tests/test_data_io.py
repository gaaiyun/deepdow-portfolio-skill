"""data_io.py 测试：合成数据 + CSV 读写，不发网络请求。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from data_io import (
    ReturnsDataset,
    load_csv,
    load_yfinance,
    save_csv,
    synthetic_returns,
)


# --- synthetic ---------------------------------------------------------------

def test_synthetic_returns_default_shape():
    ds = synthetic_returns()
    assert isinstance(ds, ReturnsDataset)
    assert ds.source == "synthetic"
    assert ds.n_periods == 500
    assert ds.n_assets == 10


def test_synthetic_returns_deterministic_with_seed():
    a = synthetic_returns(seed=42)
    b = synthetic_returns(seed=42)
    pd.testing.assert_frame_equal(a.returns, b.returns)


def test_synthetic_returns_different_seeds_differ():
    a = synthetic_returns(seed=1)
    b = synthetic_returns(seed=2)
    assert not a.returns.equals(b.returns)


def test_synthetic_returns_custom_shape():
    ds = synthetic_returns(n_periods=100, n_assets=3, seed=0)
    assert ds.returns.shape == (100, 3)
    assert ds.asset_names == ["ASSET_00", "ASSET_01", "ASSET_02"]


def test_synthetic_returns_values_property():
    ds = synthetic_returns(n_periods=50, n_assets=4, seed=0)
    arr = ds.values
    assert arr.shape == (50, 4)
    assert isinstance(arr, np.ndarray)


# --- CSV round-trip ----------------------------------------------------------

def test_save_and_load_csv_roundtrip(tmp_path):
    ds = synthetic_returns(n_periods=100, n_assets=5, seed=0)
    p = tmp_path / "rets.csv"
    save_csv(ds, str(p))
    assert p.exists()

    loaded = load_csv(str(p))
    assert loaded.source == "csv"
    assert loaded.n_assets == 5
    assert loaded.n_periods == 100
    # 数值近似
    np.testing.assert_allclose(loaded.values, ds.values, rtol=1e-9)


def test_load_csv_explicit_date_column(tmp_path):
    df = pd.DataFrame({
        "trade_date": pd.date_range("2020-01-01", periods=10, freq="B"),
        "A": np.random.RandomState(0).randn(10) * 0.01,
        "B": np.random.RandomState(1).randn(10) * 0.01,
    })
    p = tmp_path / "named_date.csv"
    df.to_csv(p, index=False)

    ds = load_csv(str(p), date_column="trade_date")
    assert ds.n_periods == 10
    assert ds.asset_names == ["A", "B"]


# --- yfinance (no network) ---------------------------------------------------

def test_load_yfinance_raises_when_not_installed(monkeypatch):
    """断 yfinance 模块时应给出明确错。"""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "yfinance":
            raise ImportError("simulated missing yfinance")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="yfinance"):
        load_yfinance(["AAPL"], start="2024-01-01")
