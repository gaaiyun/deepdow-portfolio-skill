"""simple_model.py 测试 —— 用合成数据 + 少 epoch 跑得快。

仅在 torch 可用时启用；torch 未装就 skip 整个文件。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

torch = pytest.importorskip("torch")

from simple_model import (
    SimpleAllocator,
    SimpleNet,
    TrainHistory,
    predict_weights,
    rolling_windows,
    sharpe_ratio_loss,
    train_simple,
)


# --- rolling_windows ---------------------------------------------------------

def test_rolling_windows_shape():
    rng = np.random.RandomState(0)
    rets = rng.randn(200, 5) * 0.01
    X, y = rolling_windows(rets, lookback=20, gap=2, horizon=10)
    # n_samples = 200 - 20 - 10 - 2 + 1 = 169
    assert X.shape == (169, 1, 20, 5)
    assert y.shape == (169, 1, 10, 5)


def test_rolling_windows_raises_when_too_few():
    rets = np.zeros((10, 3))
    with pytest.raises(ValueError):
        rolling_windows(rets, lookback=20, gap=2, horizon=10)


def test_rolling_windows_rejects_1d():
    with pytest.raises(ValueError):
        rolling_windows(np.zeros(50), lookback=10, gap=1, horizon=5)


# --- SimpleAllocator ---------------------------------------------------------

def test_simple_allocator_sums_to_one():
    alloc = SimpleAllocator()
    x = torch.randn(8, 5)
    temp = torch.ones(1)
    w = alloc(x, temp)
    assert w.shape == (8, 5)
    assert torch.allclose(w.sum(dim=-1), torch.ones(8), atol=1e-6)


def test_simple_allocator_high_temperature_uniform():
    alloc = SimpleAllocator()
    x = torch.tensor([[1.0, 2.0, 5.0]])
    w_low = alloc(x, torch.tensor([0.1]))
    w_high = alloc(x, torch.tensor([100.0]))
    # 高温度让分布更平
    var_low = w_low.var().item()
    var_high = w_high.var().item()
    assert var_high < var_low


# --- SimpleNet ---------------------------------------------------------------

def test_simple_net_output_shape():
    net = SimpleNet(n_assets=10, lookback=20, dropout_p=0.0)
    x = torch.randn(4, 1, 20, 10)
    w = net(x)
    assert w.shape == (4, 10)
    # 检查 softmax: 每行和 = 1
    assert torch.allclose(w.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_simple_net_eval_mode_deterministic():
    net = SimpleNet(n_assets=5, lookback=10, dropout_p=0.5)
    net.eval()
    x = torch.randn(2, 1, 10, 5)
    w1 = net(x)
    w2 = net(x)
    assert torch.allclose(w1, w2)


# --- sharpe_ratio_loss -------------------------------------------------------

def test_sharpe_loss_is_scalar():
    weights = torch.softmax(torch.randn(8, 5), dim=-1)
    future = torch.randn(8, 1, 10, 5) * 0.01
    loss = sharpe_ratio_loss(weights, future)
    assert loss.dim() == 0


def test_sharpe_loss_positive_returns_negative_loss():
    """组合一直涨 → -Sharpe < 0."""
    weights = torch.full((4, 3), 1 / 3)
    future = torch.full((4, 1, 10, 3), 0.005)  # 正收益
    # 全部一样导致 std = 0，加 eps 后 sharpe 会非常大，loss 非常小（负）
    loss = sharpe_ratio_loss(weights, future)
    assert loss.item() < 0


def test_sharpe_loss_accepts_3d_future():
    """squeeze 兼容性：直接传 (S, H, N) 也行。"""
    weights = torch.softmax(torch.randn(4, 3), dim=-1)
    future = torch.randn(4, 10, 3) * 0.01
    loss = sharpe_ratio_loss(weights, future)
    assert loss.dim() == 0


# --- train_simple end-to-end ------------------------------------------------

def test_train_simple_smoke():
    """跑一次完整训练，验证返回 (net, history)。"""
    rng = np.random.RandomState(0)
    rets = rng.randn(200, 4) * 0.01
    X, y = rolling_windows(rets, lookback=10, gap=1, horizon=5)
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()

    net, hist = train_simple(
        X_t[:100], y_t[:100], X_t[100:], y_t[100:],
        n_assets=4, lookback=10,
        epochs=3, patience=10, seed=0, verbose=False,
    )
    assert isinstance(net, SimpleNet)
    assert isinstance(hist, TrainHistory)
    assert len(hist.train_losses) == 3
    assert len(hist.val_losses) == 3


def test_train_simple_reproducible_with_seed():
    rng = np.random.RandomState(0)
    rets = rng.randn(150, 3) * 0.01
    X, y = rolling_windows(rets, lookback=10, gap=1, horizon=5)
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()

    n1, h1 = train_simple(X_t[:80], y_t[:80], X_t[80:], y_t[80:],
                          n_assets=3, lookback=10, epochs=3, seed=42, verbose=False)
    n2, h2 = train_simple(X_t[:80], y_t[:80], X_t[80:], y_t[80:],
                          n_assets=3, lookback=10, epochs=3, seed=42, verbose=False)
    assert h1.train_losses == h2.train_losses
    assert h1.val_losses == h2.val_losses


# --- predict_weights ---------------------------------------------------------

def test_predict_weights_shape_and_sums_to_one():
    net = SimpleNet(n_assets=5, lookback=10, dropout_p=0)
    X = torch.randn(7, 1, 10, 5)
    w = predict_weights(net, X)
    assert w.shape == (7, 5)
    np.testing.assert_allclose(w.sum(axis=1), np.ones(7), atol=1e-5)


def test_predict_weights_returns_numpy():
    net = SimpleNet(n_assets=3, lookback=8, dropout_p=0)
    X = torch.randn(2, 1, 8, 3)
    w = predict_weights(net, X)
    assert isinstance(w, np.ndarray)
