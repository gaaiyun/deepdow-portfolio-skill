"""无 deepdow 依赖的 DL 组合优化器。

把 ``quickstart.py`` 里的 ``SimpleNet`` 提炼成可 import / 可测的模块。
只用 PyTorch，不依赖 deepdow / cvxpylayers，便于在 CI 跑、便于改、便于
和 baselines 在同一份数据上对比。

主要 API：

- ``SimpleAllocator`` — 带温度的 softmax 分配器
- ``SimpleNet`` — Dropout + Linear + SimpleAllocator
- ``sharpe_ratio_loss(weights, returns)`` — 最小化 -Sharpe
- ``rolling_windows(returns, lookback, gap, horizon)`` — 切训练样本
- ``train_simple(...)`` — 完整训练循环，返回 (network, history)
- ``predict_weights(network, X)`` — 用训好的网络出权重
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch import nn


# --- 网络结构 ----------------------------------------------------------------

class SimpleAllocator(nn.Module):
    """带温度参数的 Softmax 分配器。"""

    def forward(self, x: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x / temperature, dim=-1)


class SimpleNet(nn.Module):
    """最小可用的 DL 组合优化网络。

    输入：``(n_samples, 1, lookback, n_assets)`` —— 1 个 channel 是收益率，
    lookback 个时间步，n_assets 个资产。
    输出：``(n_samples, n_assets)`` —— softmax 后的权重。
    """

    def __init__(self, n_assets: int, lookback: int, dropout_p: float = 0.5):
        super().__init__()
        self.n_assets = n_assets
        self.lookback = lookback
        n_features = n_assets * lookback
        self.dropout = nn.Dropout(p=dropout_p)
        self.dense = nn.Linear(n_features, n_assets, bias=True)
        self.allocator = SimpleAllocator()
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = x.shape[0]
        x = x.view(n_samples, -1)
        x = self.dropout(x)
        x = self.dense(x)
        # 限制温度 > 0，否则 softmax 数值爆
        temp = torch.clamp(self.temperature, min=1e-3)
        return self.allocator(x, temp)


# --- 损失函数 ----------------------------------------------------------------

def sharpe_ratio_loss(weights: torch.Tensor, future_returns: torch.Tensor,
                      eps: float = 1e-6) -> torch.Tensor:
    """最小化 -Sharpe = -mean / std。

    Parameters
    ----------
    weights : (n_samples, n_assets)
    future_returns : (n_samples, 1, horizon, n_assets) —— 未来 horizon 期收益
    """
    # squeeze 掉 channel 维
    if future_returns.dim() == 4:
        future_returns = future_returns.squeeze(1)
    # (n_samples, horizon, n_assets) × (n_samples, 1, n_assets) → (n_samples, horizon)
    portfolio_ret = torch.sum(weights.unsqueeze(1) * future_returns, dim=-1)
    mean_r = portfolio_ret.mean(dim=1)
    std_r = portfolio_ret.std(dim=1) + eps
    sharpe = mean_r / std_r
    return -sharpe.mean()


# --- 数据切窗口 --------------------------------------------------------------

def rolling_windows(returns: np.ndarray, lookback: int, gap: int, horizon: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """把 (T, N) 历史收益率切成 (X, y) 训练样本。

    每个样本：
    - X[i] = returns[i - lookback : i]                    形状 (lookback, N)
    - y[i] = returns[i + gap : i + gap + horizon]         形状 (horizon, N)

    返回 4D：X (S, 1, lookback, N), y (S, 1, horizon, N)，符合 SimpleNet 输入。
    """
    if returns.ndim != 2:
        raise ValueError(f"returns must be 2D, got {returns.shape}")
    T = returns.shape[0]
    X_list, y_list = [], []
    for i in range(lookback, T - horizon - gap + 1):
        X_list.append(returns[i - lookback: i, :])
        y_list.append(returns[i + gap: i + gap + horizon, :])
    if not X_list:
        raise ValueError(
            f"返回数据太少：T={T}, lookback={lookback}, "
            f"gap={gap}, horizon={horizon} 切不出样本"
        )
    X = np.stack(X_list, axis=0)[:, None, ...]
    y = np.stack(y_list, axis=0)[:, None, ...]
    return X, y


# --- 训练循环 ----------------------------------------------------------------

@dataclass
class TrainHistory:
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    stopped_epoch: Optional[int] = None
    best_val_loss: float = float("inf")


def train_simple(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_assets: int,
    lookback: int,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 15,
    dropout_p: float = 0.5,
    seed: Optional[int] = 42,
    verbose: bool = False,
) -> Tuple[SimpleNet, TrainHistory]:
    """训完一个 SimpleNet 并返回 (网络, 训练历史)。

    所有张量已经是 torch.Tensor、已标准化、已切好样本（用 ``rolling_windows``）。
    早停规则：val_loss 连续 ``patience`` 个 epoch 不改善就停。
    """
    if seed is not None:
        torch.manual_seed(seed)

    net = SimpleNet(n_assets=n_assets, lookback=lookback, dropout_p=dropout_p)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, amsgrad=True)

    history = TrainHistory()
    counter = 0
    for epoch in range(epochs):
        net.train()
        optimizer.zero_grad()
        weights_tr = net(X_train)
        loss_tr = sharpe_ratio_loss(weights_tr, y_train)
        loss_tr.backward()
        optimizer.step()
        history.train_losses.append(float(loss_tr.item()))

        net.eval()
        with torch.no_grad():
            weights_val = net(X_val)
            loss_val = sharpe_ratio_loss(weights_val, y_val)
        history.val_losses.append(float(loss_val.item()))

        if loss_val.item() < history.best_val_loss:
            history.best_val_loss = float(loss_val.item())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                history.stopped_epoch = epoch + 1
                if verbose:
                    print(f"  early stop at epoch {epoch + 1}")
                break

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1:3d}  train={loss_tr.item():.4f}  val={loss_val.item():.4f}")

    return net, history


def predict_weights(network: SimpleNet, X: torch.Tensor) -> np.ndarray:
    """用训好的网络出权重，返回 numpy (n_samples, n_assets)。"""
    network.eval()
    with torch.no_grad():
        w = network(X)
    return w.cpu().numpy()
