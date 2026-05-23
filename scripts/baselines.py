"""非深度学习的组合优化基准 (baselines)。

deepdow 的论文 / 仓库示例都跳过了一个最实际的问题：跑出来的神经网络组合
到底比"什么也不做"强多少？这里提供四个常见非 DL 基线，全部基于 numpy，
无外部依赖，能在 CI 跑得动：

- ``equal_weight``：1/N，组合理论里出名的"难打败"。
- ``momentum_topk``：选过去 lookback 窗口收益最高的前 k 只，等权。
- ``min_variance``：解析 Markowitz 最小方差（无约束 + 缩水后协方差矩阵）。
- ``risk_parity``：等风险贡献，固定点迭代（Maillard et al. 2010）。

所有 allocator 函数签名统一：``(returns, **kwargs) -> weights``

其中 ``returns`` 是 (T, N) 的历史收益率，``weights`` 是 (N,) 一维向量，
``sum(weights) == 1``，``weights >= 0``（默认禁止做空）。
"""
from __future__ import annotations

import numpy as np


def equal_weight(returns: np.ndarray, **_) -> np.ndarray:
    """1/N。完全忽略 returns 内容，但仍然要它的 shape。"""
    if returns.ndim != 2:
        raise ValueError(f"returns must be 2D (T, N), got shape {returns.shape}")
    n = returns.shape[1]
    return np.ones(n) / n


def momentum_topk(returns: np.ndarray, top_k: int = 5,
                  lookback: int | None = None) -> np.ndarray:
    """选 lookback 窗口累计收益 Top-K，等权。

    Parameters
    ----------
    returns : (T, N)
    top_k : 选前 K 名
    lookback : 用最后多少行算累计收益。None 表示用全部。
    """
    if returns.ndim != 2:
        raise ValueError(f"returns must be 2D, got {returns.shape}")
    n = returns.shape[1]
    top_k = min(top_k, n)
    window = returns if lookback is None else returns[-lookback:]
    cum_ret = (1 + window).prod(axis=0) - 1
    top_idx = np.argsort(cum_ret)[-top_k:]
    weights = np.zeros(n)
    weights[top_idx] = 1.0 / top_k
    return weights


def min_variance(returns: np.ndarray, shrinkage: float = 0.1,
                 allow_short: bool = False) -> np.ndarray:
    """最小方差组合（解析解 + Ledoit-Wolf 风格缩水）。

    无约束解 w ∝ Σ⁻¹ · 1。如果 ``allow_short=False`` 把负数截到 0 再
    归一化（不是严格的 KKT 解，但够实用）。

    Parameters
    ----------
    returns : (T, N)
    shrinkage : Σ 向单位阵缩水的力度（0 表示纯样本协方差，1 表示对角）。
    allow_short : 是否允许负权重。
    """
    if returns.ndim != 2:
        raise ValueError(f"returns must be 2D, got {returns.shape}")
    n = returns.shape[1]
    if returns.shape[0] < 2:
        # 样本太少，退化为等权
        return np.ones(n) / n

    sample_cov = np.cov(returns, rowvar=False)
    # 缩水到对角（avg variance * I）
    avg_var = np.trace(sample_cov) / n
    target = avg_var * np.eye(n)
    cov = (1 - shrinkage) * sample_cov + shrinkage * target

    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # 奇异 → 加小 jitter
        inv = np.linalg.inv(cov + 1e-6 * np.eye(n))

    ones = np.ones(n)
    raw = inv @ ones

    if not allow_short:
        raw = np.clip(raw, 0, None)

    total = raw.sum()
    if total <= 1e-12:
        # 全部被截零，回退等权
        return np.ones(n) / n
    return raw / total


def risk_parity(returns: np.ndarray, max_iter: int = 500,
                tol: float = 1e-8) -> np.ndarray:
    """等风险贡献组合（Maillard, Roncalli, Teïletche 2010）。

    每只资产对组合方差的边际贡献相等：``w_i × (Σw)_i = const``。

    固定点迭代（Spinu 2013 / Bai et al. 2016 风格）：

        w_i^{k+1} = w_i^k × sqrt(target_rc / (w_i^k × (Σw^k)_i))
        w ← w / sum(w)

    比朴素的 ``w = target/marginal`` 收敛性更好（带平方根的阻尼）。

    Parameters
    ----------
    returns : (T, N)
    max_iter, tol : 迭代上限和收敛阈
    """
    if returns.ndim != 2:
        raise ValueError(f"returns must be 2D, got {returns.shape}")
    n = returns.shape[1]
    if returns.shape[0] < 2:
        return np.ones(n) / n

    cov = np.cov(returns, rowvar=False)
    cov = cov + 1e-10 * np.eye(n)        # 防奇异

    # 初值：逆波动率
    sigma = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    w = (1 / sigma)
    w = w / w.sum()

    target = 1.0 / n  # 归一化后的目标 risk contribution（占总方差的份额）

    for _ in range(max_iter):
        marginal = cov @ w                    # ∂σ²/∂w
        rc = w * marginal                     # 各资产对总方差的贡献
        total_var = rc.sum()
        if total_var <= 1e-20:
            break
        rc_share = rc / total_var             # 占比，目标是 1/n
        # 阻尼乘法更新：朝目标走
        w = w * np.sqrt(target / np.maximum(rc_share, 1e-14))
        w = w / w.sum()
        if np.max(np.abs(rc_share - target)) < tol:
            break

    return w


# 注册表：CLI / 测试可以按字符串名拿 allocator
BASELINES: dict[str, callable] = {
    "equal_weight": equal_weight,
    "momentum_topk": momentum_topk,
    "min_variance": min_variance,
    "risk_parity": risk_parity,
}
