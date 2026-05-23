"""组合表现评估：指标 + 多策略对比报告。

输入是一段历史收益率 (T, N) 和若干 (N,) 权重向量，输出每个组合的：

- 年化收益率
- 年化波动率
- Sharpe（无风险利率默认 0）
- Sortino（只用下行波动）
- 最大回撤（max drawdown）
- Calmar（年化收益 / |max drawdown|）

外加一个简单的"再平衡式"模拟：固定权重持有，每个 period 用持仓权重 ×
日收益得到组合日收益。这是评估"持有"型组合最朴素的方式，不考虑交易费、
滑点、税。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class PerformanceMetrics:
    name: str
    n_periods: int
    cum_return: float                 # 累计收益（不是年化）
    annualized_return: float          # 假设 252 个 period/年
    annualized_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float               # 负数，比如 -0.2 表示最大跌 20%
    calmar: float
    weights: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_periods": self.n_periods,
            "cum_return": float(self.cum_return),
            "annualized_return": float(self.annualized_return),
            "annualized_volatility": float(self.annualized_volatility),
            "sharpe": float(self.sharpe),
            "sortino": float(self.sortino),
            "max_drawdown": float(self.max_drawdown),
            "calmar": float(self.calmar),
            "weights": self.weights.tolist(),
        }


def _max_drawdown(equity_curve: np.ndarray) -> float:
    """从 equity 序列算最大回撤（负值）。"""
    if len(equity_curve) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min())


def evaluate(name: str, returns: np.ndarray, weights: np.ndarray,
             periods_per_year: int = 252,
             rf: float = 0.0) -> PerformanceMetrics:
    """评估一个固定权重组合在历史收益上的表现。

    Parameters
    ----------
    name : 组合名（用于报告）
    returns : (T, N) 历史收益率
    weights : (N,) 固定权重，sum=1
    periods_per_year : 年化因子，日线 252，周线 52，月线 12
    rf : 无风险利率（同 period 单位）
    """
    returns = np.asarray(returns, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if returns.ndim != 2:
        raise ValueError(f"returns must be 2D (T, N), got {returns.shape}")
    if weights.ndim != 1 or weights.shape[0] != returns.shape[1]:
        raise ValueError(
            f"weights shape {weights.shape} 与 returns 列数 {returns.shape[1]} 不匹配"
        )

    # 组合日收益
    portfolio_ret = returns @ weights              # (T,)
    # 累计净值（从 1 开始）
    equity_curve = np.cumprod(1 + portfolio_ret)
    cum_return = float(equity_curve[-1] - 1) if len(equity_curve) else 0.0

    excess = portfolio_ret - rf
    mean_excess = float(excess.mean())
    std = float(portfolio_ret.std(ddof=1)) if len(portfolio_ret) > 1 else 0.0

    # 年化
    ann_ret = mean_excess * periods_per_year + rf * periods_per_year
    # 实际上更标准的是 (1+mean)^pp - 1，但 mean 已经在 daily 上接近线性
    ann_ret = float((1 + portfolio_ret.mean()) ** periods_per_year - 1)
    ann_vol = float(std * np.sqrt(periods_per_year))

    sharpe = float(mean_excess / std * np.sqrt(periods_per_year)) if std > 1e-12 else 0.0

    # Sortino：只用下行波动
    downside = np.minimum(portfolio_ret - rf, 0)
    downside_std = float(np.sqrt((downside ** 2).mean())) if len(downside) else 0.0
    sortino = float(mean_excess / downside_std * np.sqrt(periods_per_year)) \
        if downside_std > 1e-12 else 0.0

    mdd = _max_drawdown(equity_curve)
    calmar = float(ann_ret / abs(mdd)) if abs(mdd) > 1e-12 else 0.0

    return PerformanceMetrics(
        name=name,
        n_periods=len(portfolio_ret),
        cum_return=cum_return,
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        weights=weights,
    )


def compare(returns: np.ndarray,
            weights_by_name: Dict[str, np.ndarray],
            periods_per_year: int = 252,
            rf: float = 0.0) -> List[PerformanceMetrics]:
    """对一组 {名字: 权重} 同时评估，返回 metrics list。"""
    return [
        evaluate(name, returns, w, periods_per_year=periods_per_year, rf=rf)
        for name, w in weights_by_name.items()
    ]


def render_report(metrics_list: List[PerformanceMetrics]) -> str:
    """把对比结果排成等宽表格。"""
    headers = ["Strategy", "CumRet", "AnnRet", "AnnVol", "Sharpe", "Sortino", "MaxDD", "Calmar"]
    rows = [headers]
    for m in metrics_list:
        rows.append([
            m.name,
            f"{m.cum_return:.2%}",
            f"{m.annualized_return:.2%}",
            f"{m.annualized_volatility:.2%}",
            f"{m.sharpe:.2f}",
            f"{m.sortino:.2f}",
            f"{m.max_drawdown:.2%}",
            f"{m.calmar:.2f}",
        ])
    # 自动算列宽
    col_widths = [max(len(r[c]) for r in rows) for c in range(len(headers))]
    lines = []
    for i, row in enumerate(rows):
        line = "  ".join(c.ljust(col_widths[j]) for j, c in enumerate(row))
        lines.append(line)
        if i == 0:
            lines.append("-" * len(line))
    return "\n".join(lines)
