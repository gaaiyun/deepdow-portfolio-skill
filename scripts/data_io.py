"""读 / 写收益率数据。

提供三种来源：

1. ``load_csv(path)``：从 CSV 读，索引是日期，列是各资产收益率。
2. ``load_yfinance(tickers, start, end)``：用 yfinance 抓收盘价转日收益率。
   yfinance 是可选依赖；没装时这个函数 raise。
3. ``synthetic_returns(...)``：生成合成数据，跑测试和 demo 用，不需联网。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class ReturnsDataset:
    returns: pd.DataFrame    # index = 日期，columns = 资产名
    source: str              # "csv" / "yfinance" / "synthetic"

    @property
    def values(self) -> np.ndarray:
        return self.returns.values

    @property
    def asset_names(self) -> List[str]:
        return list(self.returns.columns)

    @property
    def n_periods(self) -> int:
        return len(self.returns)

    @property
    def n_assets(self) -> int:
        return self.returns.shape[1]


def load_csv(path: str, date_column: Optional[str] = None) -> ReturnsDataset:
    """从 CSV 读收益率。

    要求第一列（或 ``date_column``）是日期，后续列每列一个资产收益率。
    """
    df = pd.read_csv(path)
    if date_column is None:
        # 假设第一列是日期
        date_column = df.columns[0]
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column).sort_index()
    df = df.astype(float)
    return ReturnsDataset(returns=df, source="csv")


def load_yfinance(tickers: List[str], start: str, end: Optional[str] = None,
                  use_adjusted: bool = True) -> ReturnsDataset:
    """从 yfinance 抓收盘价并算日对数收益率。

    Parameters
    ----------
    tickers : 比如 ["AAPL", "MSFT", "GOOG"]
    start, end : "YYYY-MM-DD"
    use_adjusted : 用复权收盘价（推荐）

    Raises
    ------
    ImportError : yfinance 未安装
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance 未安装。运行 `pip install yfinance` 后再用此函数，"
            "或改用 load_csv / synthetic_returns。"
        ) from e

    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=use_adjusted, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance 返回空数据：tickers={tickers} start={start} end={end}")

    # 多 ticker 时 raw 是 MultiIndex columns，取 Close 那一层
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(how="all")
    returns = prices.pct_change().dropna()
    return ReturnsDataset(returns=returns, source="yfinance")


def synthetic_returns(
    n_periods: int = 500,
    n_assets: int = 10,
    mu_range: tuple = (-0.0005, 0.001),
    sigma_range: tuple = (0.01, 0.03),
    seed: Optional[int] = 42,
) -> ReturnsDataset:
    """生成合成收益率数据。

    每个资产用不同的 mean / std 采高斯分布，资产间独立。
    主要给测试 / demo 用，不要拿这个数据训出来的模型去做实盘判断。
    """
    rng = np.random.default_rng(seed)
    mus = rng.uniform(*mu_range, size=n_assets)
    sigmas = rng.uniform(*sigma_range, size=n_assets)

    rets = rng.normal(loc=mus, scale=sigmas, size=(n_periods, n_assets))
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="B")
    cols = [f"ASSET_{i:02d}" for i in range(n_assets)]
    df = pd.DataFrame(rets, index=dates, columns=cols)
    return ReturnsDataset(returns=df, source="synthetic")


def save_csv(dataset: ReturnsDataset, path: str) -> None:
    """把 ReturnsDataset 写回 CSV。"""
    dataset.returns.to_csv(path)
