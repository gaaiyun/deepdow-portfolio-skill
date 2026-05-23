"""无外部数据 / 无 deepdow 依赖的快速开始 demo。

用合成收益率训一个 SimpleNet，跑所有非 DL 基准，对比结果输出到终端。
等价于：

    python -m deepdow_portfolio_skill compare --synthetic --epochs 20

但这里写成一个独立脚本，方便初次用户直接 ``python scripts/quickstart.py`` 看流程。
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from baselines import BASELINES                            # noqa: E402
from data_io import synthetic_returns                       # noqa: E402
from portfolio_eval import compare, render_report           # noqa: E402
from simple_model import (                                  # noqa: E402
    predict_weights,
    rolling_windows,
    train_simple,
)


def main() -> None:
    print("=" * 64)
    print("deepdow-portfolio-skill quickstart — synthetic data demo")
    print("=" * 64)

    ds = synthetic_returns(n_periods=600, n_assets=8, seed=42)
    rets = ds.values
    print(f"data shape: {rets.shape}  assets: {ds.asset_names[:3]}...")

    lookback, gap, horizon = 20, 1, 5
    split_ix = int(len(rets) * 0.7)
    train_rets, val_rets = rets[:split_ix], rets[split_ix:]

    X_tr, y_tr = rolling_windows(train_rets, lookback, gap, horizon)
    X_va, y_va = rolling_windows(val_rets, lookback, gap, horizon)
    means = X_tr.mean(axis=(0, 2), keepdims=True)
    stds = X_tr.std(axis=(0, 2), keepdims=True) + 1e-6
    X_tr_s = (X_tr - means) / stds
    X_va_s = (X_va - means) / stds

    print("training SimpleNet (20 epochs)...")
    net, hist = train_simple(
        torch.from_numpy(X_tr_s).float(),
        torch.from_numpy(y_tr).float(),
        torch.from_numpy(X_va_s).float(),
        torch.from_numpy(y_va).float(),
        n_assets=ds.n_assets,
        lookback=lookback,
        epochs=20,
        seed=42,
        verbose=False,
    )
    print(f"best val loss: {hist.best_val_loss:.4f}")

    dl_w = predict_weights(net, torch.from_numpy(X_va_s[-1:]).float())[0]
    weights_by_name = {name: alloc(train_rets) for name, alloc in BASELINES.items()}
    weights_by_name["SimpleNet"] = dl_w

    metrics = compare(val_rets, weights_by_name)
    metrics.sort(key=lambda m: m.sharpe, reverse=True)

    print()
    print("validation period performance (sorted by Sharpe):")
    print()
    print(render_report(metrics))
    print()
    print("note: synthetic data is i.i.d. gaussian — classic baselines often beat the DL model here.")


if __name__ == "__main__":
    main()
