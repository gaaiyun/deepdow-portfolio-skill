"""deepdow-portfolio-skill CLI。

子命令：
    fetch       从 yfinance 抓收益率存 CSV
    baseline    对 CSV 跑非 DL 基准（equal_weight / momentum / min_variance / risk_parity）
    train       训一个 SimpleNet 并存 .pth
    compare     在同一份数据上对比 SimpleNet vs 所有 baselines，输出表格 + JSON

示例：

    python __main__.py fetch --tickers AAPL,MSFT,GOOG,AMZN,META --start 2022-01-01 \
        --end 2024-01-01 -o data/tech5.csv

    python __main__.py baseline data/tech5.csv --strategy equal_weight

    python __main__.py compare data/tech5.csv --split 0.7 --epochs 30 -o report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from baselines import BASELINES                            # noqa: E402
from data_io import load_csv, load_yfinance, synthetic_returns  # noqa: E402
from portfolio_eval import compare, evaluate, render_report  # noqa: E402


def cmd_fetch(args) -> int:
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        sys.stderr.write("[error] --tickers 不能为空\n")
        return 1
    try:
        ds = load_yfinance(tickers, start=args.start, end=args.end)
    except ImportError as e:
        sys.stderr.write(f"[error] {e}\n")
        return 2
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ds.returns.to_csv(args.output)
    sys.stderr.write(f"[ok] 写入 {args.output}  shape={ds.returns.shape}\n")
    return 0


def cmd_baseline(args) -> int:
    if args.synthetic:
        ds = synthetic_returns(n_periods=args.n_periods, n_assets=args.n_assets, seed=args.seed)
        sys.stderr.write(f"[ok] 生成合成数据 shape={ds.returns.shape}\n")
    else:
        if not args.csv:
            sys.stderr.write("[error] 需要 --csv 或 --synthetic\n")
            return 1
        ds = load_csv(args.csv)
        sys.stderr.write(f"[ok] 读 {args.csv}  shape={ds.returns.shape}\n")

    if args.strategy not in BASELINES:
        sys.stderr.write(f"[error] 未知 strategy {args.strategy}，可选 {list(BASELINES)}\n")
        return 1
    allocator = BASELINES[args.strategy]
    kwargs = {}
    if args.strategy == "momentum_topk":
        kwargs["top_k"] = args.top_k
    weights = allocator(ds.values, **kwargs)

    print(f"\n策略: {args.strategy}")
    print(f"资产: {ds.asset_names}")
    print(f"权重: {weights.round(4).tolist()}")
    metrics = evaluate(args.strategy, ds.values, weights)
    print()
    print(render_report([metrics]))
    return 0


def cmd_train(args) -> int:
    try:
        import torch
    except ImportError:
        sys.stderr.write("[error] 需要 torch：pip install torch\n")
        return 2
    from simple_model import rolling_windows, train_simple  # noqa: E402

    if args.synthetic:
        ds = synthetic_returns(n_periods=args.n_periods, n_assets=args.n_assets, seed=args.seed)
    else:
        if not args.csv:
            sys.stderr.write("[error] 需要 --csv 或 --synthetic\n")
            return 1
        ds = load_csv(args.csv)

    rets = ds.values
    split_ix = int(len(rets) * args.split)
    train_rets = rets[:split_ix]
    val_rets = rets[split_ix:]

    X_tr, y_tr = rolling_windows(train_rets, args.lookback, args.gap, args.horizon)
    X_va, y_va = rolling_windows(val_rets, args.lookback, args.gap, args.horizon)

    # 标准化（用训练集的统计量）
    means = X_tr.mean(axis=(0, 2), keepdims=True)
    stds = X_tr.std(axis=(0, 2), keepdims=True) + 1e-6
    X_tr = (X_tr - means) / stds
    X_va = (X_va - means) / stds

    net, hist = train_simple(
        torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float(),
        torch.from_numpy(X_va).float(), torch.from_numpy(y_va).float(),
        n_assets=ds.n_assets, lookback=args.lookback,
        epochs=args.epochs, lr=args.lr,
        patience=args.patience, seed=args.seed,
        verbose=True,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": net.state_dict(),
        "n_assets": ds.n_assets, "lookback": args.lookback,
        "asset_names": ds.asset_names,
        "means": means, "stds": stds,
    }, args.output)
    sys.stderr.write(f"[ok] 模型已保存 {args.output}  best_val_loss={hist.best_val_loss:.4f}\n")
    return 0


def cmd_compare(args) -> int:
    try:
        import torch
    except ImportError:
        sys.stderr.write("[error] 需要 torch：pip install torch\n")
        return 2
    from simple_model import rolling_windows, train_simple, predict_weights  # noqa: E402

    if args.synthetic:
        ds = synthetic_returns(n_periods=args.n_periods, n_assets=args.n_assets, seed=args.seed)
    else:
        if not args.csv:
            sys.stderr.write("[error] 需要 --csv 或 --synthetic\n")
            return 1
        ds = load_csv(args.csv)

    rets = ds.values
    split_ix = int(len(rets) * args.split)
    train_rets, val_rets = rets[:split_ix], rets[split_ix:]

    # 训 SimpleNet
    X_tr, y_tr = rolling_windows(train_rets, args.lookback, args.gap, args.horizon)
    X_va, y_va = rolling_windows(val_rets, args.lookback, args.gap, args.horizon)
    means = X_tr.mean(axis=(0, 2), keepdims=True)
    stds = X_tr.std(axis=(0, 2), keepdims=True) + 1e-6
    X_tr_s = (X_tr - means) / stds
    X_va_s = (X_va - means) / stds

    net, hist = train_simple(
        torch.from_numpy(X_tr_s).float(), torch.from_numpy(y_tr).float(),
        torch.from_numpy(X_va_s).float(), torch.from_numpy(y_va).float(),
        n_assets=ds.n_assets, lookback=args.lookback,
        epochs=args.epochs, lr=args.lr, patience=args.patience, seed=args.seed,
        verbose=False,
    )

    # DL 网络用最后一个 lookback 窗口的权重作为"代表权重"（对比静态权重的方法）
    dl_weights = predict_weights(net, torch.from_numpy(X_va_s[-1:]).float())[0]
    sys.stderr.write(f"[ok] 训完 SimpleNet  best_val_loss={hist.best_val_loss:.4f}\n")

    # 对所有 baseline 在训练集上拟合，得到一个固定权重
    weights_by_name = {}
    for name, alloc in BASELINES.items():
        weights_by_name[name] = alloc(train_rets)
    weights_by_name["SimpleNet"] = dl_weights

    # 在 val 段评估
    metrics = compare(val_rets, weights_by_name)
    metrics.sort(key=lambda m: m.sharpe, reverse=True)

    print()
    print(render_report(metrics))
    print()

    if args.output:
        payload = {
            "asset_names": ds.asset_names,
            "n_train": len(train_rets), "n_val": len(val_rets),
            "metrics": [m.to_dict() for m in metrics],
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        sys.stderr.write(f"[ok] 报告 JSON 写入 {args.output}\n")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deepdow-skill",
                                description="DL 组合优化 vs 非 DL 基准 — 在同一份数据上对比")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("fetch", help="从 yfinance 抓收益率到 CSV")
    sp.add_argument("--tickers", required=True, help="逗号分隔，如 AAPL,MSFT,GOOG")
    sp.add_argument("--start", required=True)
    sp.add_argument("--end")
    sp.add_argument("-o", "--output", required=True)
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("baseline", help="跑单个非 DL 基准")
    sp.add_argument("--csv")
    sp.add_argument("--synthetic", action="store_true",
                    help="用合成数据，方便无网络场景 demo")
    sp.add_argument("--n-periods", type=int, default=500)
    sp.add_argument("--n-assets", type=int, default=10)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--strategy", default="equal_weight", choices=list(BASELINES))
    sp.add_argument("--top-k", type=int, default=5, help="momentum_topk 专用")
    sp.set_defaults(func=cmd_baseline)

    sp = sub.add_parser("train", help="训一个 SimpleNet 并存 .pth")
    sp.add_argument("--csv")
    sp.add_argument("--synthetic", action="store_true")
    sp.add_argument("--n-periods", type=int, default=500)
    sp.add_argument("--n-assets", type=int, default=10)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--split", type=float, default=0.7, help="训练集比例")
    sp.add_argument("--lookback", type=int, default=40)
    sp.add_argument("--gap", type=int, default=2)
    sp.add_argument("--horizon", type=int, default=20)
    sp.add_argument("--epochs", type=int, default=30)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--patience", type=int, default=15)
    sp.add_argument("-o", "--output", default="outputs/model.pth")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("compare", help="SimpleNet vs 全部 baselines 同数据对比")
    sp.add_argument("--csv")
    sp.add_argument("--synthetic", action="store_true")
    sp.add_argument("--n-periods", type=int, default=500)
    sp.add_argument("--n-assets", type=int, default=10)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--split", type=float, default=0.7)
    sp.add_argument("--lookback", type=int, default=40)
    sp.add_argument("--gap", type=int, default=2)
    sp.add_argument("--horizon", type=int, default=20)
    sp.add_argument("--epochs", type=int, default=30)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--patience", type=int, default=15)
    sp.add_argument("-o", "--output", help="把对比结果写 JSON 到该路径")
    sp.set_defaults(func=cmd_compare)

    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
