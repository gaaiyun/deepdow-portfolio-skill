---
name: deepdow-portfolio-skill
description: 深度学习组合优化与四个非 DL 基准（equal_weight / momentum / min_variance / risk_parity）在同一份数据上做 Sharpe / 最大回撤 / Calmar 对比的实验工具。
---

# deepdow-portfolio-skill

## 什么时候用

- "帮我跑一下 AAPL / MSFT / GOOG / AMZN 这几个的最小方差组合"
- "等权 1/N 和风险平价在过去两年谁更好"
- "我想训一个 DL 组合优化模型，看看比经典方法好多少"
- "把我这份 CSV 收益率数据跑一遍 portfolio backtest"

## 入口

```bash
python __main__.py fetch --tickers AAPL,MSFT,GOOG --start 2022-01-01 -o data.csv
python __main__.py baseline --csv data.csv --strategy risk_parity
python __main__.py train    --csv data.csv --epochs 30 -o outputs/model.pth
python __main__.py compare  --csv data.csv -o report.json
```

库调用入口：

- `scripts/baselines.py::BASELINES` — `{"equal_weight": fn, "momentum_topk": fn, ...}`
- `scripts/portfolio_eval.py::compare(returns, weights_by_name) -> List[PerformanceMetrics]`
- `scripts/data_io.py::load_yfinance / load_csv / synthetic_returns`
- `scripts/simple_model.py::SimpleNet / train_simple / predict_weights`

## 评估指标

| 指标 | 含义 |
|---|---|
| `cum_return` | 整段累计收益（不年化） |
| `annualized_return` | 按 252 period/年年化 |
| `annualized_volatility` | 同上 |
| `sharpe` | `(mean - rf) / std × √252` |
| `sortino` | 只用下行波动算分母 |
| `max_drawdown` | 累计净值最大回撤（负数） |
| `calmar` | 年化收益 / |最大回撤| |

## 依赖

- 必需：Python 3.10+，numpy / pandas / torch
- 可选 yfinance（抓真实日线）
- 可选 deepdow（用真 deepdow 框架那条路径：`scripts/train.py` / `predict.py`）

## 注意事项

- 合成数据是 i.i.d. 高斯，没可学时序结构；DL 模型在合成数据上没必然优势。
- 拿对比报告做投资决策前请用至少一段 out-of-sample 数据复核，不构成投资建议。
- 月线 / 周线数据要把 `evaluate(..., periods_per_year=12)` / `=52` 传对。
