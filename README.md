# deepdow-portfolio-skill

围绕 [deepdow](https://github.com/jankrepl/deepdow) 框架的组合优化实验工具：
**用同一份数据训一个 DL 模型 + 跑四个非 DL 基准，按 Sharpe / 最大回撤 / Calmar 排出
胜负**。

把"深度学习真的能打败 1/N 吗？"这个常被回避的问题，做成几条命令能跑完的对比报告。

## v2 新增（可在 CI 跑、不依赖 deepdow）

| 模块 | 干什么 |
|---|---|
| `scripts/baselines.py` | 四个非 DL 基准：`equal_weight` / `momentum_topk` / `min_variance`（带 Ledoit-Wolf 缩水）/ `risk_parity`（Spinu 阻尼迭代） |
| `scripts/portfolio_eval.py` | 固定权重组合在历史上的表现：年化收益 / 波动 / Sharpe / Sortino / 最大回撤 / Calmar |
| `scripts/data_io.py` | 三个数据源：CSV / yfinance / 合成（不联网） |
| `scripts/simple_model.py` | 从 `quickstart.py` 提炼的 `SimpleNet`：Dropout → Linear → Softmax 分配，**无 deepdow / cvxpylayers 依赖**，PyTorch 直跑 |
| `__main__.py` | 统一 CLI：`fetch` / `baseline` / `train` / `compare` |

测试 53 个，跑 4 秒，不发网络请求、不需要 GPU。

## 安装

```bash
pip install -r requirements.txt
# 抓真数据：pip install yfinance
# 用真 deepdow 框架那条路径：pip install deepdow
```

## 快速开始

```bash
# 1. 合成数据 demo（不需要联网、不需要 GPU、不需要 deepdow）
python scripts/quickstart.py
# 输出：训一个 SimpleNet + 跑 4 个 baseline + 按 Sharpe 排表

# 2. 从 yfinance 抓真实日线
python __main__.py fetch --tickers AAPL,MSFT,GOOG,AMZN,META \
    --start 2022-01-01 --end 2024-01-01 -o data/tech5.csv

# 3. 单独跑一个 baseline 看权重 + 表现
python __main__.py baseline --csv data/tech5.csv --strategy risk_parity

# 4. 同数据上：SimpleNet vs 所有 baseline，输出对比表 + JSON 报告
python __main__.py compare --csv data/tech5.csv --split 0.7 \
    --lookback 20 --horizon 5 --epochs 30 -o report.json
```

`compare` 命令是核心 —— 同一份数据按 split 切训练 / 验证段，分别在验证段评估，
不同策略放一张表比，避免"挑数据 + 挑指标"式的假对比。

## 一个真实输出例子

```
$ python scripts/quickstart.py
data shape: (600, 8)  assets: ['ASSET_00', 'ASSET_01', 'ASSET_02']...
training SimpleNet (20 epochs)...
best val loss: -0.0544

validation period performance (sorted by Sharpe):

Strategy       CumRet  AnnRet  AnnVol  Sharpe  Sortino  MaxDD    Calmar
-----------------------------------------------------------------------
min_variance   8.18%   12.21%  10.19%  1.13    1.68     -5.88%   2.08
risk_parity    8.16%   12.23%  10.58%  1.09    1.60     -5.71%   2.14
SimpleNet      8.59%   12.93%  11.21%  1.08    1.59     -7.52%   1.72
equal_weight   8.60%   13.02%  11.76%  1.04    1.52     -6.32%   2.06
momentum_topk  0.42%   1.35%   12.34%  0.11    0.16     -10.94%  0.12

note: synthetic data is i.i.d. gaussian — classic baselines often beat the DL model here.
```

注意：在合成 i.i.d. 高斯数据上，最小方差和风险平价稳定打过 DL —— 因为没有可学的
时序结构。这个"DL 没特别赢"的结果本身就是有用的信号，省得你白训。

## 库调用

```python
from scripts.baselines import min_variance, risk_parity
from scripts.portfolio_eval import compare, render_report
from scripts.data_io import load_csv

ds = load_csv("data/tech5.csv")
rets = ds.values

train, val = rets[:int(len(rets)*0.7)], rets[int(len(rets)*0.7):]

# 拟合静态权重在训练段
weights = {
    "MinVar":     min_variance(train, shrinkage=0.2),
    "RiskParity": risk_parity(train),
}

# 验证段评估
metrics = compare(val, weights)
print(render_report(metrics))
```

## v1 deepdow 那条路径（仍保留）

如果装得动完整 `deepdow`（含 `cvxpylayers` —— Windows 编译麻烦），可以走 v1：

```bash
pip install deepdow

# 训练（用真 deepdow 框架的 RNN + AttentionCollapse + SoftmaxAllocator）
python scripts/train.py --data your_data.csv --lookback 40 --horizon 20

# 预测
python scripts/predict.py --model outputs/model.pth --data test_data.csv
```

`scripts/train.py` / `predict.py` / `examples/stock_portfolio.py` 用的是 deepdow
原生 API（`RigidDataLoader` / `Run` / `BachelierNet` 等），数据复杂、依赖重，但
完整性更高。

## 校验码 / 错误码 / 设计取舍

- **不预设 DL 一定赢**：所有 baseline 都在训练段拟合权重然后到验证段评估，与 DL
  使用同一份数据切分，对比口径一致。
- **min_variance 默认禁止做空**：负权重截到 0 再归一化。允许做空设 `allow_short=True`。
- **risk_parity 用 Spinu 2013 阻尼乘法迭代**：比朴素的 `w = target/marginal` 数值
  稳定得多（避免权重在低协方差资产上爆炸）。
- **SimpleNet 不带可微分凸优化**：故意省掉 `cvxpylayers`，因为它在 Windows 上要
  C++ 编译器 + CMake，绝大多数用户装不上。SimpleNet 的 `Linear + Softmax` 已经
  够说明"DL 在组合优化里是否有用"。
- **合成数据用 i.i.d. 高斯**：故意不藏可学结构，让 baseline 公平打 DL，避免给
  人"DL 神乎其神"的假象。

## 项目结构

```
deepdow-portfolio-skill/
├── __main__.py                  # CLI：fetch / baseline / train / compare
├── scripts/
│   ├── baselines.py             # v2：4 个非 DL 基准
│   ├── portfolio_eval.py        # v2：表现指标 + 对比报告
│   ├── data_io.py               # v2：CSV / yfinance / 合成数据
│   ├── simple_model.py          # v2：无 deepdow 依赖的 SimpleNet
│   ├── quickstart.py            # v2：合成数据 demo
│   ├── train.py                 # v1：用真 deepdow 训练
│   └── predict.py               # v1：用真 deepdow 预测
├── examples/
│   └── stock_portfolio.py       # v1：BachelierNet + 多因子示例
├── tests/                       # 53 个 pytest，不联网
└── requirements.txt
```

## 测试

```bash
pip install pytest
pytest tests/ -v
```

`tests/test_simple_model.py` 在 torch 未装时整文件 skip，其它三个文件纯 numpy，
能在任何 CI 环境跑通。

## 已知限制

- `compare` / `train` 默认只输出"最后一个 lookback 窗口"的 DL 权重作为静态对比 —
  这是简化处理。如果想做滚动重算 DL 权重 + 滚动重平衡的对比，需要自己改 `compare`
  的循环。
- `min_variance` 截负到 0 不是严格 KKT 解，结果在极端情况下可能不是最优。
- `momentum_topk` 等权分配 top-K，没有按动量大小加权。
- 所有指标按 252 个 period/年算年化 —— 月线 / 周线数据要传 `periods_per_year`。

## 许可

Apache License 2.0 — 见 [LICENSE](LICENSE)。原 deepdow 框架的版权归 jankrepl。
