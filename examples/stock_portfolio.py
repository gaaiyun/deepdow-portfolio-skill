"""
Stock Portfolio Example - 多因子股票组合优化

使用多个技术因子构建股票投资组合
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path

try:
    from deepdow.nn import BachelierNet
    from deepdow.data import InRAMDataset, RigidDataLoader, prepare_standard_scaler, Scale
    from deepdow.experiments import Run
    from deepdow.losses import SharpeRatio, MaximumDrawdown
    from deepdow.callbacks import EarlyStoppingCallback
    from deepdow.benchmarks import OneOverN
    from deepdow.visualize import generate_metrics_table, plot_metrics
    import matplotlib.pyplot as plt
except ImportError:
    print("❌ deepdow 未安装！请运行: pip install deepdow")
    exit(1)


def calculate_technical_indicators(prices):
    """计算技术指标
    
    Parameters
    ----------
    prices : pd.DataFrame
        价格数据，形状 (n_timesteps, n_assets)
    
    Returns
    -------
    features : np.ndarray
        特征矩阵，形状 (n_timesteps, n_channels, n_assets)
    """
    print("📊 计算技术指标...")
    
    # 1. 收益率
    returns = prices.pct_change().fillna(0)
    
    # 2. 动量 (20日)
    momentum = prices.pct_change(20).fillna(0)
    
    # 3. 波动率 (20日滚动标准差)
    volatility = returns.rolling(20).std().fillna(0)
    
    # 4. RSI (相对强弱指标)
    delta = returns
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50) / 100  # 归一化到 [0, 1]
    
    # 5. 布林带位置
    sma = prices.rolling(20).mean()
    std = prices.rolling(20).std()
    bb_position = (prices - sma) / (2 * std + 1e-10)
    bb_position = bb_position.fillna(0).clip(-1, 1)
    
    # 堆叠特征
    features = np.stack([
        returns.values,
        momentum.values,
        volatility.values,
        rsi.values,
        bb_position.values
    ], axis=1)
    
    print(f"  ✅ 特征形状: {features.shape}")
    print(f"  ✅ 特征: 收益率, 动量, 波动率, RSI, 布林带")
    
    return features


def prepare_stock_dataset(prices, lookback=40, gap=2, horizon=20):
    """准备股票数据集"""
    n_timesteps, n_assets = prices.shape
    
    # 计算技术指标
    features = calculate_technical_indicators(prices)
    
    # 收益率作为目标
    returns = prices.pct_change().fillna(0).values
    
    # 滚动窗口
    X_list, y_list = [], []
    for i in range(lookback, n_timesteps - horizon - gap + 1):
        X_list.append(features[i - lookback: i, :, :])
        y_list.append(returns[i + gap: i + gap + horizon, :])
    
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)[:, None, ...]
    
    print(f"\n📊 数据集准备:")
    print(f"  - X 形状: {X.shape}")
    print(f"  - y 形状: {y.shape}")
    
    return X, y


def main():
    print("=" * 60)
    print("📈 多因子股票组合优化示例")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 生成模拟股票价格数据
    print("\n📊 生成模拟股票数据...")
    n_timesteps, n_assets = 1000, 50
    
    # 模拟价格（随机游走 + 趋势）
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, (n_timesteps, n_assets))
    prices = pd.DataFrame(100 * np.exp(returns.cumsum(axis=0)))
    
    print(f"  ✅ 生成 {n_assets} 只股票，{n_timesteps} 个时间步")
    
    # 准备数据集
    lookback, gap, horizon = 40, 2, 20
    X, y = prepare_stock_dataset(prices, lookback, gap, horizon)
    
    n_samples = X.shape[0]
    split_ix = int(n_samples * 0.8)
    indices_train = list(range(split_ix))
    indices_test = list(range(split_ix, n_samples))
    
    # 标准化
    means, stds = prepare_standard_scaler(X, indices=indices_train)
    dataset = InRAMDataset(X, y, transform=Scale(means, stds))
    
    # 数据加载器
    dataloader_train = RigidDataLoader(dataset, indices=indices_train, batch_size=32)
    dataloader_test = RigidDataLoader(dataset, indices=indices_test, batch_size=32)
    
    # 创建网络
    print("\n🧠 创建 BachelierNet...")
    network = BachelierNet(
        n_input_channels=5,  # 5个技术指标
        n_assets=n_assets,
        hidden_size=32,
        max_weight=0.1,  # 单只股票最大10%
        shrinkage_strategy='diagonal',
        p=0.3
    )
    
    print(network)
    
    # 损失函数
    loss = SharpeRatio() + MaximumDrawdown()
    
    # 训练
    print("\n🏋️ 开始训练...")
    run = Run(
        network,
        loss,
        dataloader_train,
        val_dataloaders={'test': dataloader_test},
        optimizer=torch.optim.Adam(network.parameters(), lr=0.001),
        callbacks=[EarlyStoppingCallback(
            metric_name='loss',
            dataloader_name='test',
            patience=10
        )]
    )
    
    history = run.launch(30)
    
    # 评估
    print("\n📊 评估模型...")
    network = network.eval()
    
    benchmarks = {
        '1overN': OneOverN(),
        'BachelierNet': network
    }
    
    metrics = {
        'Sharpe': SharpeRatio(),
        'MaxDD': MaximumDrawdown()
    }
    
    metrics_table = generate_metrics_table(benchmarks, dataloader_test, metrics)
    
    # 可视化
    output_dir = Path("outputs/stock_portfolio")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plot_metrics(metrics_table)
    plt.savefig(output_dir / "metrics.png", dpi=150, bbox_inches='tight')
    print(f"\n✅ 保存指标图: {output_dir / 'metrics.png'}")
    
    # 保存模型
    torch.save({
        'model_state_dict': network.state_dict(),
        'n_assets': n_assets,
        'lookback': lookback
    }, output_dir / "model.pth")
    
    print("\n" + "=" * 60)
    print("✅ 股票组合优化示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
