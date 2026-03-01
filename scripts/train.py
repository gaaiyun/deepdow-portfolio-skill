"""
Train - 真实数据训练脚本

使用真实市场数据训练深度学习投资组合优化模型
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from deepdow.benchmarks import OneOverN
    from deepdow.callbacks import EarlyStoppingCallback
    from deepdow.data import InRAMDataset, RigidDataLoader, prepare_standard_scaler, Scale
    from deepdow.experiments import Run
    from deepdow.layers import SoftmaxAllocator, RNN, AttentionCollapse
    from deepdow.losses import MeanReturns, SharpeRatio, MaximumDrawdown
    from deepdow.visualize import generate_metrics_table, plot_metrics
except ImportError:
    print("❌ deepdow 未安装！请运行: pip install deepdow")
    exit(1)


class PortfolioNet(torch.nn.Module):
    """投资组合优化网络"""
    
    def __init__(self, n_channels, n_assets, lookback, hidden_size=32, p=0.3):
        super().__init__()
        
        self.rnn = RNN(n_channels=n_channels, hidden_size=hidden_size, cell_type='LSTM')
        self.collapse = AttentionCollapse(n_channels=hidden_size)
        
        n_features = hidden_size * n_assets
        self.dropout = torch.nn.Dropout(p=p)
        self.dense = torch.nn.Linear(n_features, n_assets)
        self.allocate = SoftmaxAllocator(temperature=None)
        self.temperature = torch.nn.Parameter(torch.ones(1), requires_grad=True)
    
    def forward(self, x):
        n_samples = x.shape[0]
        
        # RNN 特征提取
        x = self.rnn(x)
        
        # 时间聚合
        x = self.collapse(x)
        
        # 展平并分配
        x = x.view(n_samples, -1)
        x = self.dropout(x)
        x = self.dense(x)
        
        temperatures = torch.ones(n_samples).to(x.device, x.dtype) * self.temperature
        weights = self.allocate(x, temperatures)
        
        return weights


def load_data(data_path):
    """加载数据
    
    Parameters
    ----------
    data_path : str
        CSV 文件路径，格式：日期 × 资产收益率
    
    Returns
    -------
    returns : np.ndarray
        收益率矩阵，形状 (n_timesteps, n_assets)
    asset_names : list
        资产名称列表
    """
    print(f"📂 加载数据: {data_path}")
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    returns = df.values
    asset_names = df.columns.tolist()
    
    print(f"  ✅ 数据形状: {returns.shape}")
    print(f"  ✅ 资产数量: {len(asset_names)}")
    print(f"  ✅ 时间范围: {df.index[0]} 至 {df.index[-1]}")
    
    return returns, asset_names


def prepare_dataset(returns, lookback, gap, horizon, train_ratio=0.8):
    """准备数据集
    
    Parameters
    ----------
    returns : np.ndarray
        收益率矩阵
    lookback : int
        回看窗口
    gap : int
        预测间隔
    horizon : int
        持有期
    train_ratio : float
        训练集比例
    
    Returns
    -------
    dataset : InRAMDataset
        数据集
    indices_train : list
        训练索引
    indices_test : list
        测试索引
    """
    n_timesteps, n_assets = returns.shape
    
    # 滚动窗口
    X_list, y_list = [], []
    for i in range(lookback, n_timesteps - horizon - gap + 1):
        X_list.append(returns[i - lookback: i, :])
        y_list.append(returns[i + gap: i + gap + horizon, :])
    
    X = np.stack(X_list, axis=0)[:, None, ...]
    y = np.stack(y_list, axis=0)[:, None, ...]
    
    n_samples = X.shape[0]
    split_ix = int(n_samples * train_ratio)
    
    indices_train = list(range(split_ix))
    indices_test = list(range(split_ix, n_samples))
    
    # 标准化
    means, stds = prepare_standard_scaler(X, indices=indices_train)
    dataset = InRAMDataset(X, y, transform=Scale(means, stds))
    
    print(f"\n📊 数据集准备完成:")
    print(f"  - 样本数: {n_samples}")
    print(f"  - 训练集: {len(indices_train)}")
    print(f"  - 测试集: {len(indices_test)}")
    
    return dataset, indices_train, indices_test, means, stds


def train_model(network, loss_fn, dataloader_train, dataloader_test, epochs, lr):
    """训练模型"""
    print(f"\n🏋️ 开始训练 ({epochs} epochs)...")
    
    run = Run(
        network,
        loss_fn,
        dataloader_train,
        val_dataloaders={'test': dataloader_test},
        optimizer=torch.optim.Adam(network.parameters(), lr=lr, amsgrad=True),
        callbacks=[EarlyStoppingCallback(
            metric_name='loss',
            dataloader_name='test',
            patience=15
        )]
    )
    
    history = run.launch(epochs)
    
    print("✅ 训练完成！")
    return history


def evaluate_model(network, dataloader_test, output_dir):
    """评估模型"""
    print("\n📊 评估模型...")
    
    network = network.eval()
    
    benchmarks = {
        '1overN': OneOverN(),
        'network': network
    }
    
    metrics = {
        'MaxDD': MaximumDrawdown(),
        'Sharpe': SharpeRatio(),
        'MeanReturn': MeanReturns()
    }
    
    metrics_table = generate_metrics_table(benchmarks, dataloader_test, metrics)
    
    # 可视化
    fig = plot_metrics(metrics_table)
    plt.savefig(output_dir / "metrics.png", dpi=150, bbox_inches='tight')
    print(f"  ✅ 保存指标图: {output_dir / 'metrics.png'}")
    
    return metrics_table


def main():
    parser = argparse.ArgumentParser(description='训练深度学习投资组合优化模型')
    parser.add_argument('--data', type=str, required=True, help='CSV 数据文件路径')
    parser.add_argument('--lookback', type=int, default=40, help='回看窗口长度')
    parser.add_argument('--horizon', type=int, default=20, help='持有期长度')
    parser.add_argument('--gap', type=int, default=2, help='预测间隔')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--hidden-size', type=int, default=32, help='隐藏层大小')
    parser.add_argument('--output', type=str, default='outputs', help='输出目录')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 deepdow 训练脚本 - 真实数据")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载数据
    returns, asset_names = load_data(args.data)
    n_assets = len(asset_names)
    
    # 准备数据集
    dataset, indices_train, indices_test, means, stds = prepare_dataset(
        returns, args.lookback, args.gap, args.horizon
    )
    
    # 创建数据加载器
    dataloader_train = RigidDataLoader(dataset, indices=indices_train, batch_size=args.batch_size)
    dataloader_test = RigidDataLoader(dataset, indices=indices_test, batch_size=args.batch_size)
    
    # 创建网络
    print(f"\n🧠 创建网络 (hidden_size={args.hidden_size})...")
    network = PortfolioNet(
        n_channels=1,
        n_assets=n_assets,
        lookback=args.lookback,
        hidden_size=args.hidden_size
    )
    
    # 定义损失
    loss_fn = MaximumDrawdown() + 2 * MeanReturns() + SharpeRatio()
    
    # 训练
    history = train_model(network, loss_fn, dataloader_train, dataloader_test, args.epochs, args.lr)
    
    # 评估
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    metrics_table = evaluate_model(network, dataloader_test, output_dir)
    
    # 保存模型
    model_path = output_dir / "model.pth"
    torch.save({
        'model_state_dict': network.state_dict(),
        'n_assets': n_assets,
        'lookback': args.lookback,
        'hidden_size': args.hidden_size,
        'means': means,
        'stds': stds,
        'asset_names': asset_names
    }, model_path)
    print(f"\n💾 保存模型: {model_path}")
    
    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
