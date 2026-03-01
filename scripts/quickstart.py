"""
Quickstart - Deep Learning Portfolio Optimization (Simplified Version)

快速开始示例：不依赖 cvxpylayers 的简化版本
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


class SimpleAllocator(torch.nn.Module):
    """简单的 Softmax 分配器"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x, temperature=1.0):
        """Softmax 分配
        
        Parameters
        ----------
        x : torch.Tensor
            原始分数，形状 (n_samples, n_assets)
        temperature : float
            温度参数
        
        Returns
        -------
        weights : torch.Tensor
            归一化权重，形状 (n_samples, n_assets)
        """
        return torch.softmax(x / temperature, dim=1)


class SimpleNet(torch.nn.Module):
    """简单的投资组合优化网络"""
    
    def __init__(self, n_assets, lookback, p=0.5):
        super().__init__()
        n_features = n_assets * lookback
        
        self.dropout = torch.nn.Dropout(p=p)
        self.dense = torch.nn.Linear(n_features, n_assets, bias=True)
        self.allocate = SimpleAllocator()
        self.temperature = torch.nn.Parameter(torch.ones(1), requires_grad=True)
    
    def forward(self, x):
        """前向传播
        
        Parameters
        ----------
        x : torch.Tensor
            形状 (n_samples, 1, lookback, n_assets)
        
        Returns
        -------
        weights : torch.Tensor
            形状 (n_samples, n_assets)
        """
        n_samples = x.shape[0]
        x = x.view(n_samples, -1)  # 展平特征
        x = self.dropout(x)
        x = self.dense(x)
        
        weights = self.allocate(x, self.temperature)
        
        return weights


def generate_synthetic_data(n_timesteps=1000, n_assets=20, lookback=40, gap=2, horizon=20):
    """生成合成数据"""
    print(f"📊 生成合成数据: {n_timesteps} 时间步, {n_assets} 资产")
    
    # 生成正弦波收益率
    returns = np.zeros((n_timesteps, n_assets))
    for i in range(n_assets):
        freq = 1 / np.random.randint(3, lookback)
        phase = np.random.randint(0, lookback)
        t = np.arange(n_timesteps)
        returns[:, i] = 0.05 * np.sin(2 * np.pi * freq * t + phase)
    
    # 添加噪声
    returns += np.random.normal(scale=0.02, size=returns.shape)
    
    # 滚动窗口
    X_list, y_list = [], []
    for i in range(lookback, n_timesteps - horizon - gap + 1):
        X_list.append(returns[i - lookback: i, :])
        y_list.append(returns[i + gap: i + gap + horizon, :])
    
    X = np.stack(X_list, axis=0)[:, None, ...]
    y = np.stack(y_list, axis=0)[:, None, ...]
    
    print(f"✅ 数据形状: X={X.shape}, y={y.shape}")
    return X, y


def sharpe_ratio_loss(weights, returns):
    """Sharpe Ratio 损失（最小化负 Sharpe）
    
    Parameters
    ----------
    weights : torch.Tensor
        权重，形状 (n_samples, n_assets)
    returns : torch.Tensor
        收益率，形状 (n_samples, 1, horizon, n_assets)
    
    Returns
    -------
    loss : torch.Tensor
        标量损失
    """
    # 计算组合收益
    returns_squeezed = returns.squeeze(1)  # (n_samples, horizon, n_assets)
    portfolio_returns = torch.sum(weights.unsqueeze(1) * returns_squeezed, dim=2)  # (n_samples, horizon)
    
    # 计算 Sharpe Ratio
    mean_return = portfolio_returns.mean(dim=1)
    std_return = portfolio_returns.std(dim=1) + 1e-6
    sharpe = mean_return / std_return
    
    # 返回负 Sharpe（最小化）
    return -sharpe.mean()


def train_model(network, X_train, y_train, X_test, y_test, epochs=30, lr=0.001):
    """训练模型"""
    print(f"\n🏋️ 开始训练 ({epochs} epochs)...")
    
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, amsgrad=True)
    
    train_losses = []
    test_losses = []
    
    best_test_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练
        network.train()
        optimizer.zero_grad()
        
        weights = network(X_train)
        loss = sharpe_ratio_loss(weights, y_train)
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # 验证
        network.eval()
        with torch.no_grad():
            weights_test = network(X_test)
            test_loss = sharpe_ratio_loss(weights_test, y_test)
            test_losses.append(test_loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
        
        # 早停
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⚠️ 早停于 epoch {epoch+1}")
                break
    
    print("✅ 训练完成！")
    return train_losses, test_losses


def evaluate_model(network, X_test, y_test):
    """评估模型"""
    print("\n📊 评估模型...")
    
    network.eval()
    with torch.no_grad():
        weights = network(X_test)
        
        # 计算组合收益
        returns_squeezed = y_test.squeeze(1)
        portfolio_returns = torch.sum(weights.unsqueeze(1) * returns_squeezed, dim=2)
        
        # 统计
        mean_return = portfolio_returns.mean().item()
        std_return = portfolio_returns.std().item()
        sharpe = mean_return / (std_return + 1e-6)
        
        # 最大回撤
        cumulative = torch.cumsum(portfolio_returns, dim=1)
        running_max = torch.cummax(cumulative, dim=1)[0]
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min().item()
        
        print(f"  ✅ 平均收益: {mean_return:.4f}")
        print(f"  ✅ 波动率: {std_return:.4f}")
        print(f"  ✅ Sharpe Ratio: {sharpe:.4f}")
        print(f"  ✅ 最大回撤: {max_drawdown:.4f}")
        
        return weights, portfolio_returns


def visualize_results(train_losses, test_losses, weights, output_dir):
    """可视化结果"""
    print("\n📈 生成可视化...")
    
    output_dir.mkdir(exist_ok=True)
    
    # 1. 训练历史
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(test_losses, label='Test Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Negative Sharpe)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "training_history.png", dpi=150, bbox_inches='tight')
    print(f"  ✅ 保存训练历史: {output_dir / 'training_history.png'}")
    
    # 2. 权重热图
    weights_np = weights.numpy()
    plt.figure(figsize=(14, 8))
    plt.imshow(weights_np.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(label='Weight')
    plt.xlabel('Sample')
    plt.ylabel('Asset')
    plt.title('Portfolio Weights')
    plt.savefig(output_dir / "weights.png", dpi=150, bbox_inches='tight')
    print(f"  ✅ 保存权重图: {output_dir / 'weights.png'}")
    
    # 3. 权重分布
    plt.figure(figsize=(10, 6))
    plt.hist(weights_np.flatten(), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('Weight Distribution')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "weight_distribution.png", dpi=150, bbox_inches='tight')
    print(f"  ✅ 保存权重分布: {output_dir / 'weight_distribution.png'}")


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 deepdow 快速开始 - 深度学习投资组合优化（简化版）")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(4)
    np.random.seed(5)
    
    # 参数设置
    n_timesteps, n_assets = 1000, 20
    lookback, gap, horizon = 40, 2, 20
    n_samples = n_timesteps - lookback - horizon - gap + 1
    
    # 训练/测试分割
    split_ix = int(n_samples * 0.8)
    
    print(f"\n📈 参数配置:")
    print(f"  - 回看窗口: {lookback}")
    print(f"  - 预测间隔: {gap}")
    print(f"  - 持有期: {horizon}")
    print(f"  - 训练样本: {split_ix}")
    print(f"  - 测试样本: {n_samples - split_ix}")
    
    # 生成数据
    X, y = generate_synthetic_data(n_timesteps, n_assets, lookback, gap, horizon)
    
    # 数据标准化
    print("\n🔧 数据标准化...")
    X_train = X[:split_ix]
    means = X_train.mean(axis=(0, 2), keepdims=True)
    stds = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    X = (X - means) / stds
    
    # 转换为 Tensor
    X_train = torch.from_numpy(X[:split_ix]).float()
    y_train = torch.from_numpy(y[:split_ix]).float()
    X_test = torch.from_numpy(X[split_ix:]).float()
    y_test = torch.from_numpy(y[split_ix:]).float()
    
    # 创建网络
    print("\n🧠 创建神经网络...")
    network = SimpleNet(n_assets, lookback, p=0.5)
    print(network)
    
    # 训练
    train_losses, test_losses = train_model(network, X_train, y_train, X_test, y_test, epochs=30, lr=0.001)
    
    # 评估
    weights, portfolio_returns = evaluate_model(network, X_test, y_test)
    
    # 可视化
    output_dir = Path("outputs")
    visualize_results(train_losses, test_losses, weights, output_dir)
    
    # 保存模型
    model_path = output_dir / "model.pth"
    torch.save({
        'model_state_dict': network.state_dict(),
        'n_assets': n_assets,
        'lookback': lookback,
        'means': means,
        'stds': stds
    }, model_path)
    print(f"\n💾 保存模型: {model_path}")
    
    # 总结
    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)
    print(f"\n📁 输出文件:")
    print(f"  - 模型: {model_path}")
    print(f"  - 训练历史: {output_dir / 'training_history.png'}")
    print(f"  - 权重图: {output_dir / 'weights.png'}")
    print(f"  - 权重分布: {output_dir / 'weight_distribution.png'}")
    print("\n🎉 快速开始示例运行成功！")
    print("\n💡 提示: 这是简化版本，不依赖 cvxpylayers")
    print("   完整功能请安装: pip install cvxpylayers")


if __name__ == "__main__":
    main()
