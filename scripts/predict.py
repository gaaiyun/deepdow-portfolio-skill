"""
Predict - 模型预测脚本

使用训练好的模型进行资产配置预测
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

try:
    from deepdow.data import InRAMDataset, RigidDataLoader, Scale
    from deepdow.visualize import generate_weights_table, plot_weight_heatmap
    import matplotlib.pyplot as plt
except ImportError:
    print(" deepdow 未安装！请运行: pip install deepdow")
    exit(1)


def load_model(model_path):
    """加载模型"""
    print(f" 加载模型: {model_path}")
    
    checkpoint = torch.load(model_path)
    
    # 重建网络
    from train import PortfolioNet
    
    network = PortfolioNet(
        n_channels=1,
        n_assets=checkpoint['n_assets'],
        lookback=checkpoint['lookback'],
        hidden_size=checkpoint['hidden_size']
    )
    
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()
    
    print(f"   模型加载成功")
    print(f"  - 资产数: {checkpoint['n_assets']}")
    print(f"  - 回看窗口: {checkpoint['lookback']}")
    
    return network, checkpoint


def load_test_data(data_path, lookback):
    """加载测试数据"""
    print(f"\n 加载测试数据: {data_path}")
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    returns = df.values
    asset_names = df.columns.tolist()
    dates = df.index
    
    print(f"   数据形状: {returns.shape}")
    print(f"   时间范围: {dates[0]} 至 {dates[-1]}")
    
    # 准备预测数据
    X_list = []
    valid_dates = []
    
    for i in range(lookback, len(returns)):
        X_list.append(returns[i - lookback: i, :])
        valid_dates.append(dates[i])
    
    X = np.stack(X_list, axis=0)[:, None, ...]
    
    return X, asset_names, valid_dates


def predict(network, X, means, stds):
    """预测权重"""
    print("\n 开始预测...")
    
    # 标准化
    X_scaled = (X - means) / stds
    X_tensor = torch.from_numpy(X_scaled).float()
    
    # 预测
    with torch.no_grad():
        weights = network(X_tensor)
    
    weights_np = weights.numpy()
    
    print(f"   预测完成，生成 {len(weights_np)} 个配置")
    
    return weights_np


def save_predictions(weights, asset_names, dates, output_path):
    """保存预测结果"""
    df = pd.DataFrame(weights, columns=asset_names, index=dates)
    df.to_csv(output_path)
    print(f"\n 保存预测结果: {output_path}")
    
    # 显示统计
    print("\n 预测统计:")
    print(df.describe())
    
    return df


def visualize_predictions(weights_df, output_dir):
    """可视化预测结果"""
    print("\n 生成可视化...")
    
    # 权重热图
    plt.figure(figsize=(14, 8))
    plt.imshow(weights_df.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(label='Weight')
    plt.xlabel('Time')
    plt.ylabel('Asset')
    plt.title('Portfolio Weights Over Time')
    plt.yticks(range(len(weights_df.columns)), weights_df.columns)
    plt.tight_layout()
    plt.savefig(output_dir / "weights_heatmap.png", dpi=150, bbox_inches='tight')
    print(f"   保存权重热图: {output_dir / 'weights_heatmap.png'}")
    
    # 权重分布
    plt.figure(figsize=(12, 6))
    weights_df.plot(kind='area', stacked=True, alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Weight')
    plt.title('Portfolio Allocation Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / "weights_area.png", dpi=150, bbox_inches='tight')
    print(f"   保存权重分布图: {output_dir / 'weights_area.png'}")
    
    # 集中度分析
    concentration = (weights_df ** 2).sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    concentration.plot()
    plt.axhline(y=1/len(weights_df.columns), color='r', linestyle='--', label='Equal Weight')
    plt.xlabel('Time')
    plt.ylabel('Concentration (HHI)')
    plt.title('Portfolio Concentration Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "concentration.png", dpi=150, bbox_inches='tight')
    print(f"   保存集中度图: {output_dir / 'concentration.png'}")


def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型进行预测')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--data', type=str, required=True, help='测试数据 CSV 文件')
    parser.add_argument('--output', type=str, default='predictions', help='输出目录')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(" deepdow 预测脚本")
    print("=" * 60)
    
    # 加载模型
    network, checkpoint = load_model(args.model)
    
    # 加载测试数据
    X, asset_names, dates = load_test_data(args.data, checkpoint['lookback'])
    
    # 检查资产名称
    if asset_names != checkpoint['asset_names']:
        print("\n  警告: 测试数据资产与训练数据不一致！")
        print(f"  训练: {checkpoint['asset_names']}")
        print(f"  测试: {asset_names}")
    
    # 预测
    weights = predict(network, X, checkpoint['means'], checkpoint['stds'])
    
    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    weights_df = save_predictions(weights, asset_names, dates, output_dir / "predictions.csv")
    
    # 可视化
    visualize_predictions(weights_df, output_dir)
    
    print("\n" + "=" * 60)
    print(" 预测完成！")
    print("=" * 60)
    print(f"\n 输出文件:")
    print(f"  - 预测结果: {output_dir / 'predictions.csv'}")
    print(f"  - 权重热图: {output_dir / 'weights_heatmap.png'}")
    print(f"  - 权重分布: {output_dir / 'weights_area.png'}")
    print(f"  - 集中度图: {output_dir / 'concentration.png'}")


if __name__ == "__main__":
    main()
