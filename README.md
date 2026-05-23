> **维护状态说明**：本仓库当前是 AI 辅助生成的初始脚手架，未在生产环境持续打磨。代码可作为参考与起点，使用前请自行核对接口、依赖与边界条件。如果你打算接手维护、把它合并到其他项目，或者发现 bug，欢迎开 issue 或 PR。
# deepdow - Deep Learning Portfolio Optimization Skill

[![deepdow](https://img.shields.io/badge/deepdow-1111⭐-blue)](https://github.com/jankrepl/deepdow)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.5+-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**将深度学习与投资组合优化无缝结合的 OpenClaw Skill**

基于 [deepdow](https://github.com/jankrepl/deepdow) 框架（1111⭐），提供端到端可微分的投资组合优化解决方案。

## ✨ 特性

- 🧠 **深度学习驱动** - PyTorch 神经网络自动学习最优配置策略
- 📈 **端到端优化** - 从特征提取到资产分配全流程可训练
- 🎯 **凸优化集成** - 内置 Markowitz、风险预算等经典方法
- 📊 **丰富损失函数** - Sharpe Ratio、Maximum Drawdown、Mean Returns
- 🔄 **完整回测** - 训练-验证-测试完整流程
- 🎨 **可视化工具** - 权重热图、性能指标图表

## 🚀 快速开始

### 安装依赖

```bash
pip install deepdow torch numpy pandas matplotlib seaborn
```

### 运行示例

```bash
# 1. 快速开始 - 合成数据训练
python scripts/quickstart.py

# 2. 真实数据训练
python scripts/train.py --data your_data.csv --lookback 40 --horizon 20

# 3. 模型预测
python scripts/predict.py --model saved_model.pth --data test_data.csv
```

## 📖 使用场景

### 场景 1：多因子股票组合

```python
from deepdow.nn import BachelierNet

# 使用5个技术因子预测50只股票的配置
network = BachelierNet(
    n_input_channels=5,  # 收益率、动量、波动率等
    n_assets=50,
    hidden_size=32,
    max_weight=0.1  # 单只股票最大10%
)
```

### 场景 2：加密货币组合

```python
from deepdow.layers import NumericalRiskBudgeting

# 高波动率资产使用风险预算策略
class CryptoNet(torch.nn.Module):
    def __init__(self, n_assets):
        super().__init__()
        self.rnn = RNN(n_channels=1, hidden_size=16)
        self.allocate = NumericalRiskBudgeting()
```

### 场景 3：行业轮动

```python
from deepdow.layers import NCO

# 使用嵌套聚类优化进行行业分组
network = NCO(
    n_input_channels=3,
    n_clusters=5,  # 5个行业簇
    n_init=10
)
```

## 🏗️ 架构概览

```
输入数据 (n_samples, n_channels, lookback, n_assets)
    ↓
特征提取层 (RNN/Conv/Attention)
    ↓
时间聚合层 (AttentionCollapse/AverageCollapse)
    ↓
协方差估计 (CovarianceMatrix)
    ↓
资产配置层 (Markowitz/NCO/Softmax)
    ↓
输出权重 (n_samples, n_assets)
```

## 📊 核心组件

### 1. 数据加载

- **RigidDataLoader** - 固定形状数据（推荐）
- **FlexibleDataLoader** - 可变形状数据
- **InRAMDataset** - 内存数据集
- **Scale** - 数据标准化

### 2. 神经网络层

**特征提取**：
- `RNN` - 循环神经网络
- `Conv` - 卷积层
- `AttentionCollapse` - 注意力池化

**资产配置**：
- `SoftmaxAllocator` - Softmax 分配
- `AnalyticalMarkowitz` - 解析 Markowitz
- `NumericalMarkowitz` - 数值 Markowitz
- `NCO` - 嵌套聚类优化
- `NumericalRiskBudgeting` - 风险预算

### 3. 损失函数

所有损失函数设计为最小化目标：

```python
from deepdow.losses import SharpeRatio, MaximumDrawdown, MeanReturns

# 组合多个目标
loss = MaximumDrawdown() + 2 * MeanReturns() + SharpeRatio()
```

### 4. 训练框架

```python
from deepdow.experiments import Run
from deepdow.callbacks import EarlyStoppingCallback

run = Run(
    network,
    loss,
    dataloader_train,
    val_dataloaders={'test': dataloader_test},
    optimizer=torch.optim.Adam(network.parameters()),
    callbacks=[EarlyStoppingCallback(patience=15)]
)

history = run.launch(epochs=30)
```

### 5. 可视化

```python
from deepdow.visualize import plot_metrics, plot_weight_heatmap

# 性能指标图
plot_metrics(metrics_table)

# 权重热图
plot_weight_heatmap(weight_table, add_sum_column=True)
```

## 📁 项目结构

```
deepdow/
├── scripts/
│   ├── quickstart.py      # 快速开始示例
│   ├── train.py           # 训练脚本
│   ├── predict.py         # 预测脚本
│   └── utils.py           # 工具函数
├── examples/
│   ├── stock_portfolio.py # 股票组合示例
│   ├── crypto_portfolio.py # 加密货币示例
│   └── sector_rotation.py # 行业轮动示例
├── SKILL.md               # 详细文档
├── README.md              # 本文件
└── requirements.txt       # 依赖列表
```

## 🎓 理论背景

### 端到端优化

传统投资组合优化分两步：
1. **预测** - 使用 LSTM/GARCH 预测收益/协方差
2. **优化** - 求解凸优化问题

deepdow 将两步合并为一个可微分网络，通过梯度下降端到端训练。

### 可微分凸优化

使用 `cvxpylayers` 将凸优化嵌入神经网络：
- **前向传播**：求解优化问题
- **反向传播**：通过隐式函数定理计算梯度

## ⚠️ 注意事项

1. **数据质量**
   - 确保无缺失值和异常值
   - 考虑数据标准化
   - 检查极端波动

2. **过拟合风险**
   - 使用 Dropout 正则化
   - 早停策略
   - 交叉验证

3. **计算资源**
   - 凸优化层计算密集
   - 建议使用 GPU
   - 注意批次大小

4. **回测偏差**
   - 避免前视偏差
   - 考虑交易成本
   - 滑点和流动性约束

## 📚 参考资源

- **原始仓库**: https://github.com/jankrepl/deepdow
- **文档**: https://deepdow.readthedocs.io
- **论文**: 见 GitHub DOI badge

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

Apache License 2.0 - 见 [LICENSE](LICENSE) 文件

---

**维护者**: P-Box 无限编程助手  
**创建日期**: 2026-03-01  
**版本**: 1.0.0
