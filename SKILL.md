# deepdow - Deep Learning Portfolio Optimization

**深度学习驱动的投资组合优化框架**

基于 deepdow (1111⭐) 的 OpenClaw Skill 实现，将深度学习与投资组合优化无缝结合。

## 🎯 核心功能

- **端到端可微分优化** - 从特征提取到资产配置全流程可训练
- **PyTorch 集成** - 利用深度学习强大的表征能力
- **凸优化层** - 集成 cvxpylayers 实现 Markowitz、风险预算等经典方法
- **丰富的损失函数** - Sharpe Ratio、Maximum Drawdown、Mean Returns 等
- **回测框架** - 完整的训练-验证-测试流程

## 📦 依赖安装

```bash
pip install deepdow torch numpy pandas matplotlib seaborn
```

核心依赖：
- `torch>=1.5` - PyTorch 深度学习框架
- `cvxpylayers` - 可微分凸优化层
- `numpy>=1.16.5` - 数值计算
- `pandas` - 数据处理
- `matplotlib`, `seaborn` - 可视化

## 🚀 使用方法

### 1. 快速开始 - 合成数据训练

```bash
python scripts/quickstart.py
```

使用正弦波合成数据演示完整训练流程：
- 数据生成与预处理
- 网络定义与训练
- 性能评估与可视化

### 2. 真实数据训练

```bash
python scripts/train.py --data your_data.csv --lookback 40 --horizon 20 --epochs 50
```

参数说明：
- `--data` - CSV 数据文件路径（格式：日期 × 资产收益率）
- `--lookback` - 回看窗口长度（默认 40）
- `--horizon` - 持有期长度（默认 20）
- `--gap` - 预测间隔（默认 2）
- `--epochs` - 训练轮数（默认 30）
- `--batch-size` - 批次大小（默认 32）
- `--lr` - 学习率（默认 0.001）

### 3. 模型预测

```bash
python scripts/predict.py --model saved_model.pth --data test_data.csv
```

加载训练好的模型进行资产配置预测。

## 🏗️ 网络架构

### 内置网络

1. **SimpleNet** - 基础全连接网络
   - Dropout + Dense + Softmax Allocator
   - 适合快速原型验证

2. **BachelierNet** - RNN + 凸优化
   - RNN 特征提取
   - Attention 时间聚合
   - Markowitz 优化分配

3. **CustomNet** - 自定义架构
   - 灵活组合各种层
   - 支持多种分配策略

### 可用层（deepdow.layers）

**特征提取**：
- `Conv` - 卷积层
- `RNN` - 循环神经网络
- `AttentionCollapse` - 注意力池化
- `AverageCollapse` - 平均池化

**资产配置**：
- `SoftmaxAllocator` - Softmax 分配
- `AnalyticalMarkowitz` - 解析 Markowitz
- `NumericalMarkowitz` - 数值 Markowitz
- `NCO` - 嵌套聚类优化
- `NumericalRiskBudgeting` - 风险预算

**辅助层**：
- `CovarianceMatrix` - 协方差矩阵估计
- `WeightNorm` - 权重归一化
- `KMeans` - K均值聚类

## 📊 损失函数

所有损失函数设计为**最小化目标**：

- `MeanReturns()` - 最大化平均收益（取负）
- `SharpeRatio()` - 最大化夏普比率（取负）
- `MaximumDrawdown()` - 最小化最大回撤
- `StandardDeviation()` - 最小化波动率
- `SquaredWeights()` - L2 正则化（鼓励分散化）

**组合损失**：
```python
loss = MaximumDrawdown() + 2 * MeanReturns() + SharpeRatio()
```

## 📈 评估与可视化

### 性能指标

```python
from deepdow.visualize import generate_metrics_table, plot_metrics

metrics = {
    'Sharpe': SharpeRatio(),
    'MaxDD': MaximumDrawdown(),
    'MeanReturn': MeanReturns()
}

metrics_table = generate_metrics_table(benchmarks, dataloader, metrics)
plot_metrics(metrics_table)
```

### 权重热图

```python
from deepdow.visualize import generate_weights_table, plot_weight_heatmap

weight_table = generate_weights_table(network, dataloader)
plot_weight_heatmap(weight_table, add_sum_column=True)
```

## 🎓 示例场景

### 场景 1：多因子股票组合

```python
# 使用技术指标作为输入特征
# X: (n_samples, n_channels, lookback, n_assets)
# n_channels 可以是：收益率、动量、波动率等

network = BachelierNet(
    n_input_channels=5,  # 5个因子
    n_assets=50,         # 50只股票
    hidden_size=32,
    max_weight=0.1       # 单只股票最大10%
)
```

### 场景 2：加密货币组合

```python
# 高波动率资产，使用风险预算
from deepdow.layers import NumericalRiskBudgeting

class CryptoNet(torch.nn.Module):
    def __init__(self, n_assets):
        super().__init__()
        self.rnn = RNN(n_channels=1, hidden_size=16)
        self.collapse = AttentionCollapse(n_channels=16)
        self.allocate = NumericalRiskBudgeting()
    
    def forward(self, x):
        features = self.rnn(x)
        collapsed = self.collapse(features)
        # ... 计算协方差矩阵
        weights = self.allocate(covmat)
        return weights
```

### 场景 3：行业轮动

```python
# 使用聚类进行行业分组
from deepdow.layers import KMeans, NCO

network = NCO(
    n_input_channels=3,
    n_clusters=5,  # 5个行业簇
    n_init=10
)
```

## 🔧 高级配置

### 数据加载策略

**RigidDataLoader** - 固定形状数据：
```python
dataloader = RigidDataLoader(
    dataset,
    indices=train_indices,
    batch_size=32,
    shuffle=True
)
```

**FlexibleDataLoader** - 可变形状数据：
```python
dataloader = FlexibleDataLoader(
    dataset,
    indices=train_indices,
    batch_size=32
)
```

### 训练回调

```python
from deepdow.callbacks import EarlyStoppingCallback, ModelCheckpoint

callbacks = [
    EarlyStoppingCallback(
        metric_name='loss',
        dataloader_name='val',
        patience=15
    ),
    ModelCheckpoint(
        save_dir='./checkpoints',
        metric_name='sharpe',
        mode='max'
    )
]
```

### MLflow 集成

```python
from deepdow.callbacks import MLFlowCallback

run = Run(
    network,
    loss,
    dataloader_train,
    callbacks=[MLFlowCallback(experiment_name='portfolio_opt')]
)
```

## 📚 理论背景

### 端到端优化

传统方法分两步：
1. 预测未来收益/协方差（LSTM、GARCH等）
2. 求解优化问题（凸优化）

deepdow 将两步合并为一个可微分网络：
```
输入特征 → 神经网络 → 分配层 → 权重
         (特征提取)   (优化)
```

整个流程通过梯度下降端到端训练。

### 可微分凸优化

使用 cvxpylayers 将凸优化问题嵌入神经网络：
- 前向传播：求解优化问题
- 反向传播：通过隐式函数定理计算梯度

例如 Markowitz 优化：
```
minimize    w^T Σ w
subject to  w^T μ >= r_min
            sum(w) = 1
            w >= 0
```

## ⚠️ 注意事项

1. **数据质量**：
   - 确保收益率数据无缺失值
   - 检查异常值和极端波动
   - 考虑数据标准化

2. **过拟合风险**：
   - 使用 Dropout 正则化
   - 早停策略
   - 交叉验证

3. **计算资源**：
   - 凸优化层计算密集
   - 建议使用 GPU 加速
   - 批次大小影响内存占用

4. **回测偏差**：
   - 避免前视偏差（look-ahead bias）
   - 考虑交易成本
   - 滑点和流动性约束

## 🔗 相关资源

- **原始仓库**: https://github.com/jankrepl/deepdow
- **文档**: https://deepdow.readthedocs.io
- **论文引用**: 见 GitHub README DOI badge

## 📝 引用

如果使用 deepdow 进行研究，请引用原始项目（通过 Zenodo DOI）。

---

**Skill 维护者**: P-Box 无限编程助手  
**最后更新**: 2026-03-01  
**版本**: 1.0.0
