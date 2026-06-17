# LeNet — FashionMNIST 图像分类 🧥👗👕

基于 PyTorch 实现的经典 **LeNet-5** 卷积神经网络，用于 **FashionMNIST** 服饰图像分类。

## 项目结构

```
lenet/
├── model.py          # LeNet-5 网络结构定义
├── model_train.py    # 模型训练脚本
├── model_test.py     # 模型测试 & 推理脚本
├── plot.py           # 数据集可视化预览
├── best_model.pth    # 已训练好的模型权重
└── data/             # FashionMNIST 数据集（自动下载）
```

## 网络架构

经典的 LeNet-5 结构，针对 28×28 灰度图设计：

```
Conv2d(1→6, kernel=5, padding=2) → Sigmoid → AvgPool(2)
Conv2d(6→16, kernel=5)           → Sigmoid → AvgPool(2)
Flatten → Linear(400→120) → Linear(120→84) → Linear(84→10)
```

| 层 | 类型 | 输入 | 输出 | 参数量 |
|----|------|------|------|--------|
| C1 | Conv2d + Sigmoid | 1×28×28 | 6×28×28 | 156 |
| S2 | AvgPool2d | 6×28×28 | 6×14×14 | — |
| C3 | Conv2d + Sigmoid | 6×14×14 | 16×10×10 | 2,416 |
| S4 | AvgPool2d | 16×10×10 | 16×5×5 | — |
| F5 | Linear | 400 | 120 | 48,120 |
| F6 | Linear | 120 | 84 | 10,164 |
| F7 | Linear | 84 | 10 | 850 |

**总参数量：约 6 万**

## 数据集

[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) — 10 类服饰灰度图像（28×28）：

| 标签 | 类别 | 标签 | 类别 |
|------|------|------|------|
| 0 | T恤 | 5 | 凉鞋 |
| 1 | 裤子 | 6 | 衬衫 |
| 2 | 套头衫 | 7 | 运动鞋 |
| 3 | 连衣裙 | 8 | 包 |
| 4 | 外套 | 9 | 短靴 |

- 训练集：60,000 张（按 8:2 划分为训练/验证）
- 测试集：10,000 张

## 快速开始

### 环境要求

- Python ≥ 3.8
- PyTorch ≥ 2.0
- torchvision
- matplotlib（可视化用）

```bash
pip install torch torchvision matplotlib pandas
```

### 训练模型

```bash
python model_train.py
```

训练 20 个 epoch，自动保存最优模型到 `best_model.pth`，训练完成后显示 Loss 和 Accuracy 曲线。

### 测试模型

```bash
python model_test.py
```

加载 `best_model.pth`，输出测试集整体准确率，并逐样本打印预测值与真实标签对比。

## 训练曲线示例

训练过程中会实时输出每个 epoch 的 loss 和 accuracy：

```
Epoch 0/19
----------
train loss: 0.4621  train acc: 0.8301
val loss:   0.3892  val acc:   0.8613
...
```

最终自动绘制训练/验证的 Loss 和 Accuracy 双曲线图。

## 技术要点

- **优化器**：Adam（lr=0.001）
- **损失函数**：CrossEntropyLoss
- **模型保存策略**：保留验证集准确率最高的权重（`best_model_wts`）
- **设备兼容**：自动检测 CUDA GPU / CPU

## License

MIT
