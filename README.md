# Industrial Diffusion Gen

基于扩散模型（Diffusion Model）的工业数据生成项目。

## 目标

利用扩散模型学习工业场景数据的分布，生成可用于训练、测试或数据增强的合成工业数据（如传感器时序、设备状态、缺陷图像等）。

## 技术方向

- **扩散模型**：DDPM / DDIM / Latent Diffusion 等
- **工业数据**：时序信号、表格特征、图像/点云等模态
- **应用场景**：小样本数据增强、异常检测基准、仿真数据补充

## 项目结构（规划）

```
industrial-diffusion-gen/
├── configs/        # 训练与推理配置
├── data/           # 数据集与预处理
├── models/         # 扩散模型与网络结构
├── scripts/        # 训练、采样、评估脚本
└── outputs/        # 生成结果与检查点
```

## 环境

待补充（Python / PyTorch 等）。

## 许可证

待定
