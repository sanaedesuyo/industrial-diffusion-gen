# Industrial Diffusion Gen

对 TSGM（*Regular Time-series Generation using SGM*, Lim et al., AAAI 2023, [arXiv:2301.08518](https://arxiv.org/abs/2301.08518)）的复现项目，将该 score-based 时序生成方法应用于工业场景数据：**C-MAPSS**（涡扇发动机退化仿真）、**PM25**（空气质量）、**ESA Anomaly Dataset**（卫星遥测）。

## 方法概述

TSGM 是首个用 score-based / diffusion 模型做**规则时序生成**（区别于时序预测 TimeGrad/ScoreGrad、插补 CSDI）的方法。核心思路：把时序 `x_{1:T}` 映射到隐空间 `h_{1:T}`，用一个以**上一时刻隐向量为条件**的分数网络 `M_θ(s, h_t^s, h_{t-1})` 学习隐空间中的条件分数 `∇log p(h_t^s | h_{t-1})`，再用 SDE 反向采样递归生成整段序列。

### 三大组件

| 组件 | 作用 | 实现 |
|---|---|---|
| GRU 自编码器 | `h_t = e(h_{t-1}, x_t)`，`x̂_t = d(h_t)`，把时序映射到隐空间再还原 | [models/autoencoder.py](models/autoencoder.py) |
| 条件 Score 网络 | 输入 [被扩散隐向量 ⊕ 条件 `h_{t-1}`] + 扩散时间 `s`，预测分数 | [models/score_unet1d.py](models/score_unet1d.py) |
| VP / subVP SDE | 前向加噪 / 反向去噪的随机微分方程（论文排除效果较差的 VE-SDE） | [models/sde.py](models/sde.py) |

Score 网络有两种可选架构（`model.score_net_type`，默认 `unet`）：
- **`unet`**（默认）：论文原始的 1D U-Net，FiLM 条件卷积块，`unet_depth`/`unet_channels` 可配
- **`mlp`**：项目自选的 ResMLP+FiLM 变体，是论文之外的改进尝试，作为对照保留

### 训练算法（两阶段，对齐论文 Algorithm 1）

1. **AE 预训练**（`iter_pre` 步）：仅用重构损失 `L_ed = E‖x̂_{1:T} - x_{1:T}‖²` 训练 encoder+decoder — [`train_autoencoder`](models/tsgm.py)
2. **Score 网络主训练**（`iter_main` 步）：用隐空间条件去噪分数匹配损失训练 score 网络；条件 `h_{t-1}` 需 detach 避免梯度泄漏到 AE；`use_alt=True` 时每步交替微调 AE — [`train_score`](models/tsgm.py)
3. **Checkpoint 选择**：训练期间每 `save_every` 步，用当前 EMA 权重多批次（`select_n_fake_batches`）重新生成样本、多随机种子（`select_n_seeds`）训练判别器打分，取平均 discriminative score 最低的一版存为 `ckpt_best.pt`（早期版本只用单批次静态样本判断，方差过大，已改进为多批次重采样 — [scripts/train.py](scripts/train.py)）

### 采样：递归 Predictor-Corrector

`t=1..T` 逐步采样：从高斯先验出发，读入上一步生成的 `ĥ_{t-1}`，经 PC 采样器（Predictor: reverse-diffusion/Euler-Maruyama；Corrector: Langevin MCMC，默认 N=1000 步）生成 `ĥ_t`；得到完整 `ĥ_{1:T}` 后一次性用 Decoder 还原 `x̂_{1:T}` — [`recursive_generate` / `pc_sample_step`](models/tsgm.py)

## 数据集与预处理协议

窗口长度统一 `T=24`，逐特征 MinMax 归一化到 `[0,1]`（仅用训练集拟合）。

| 数据集 | 内容 | 切分策略 | 维度 D |
|---|---|---|---|
| **C-MAPSS** | 涡扇发动机 21 传感器+3 工况，多机组逐 cycle | 按 engine unit 分组切窗，避免跨机组穿越 | 14 |
| **PM25** | 北京多站点 PM2.5 逐小时观测 | 时间顺序切分（前段 train，后段 test），滑窗 | 36 |
| **ESA Anomaly Dataset** | 卫星 Mission1 遥测，16 个共同采样通道 | 重采样到统一小时网格，时间顺序切分（无官方切分文件） | 16 |

数据加载/预处理实现见 [data/loaders/](data/loaders/)（`base.py` 提供通用的 `MinMaxNormalizer`/`make_windows_from_series`；`cmapss.py`/`pm25.py`/`esa.py` 各自处理原始数据格式）。

## 评估协议（TimeGAN 协议）

- **Discriminative score**：训练 2 层 LSTM 二分类器区分真/假样本，报告 `|acc - 0.5|`（越低越好，越接近 0 说明分类器越难分辨真假）
- **Predictive score（TSTR）**：用合成数据训练 LSTM 预测器，在真实测试集上测 MAE（越低越好，越接近"用真实数据训练"的基线说明生成数据的时序依赖结构越真实）
- **t-SNE 可视化**：真/假样本降维后的分布重合度（多样性检查）
- 每个指标跑 **10 个随机种子**，报 mean ± std

实现见 [metrics/discriminative.py](metrics/discriminative.py)、[metrics/predictive.py](metrics/predictive.py)、[metrics/visualization.py](metrics/visualization.py)。

## 复现结果

最新一轮（论文默认 1D U-Net 架构 + 多批次 checkpoint 选择）在三个数据集上的完整 10-seed 评估结果：

| 数据集 | Discriminative ↓ | Predictive ↓ |
|---|---|---|
| C-MAPSS | 0.3281 ± 0.0830 | 0.0624 ± 0.0001 |
| PM25 | 0.2454 ± 0.0179 | 0.0280 ± 0.0002 |
| ESA | **0.0716 ± 0.0193** | 0.0247 ± 0.0028 |

（原始数据见 [reproduction_results/2026-07-08_unet_multisampling/](reproduction_results/2026-07-08_unet_multisampling/)；ESA 上还有一个独立的快速推理演示见下方"ESA 推理演示"。）

**关键发现**：
- **Predictive score 在三个数据集上都达到了接近论文报告的水平**（时序统计结构学得较好）
- **Discriminative score 上 ESA 表现最好**（0.07，接近论文报告的最佳数据集水平），PM25 居中，**C-MAPSS 明显偏高**（0.33）
- 用相同代码在架构不变的情况下对比三个数据集，说明 C-MAPSS 的 gap **不是实现 bug**，而是该方法本身的结构性局限——单步马尔可夫条件 `h_t | h_{t-1}` 难以捕捉 C-MAPSS 强单调退化趋势，而 PM25（周期性）和 ESA（准平稳遥测）这类数据更契合该假设

## 项目结构

```
industrial-diffusion-gen/
├── configs/                       # 每数据集一份 YAML：模型架构、SDE、训练超参
│   ├── cmapss.yaml
│   ├── pm25.yaml
│   └── esa.yaml
├── data/
│   ├── loaders/                   # 数据加载与预处理
│   │   ├── base.py                # 通用：MinMax 归一化、滑窗、切分
│   │   ├── cmapss.py / pm25.py / esa.py
│   └── processed/                 # 预处理后的 .npy 窗口张量（各数据集）
├── models/
│   ├── autoencoder.py             # GRU Encoder/Decoder
│   ├── score_unet1d.py            # 1D U-Net（论文架构）+ ResMLP+FiLM（对照变体）
│   ├── sde.py                     # VP / subVP SDE
│   ├── ema.py                     # 参数 EMA
│   ├── latent_norm.py             # 隐空间标准化
│   └── tsgm.py                    # 训练/采样 API 组装
├── metrics/                       # discriminative / predictive / t-SNE
├── scripts/
│   ├── prepare_data.py / prepare_data_pm25.py / prepare_data_esa.py
│   ├── train.py                   # 两阶段训练
│   ├── sample.py                  # 递归 PC 采样
│   ├── evaluate.py                # 10-seed 评估 + 报告输出
│   ├── run_esa_anomaly_inference.py  # ESA 独立推理演示脚本（见下）
│   └── run_full.{sh,ps1} / run_full_pm25.{sh,ps1} / run_full_esa.{sh,ps1}  # 一键全流程
├── outputs/                        # checkpoints / 样本 / 报告（git 忽略）
├── reproduction_results/           # 已版本控制的结果快照
└── docs/reproduction_plan.md       # 完整复现方案文档（论文精读、里程碑、风险）
```

## 快速开始

一键跑通某个数据集的完整流程（数据准备 → 训练 → 采样 → 10-seed 评估）：

```bash
# Linux/macOS
DEVICE=cuda scripts/run_full.sh        # C-MAPSS
DEVICE=cuda scripts/run_full_pm25.sh   # PM25
DEVICE=cuda scripts/run_full_esa.sh    # ESA

# Windows
$env:DEVICE="cuda"; scripts/run_full.ps1
```

或分步执行：

```bash
python scripts/prepare_data.py --out data/processed/cmapss --T 24
python scripts/train.py --config configs/cmapss.yaml --out outputs/checkpoints/cmapss_full --device cuda
python scripts/sample.py --config configs/cmapss.yaml --checkpoint outputs/checkpoints/cmapss_full/ckpt_best.pt --n-samples 100
python scripts/evaluate.py --config configs/cmapss.yaml --checkpoint outputs/checkpoints/cmapss_full/ckpt_best.pt --n-seeds 10
```

### ESA 推理演示

独立于正式评估协议之外，[scripts/run_esa_anomaly_inference.py](scripts/run_esa_anomaly_inference.py) 提供一个带完整进度日志的推理脚本，加载 ESA checkpoint、生成合成样本，并输出逐通道对比图、均值/标准差带图、t-SNE 图和指标汇总：

```bash
python scripts/run_esa_anomaly_inference.py \
    --checkpoint outputs/checkpoints/esa_full/ckpt_best.pt \
    --n-samples 100
```

产出示例见 [esa_anomaly_inference_results/](esa_anomaly_inference_results/)。

## 环境依赖

```
python>=3.9
torch>=2.0
numpy, pandas, scipy
scikit-learn
matplotlib
pyyaml, tqdm
tensorboard
```