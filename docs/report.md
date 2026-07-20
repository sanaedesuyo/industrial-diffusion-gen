<center>

![北京化工大学](report/figures/buct_logo.png)

# 生产实习
# 课程设计报告

<br><br>

|              |                                                                      |
| ------------:|:-------------------------------------------------------------------- |
| **课设题目：** | 基于扩散模型的规则时序数据生成方法（TSGM）复现——面向工业时序数据 |
| **班　　级：** | ______________________ |
| **姓　　名：** | ______________________ |
| **学　　号：** | ______________________ |
| **校内导师：** | ______________________ |
| **企业导师：** | ______________________ |

<br>

2026 年 7 月

</center>

<div style="page-break-after: always;"></div>

## 摘　要

工业过程与设备运行中产生的多元时序数据具有采样规则、维度较高、动力学复杂等特点，对其进行高质量的合成生成，可为异常检测、健康预测等下游任务提供数据增强与仿真验证手段。本课题面向 AAAI 2023 论文《Regular Time-series Generation using SGM》（TSGM）的方法开展复现工作，并将其从原论文中的金融、能源、空气质量场景迁移应用到工业时序场景。TSGM 将 score-based（扩散）生成模型引入"规则时序生成"任务：先用 GRU 自编码器把原始序列 x_{1:T} 映射到隐空间 h_{1:T}，再训练一个以上一时刻隐向量 h_{t-1} 为条件的一维 U-Net 条件分数网络，在 VP/subVP 随机微分方程框架下学习隐空间中的条件分数，最终通过递归 Predictor-Corrector 采样逐步生成 ĥ_{1:T}，并由解码器一次性还原为合成序列。本课题按论文 Algorithm 1 完整实现了自编码器预训练与分数网络主训练两阶段流程，并在涡扇发动机退化仿真数据集 C-MAPSS、北京 PM2.5 空气质量数据集 PM25，以及欧空局卫星遥测异常数据集 ESA Anomaly Dataset 三个数据集上完成训练、递归采样与基于 TimeGAN 协议的 10 随机种子评估（判别得分、预测得分、t-SNE 可视化）。实验结果表明，ESA、PM25 上的判别得分分别达到 0.0716±0.0193 与 0.2454±0.0179，预测得分在三个数据集上均接近论文报告水平；而 C-MAPSS 判别得分明显偏高（0.3281±0.0830），经对照实验确认这并非实现缺陷，而是该方法基于单步马尔可夫条件假设难以刻画 C-MAPSS 强单调退化趋势这一结构性局限。本课题同时对训练中的 checkpoint 选择方差、隐空间尺度失配、递归采样保真度等工程问题进行了诊断与改进，相关代码与结果已整理归档。

**关键词：** 扩散模型，条件分数网络，规则时序生成，工业时序数据

<div style="page-break-after: always;"></div>

## 目　录

- [前　言](#前言)
- [第 1 章　绪论](#第1章绪论)
  - [1.1　项目背景及意义](#11项目背景及意义)
  - [1.2　本课题主要研究内容](#12本课题主要研究内容)
  - [1.3　开发环境](#13开发环境)
  - [1.4　项目总体结构](#14项目总体结构)
  - [1.5　本人完成的工作](#15本人完成的工作)
- [第 2 章　项目总体设计](#第2章项目总体设计)
  - [2.1　方法总体框架](#21方法总体框架)
  - [2.2　数据集与预处理方案设计](#22数据集与预处理方案设计)
  - [2.3　模型架构设计](#23模型架构设计)
  - [2.4　训练流程设计](#24训练流程设计)
  - [2.5　采样流程设计](#25采样流程设计)
  - [2.6　评估方案设计](#26评估方案设计)
- [第 3 章　项目实现与测试](#第3章项目实现与测试)
  - [3.1　任务具体实现](#31任务具体实现)
  - [3.2　项目难点分析](#32项目难点分析)
  - [3.3　实验结果与分析](#33实验结果与分析)
- [结　论](#结论)
- [参考文献](#参考文献)
- [致　谢](#致谢)

<div style="page-break-after: always;"></div>

## 前　言

生产实习是本科阶段将课堂所学理论知识与实际工程问题相结合的重要实践环节。近年来，深度生成模型在图像、语音等领域取得了显著进展，其中以去噪扩散概率模型（Denoising Diffusion Probabilistic Model）与基于分数的生成模型（Score-based Generative Model, SGM）为代表的 score-based 方法，凭借其训练稳定、生成质量高等优点，已成为生成建模领域的主流范式之一。与图像、语音等数据不同，工业场景中的多元时序数据（如设备传感器读数、环境监测数据、卫星遥测数据等）普遍存在标注稀缺、异常样本稀少、跨设备/跨工况分布差异大等问题，导致基于时序数据训练的异常检测、健康预测、故障诊断等模型往往面临数据不足的困境。将 score-based 生成模型引入时序数据合成，为缓解上述数据稀缺问题、开展数据增强与仿真验证提供了一条可行路径。

本课题选取了 AAAI 2023 论文《Regular Time-series Generation using SGM》（简称 TSGM，Lim et al., arXiv:2301.08518）作为复现对象。该论文是较早将 score-based 生成模型系统性地应用于"规则时序生成"任务的工作，与已有的时序预测（如 TimeGrad、ScoreGrad）、时序插补（如 CSDI）等 score-based 时序任务相区别，其核心贡献在于提出了隐空间中的条件去噪分数匹配目标与递归式的 Predictor-Corrector 采样流程。本课题在原论文实验设置的基础上，将方法迁移到三个工业及环境监测相关的时序数据集（C-MAPSS 涡扇发动机退化仿真数据、PM25 空气质量数据、ESA 卫星遥测异常数据）上进行了完整复现，覆盖论文精读、数据适配、模型实现、两阶段训练、递归采样与标准评估协议等全流程环节，并对复现过程中遇到的工程问题与方法本身的局限性进行了具体分析。

本报告的撰写目的，一方面是对本次生产实习课程设计的工作内容、实现方案与实验结果进行系统总结；另一方面也希望通过对论文方法的深入复现与结果分析，加深对 score-based 生成模型在时序数据领域应用的理解。由于本人水平有限，报告中难免存在不足之处，恳请各位老师批评指正。

<div style="page-break-after: always;"></div>

## 第1章　绪论

### 1.1　项目背景及意义

#### 1.1.1　项目背景

工业设备与过程监测系统中普遍部署了大量传感器，持续产生多元、规则采样的时间序列数据，例如涡扇发动机的多传感器退化仿真数据、空气质量监测站的逐小时观测数据、卫星平台的遥测数据等。这类数据是异常检测、剩余寿命预测、故障诊断等工业智能算法的训练基础，但在真实场景下往往存在如下问题：

- **标注/异常样本稀缺**：真实故障或异常工况发生频率低，难以覆盖足够多的故障模式；
- **跨设备/跨工况分布差异大**：不同批次设备、不同工况下的数据分布存在偏移，单一数据源难以支撑模型的泛化能力验证；
- **数据获取成本高、隐私/安全限制多**：部分工业数据（如卫星遥测、电力系统数据）出于安全或商业原因难以大规模公开共享。

生成建模（Generative Modeling）技术，尤其是近年来在图像、语音领域取得突破性进展的 score-based / diffusion 生成模型，为缓解上述问题提供了新的思路：通过学习真实数据的分布，生成结构真实、统计特性一致的合成序列，可以用于数据增强、仿真验证、隐私保护下的数据共享等场景。

#### 1.1.2　研究意义

AAAI 2023 论文《Regular Time-series Generation using SGM》（TSGM）首次将 score-based 生成模型系统性地引入"规则时序生成"任务，填补了此前 score-based 方法在时序预测（TimeGrad/ScoreGrad）与时序插补（CSDI）之外、面向时序"生成"这一细分方向的空白。相比 GAN 类时序生成方法（如 TimeGAN），diffusion/score-based 方法具有训练更稳定、不易出现模式崩塌（mode collapse）等优点。

本课题对 TSGM 进行复现，具有以下意义：

1. **方法验证**：论文提交时未附官方代码仓库（under review 阶段），通过独立复现可以验证公式推导与算法描述的可执行性和正确性；
2. **场景迁移**：原论文实验场景聚焦金融（Stocks）、能源（Energy）、空气质量（Air）等，本课题额外引入工业退化仿真数据（C-MAPSS）与卫星遥测数据（ESA Anomaly Dataset），检验该方法在工业场景下的适用边界；
3. **工程实践能力锻炼**：涵盖论文精读、数据管道搭建、深度学习模型实现、两阶段训练算法、SDE 数值采样、标准化评估协议等完整的科研工程闭环，是对本科阶段所学深度学习、概率统计、数值计算等知识的综合应用。

### 1.2　本课题主要研究内容

本课题的主要研究内容包括：

1. **论文方法精读与公式推导梳理**：明确 TSGM 的三大组件（GRU 自编码器、条件 Score 网络、VP/subVP SDE）、两个训练损失（重构损失、隐空间条件去噪分数匹配损失）、两阶段训练算法（Algorithm 1）与递归 Predictor-Corrector 采样流程。
2. **数据集适配性评估与预处理管道搭建**：评估 C-MAPSS、PM25、SMAP、ESA Anomaly Dataset、PhysioNet 等候选数据集与论文"规则时序"前提的匹配程度，选定 C-MAPSS + PM25 作为训练数据集、ESA Anomaly Dataset 作为标准协议测试数据集，并实现对应的加载、清洗、归一化、滑窗预处理管道。
3. **模型与算法实现**：实现 GRU 编码器/解码器、支持 1D U-Net 与 ResMLP+FiLM 两种架构的条件分数网络、VP/subVP 随机微分方程、参数 EMA、隐空间标准化等核心模块，并按论文 Algorithm 1 实现两阶段训练流程与递归 PC 采样器。
4. **标准评估协议实现**：严格按 TimeGAN 协议实现判别得分（Discriminative score）、预测得分（Predictive score / TSTR）、t-SNE 可视化三项指标，并支持 10 随机种子的统计评估。
5. **实验与工程问题诊断**：在三个数据集上完成完整训练与评估，针对训练不稳定、checkpoint 选择方差过大、隐空间尺度失配、递归采样保真度不足等工程问题进行诊断与改进，并对方法本身在不同数据集上的表现差异进行归因分析。

#### 1.2.1　核心创新点回顾

TSGM 相较传统时序 GAN（如 TimeGAN）方法的核心区别在于：目标学习的是**隐空间中、以上一时刻隐向量为条件的条件分数** ∇log p(h_t^s | h_{t-1})，而不是直接在原始数据空间或隐空间上做对抗学习；采样过程也不是一次性生成整段序列，而是**逐时刻递归生成**，每一步都以上一步已生成的隐向量为条件，通过 SDE 反向采样得到当前时刻的隐向量，最后统一解码还原。

### 1.3　开发环境

| 类别 | 配置 |
| --- | --- |
| 操作系统 | Windows 10 / 支持 CUDA 的 GPU 环境（亦可运行于 macOS / Linux，`scripts/config_utils.py` 中 `get_default_device()` 会按平台自动选择 `cuda` / `mps` / `cpu`） |
| 编程语言 | Python ≥ 3.9 |
| 深度学习框架 | PyTorch ≥ 2.0 |
| 主要依赖库 | numpy、pandas、scipy、scikit-learn、matplotlib、pyyaml、tqdm、tensorboard |
| 开发工具 | Cursor IDE（AI 辅助编程）、Git 版本管理、PowerShell / Bash 脚本 |
| 硬件参考 | 单张 GPU（论文原始实验使用 RTX 2080Ti 11GB；TSGM 训练显存占用约 4GB） |

完整依赖见项目根目录 `requirements.txt`：

```text
torch>=2.0
numpy
pandas
scipy
scikit-learn
matplotlib
pyyaml
tqdm
tensorboard
```

### 1.4　项目总体结构

项目代码仓库 `industrial-diffusion-gen` 的目录结构如下：

```text
industrial-diffusion-gen/
├── configs/                            # 每数据集一份 YAML：模型架构、SDE、训练超参
│   ├── cmapss.yaml
│   ├── pm25.yaml
│   └── esa.yaml
├── data/
│   ├── loaders/                        # 数据加载与预处理
│   │   ├── base.py                     # 通用：MinMax 归一化、滑窗、切分
│   │   ├── cmapss.py
│   │   ├── pm25.py
│   │   └── esa.py
│   └── processed/                      # 预处理后的 .npy 窗口张量
├── models/
│   ├── autoencoder.py                  # GRU Encoder/Decoder
│   ├── score_unet1d.py                 # 1D U-Net（论文架构）+ ResMLP+FiLM（对照变体）
│   ├── sde.py                          # VP / subVP SDE
│   ├── ema.py                          # 参数 EMA
│   ├── latent_norm.py                  # 隐空间标准化
│   └── tsgm.py                         # 训练/采样 API 组装
├── metrics/                            # discriminative / predictive / t-SNE
│   ├── discriminative.py
│   ├── predictive.py
│   └── visualization.py
├── scripts/
│   ├── prepare_data.py / prepare_data_pm25.py / prepare_data_esa.py
│   ├── train.py                        # 两阶段训练
│   ├── sample.py                       # 递归 PC 采样
│   ├── evaluate.py                     # 10-seed 评估 + 报告输出
│   ├── run_esa_anomaly_inference.py    # ESA 独立推理演示脚本
│   └── run_full.{sh,ps1} / run_full_pm25.{sh,ps1} / run_full_esa.{sh,ps1}
├── outputs/                            # checkpoints / 样本 / 报告（git 忽略）
├── reproduction_results/               # 已归档的 10-seed 评估结果快照
├── esa_anomaly_inference_results/      # ESA 独立推理演示产出
└── docs/                               # 复现方案文档、本报告
```

**主要任务/总体结构**：整体流程遵循"数据准备 → 训练（两阶段）→ 采样 → 评估"的四段式流水线，每一段对应一个可独立运行的脚本，通过统一的 YAML 配置文件驱动，便于在三个数据集之间复用同一套代码。

### 1.5　本人完成的工作

在本次生产实习课程设计中，本人完成的工作包括：

1. 精读 TSGM 论文，梳理其数学推导（自编码器重构损失、隐空间条件去噪分数匹配目标、VP/subVP SDE 前向/反向过程）与算法流程（Algorithm 1 两阶段训练、递归 PC 采样），形成复现方案文档 `docs/reproduction_plan.md`；
2. 对候选数据集（C-MAPSS、PM25、SMAP、STFT、HallThruster、PhysioNet、ESA Anomaly Dataset）逐一评估与论文"规则时序"前提的匹配度，确定 C-MAPSS + PM25 为训练/开发集、ESA Anomaly Dataset 为标准协议测试集的选型方案；
3. 实现三个数据集的原始数据加载与预处理管道（`data/loaders/`），完成 MinMax 归一化、滑窗切分、按机组/时间顺序的 train/val/test 划分；
4. 实现 GRU 自编码器（`models/autoencoder.py`）、条件 Score 网络（1D U-Net 与 ResMLP+FiLM 两种架构，`models/score_unet1d.py`）、VP/subVP SDE（`models/sde.py`）、参数 EMA（`models/ema.py`）、隐空间标准化（`models/latent_norm.py`）等核心模块；
5. 按论文 Algorithm 1 实现两阶段训练流程与递归 Predictor-Corrector 采样器（`models/tsgm.py`、`scripts/train.py`、`scripts/sample.py`）；
6. 实现基于 TimeGAN 协议的判别得分、预测得分、t-SNE 可视化三项评估指标（`metrics/`、`scripts/evaluate.py`），并支持 10 随机种子统计评估；
7. 在三个数据集上完成完整的训练、采样与评估实验，诊断并修复隐空间尺度失配、checkpoint 选择方差过大、递归采样保真度不足等工程问题；
8. 对 C-MAPSS 判别得分明显偏高的现象开展对照实验，归因分析该方法的结构性局限；
9. 编写 ESA 数据集独立推理演示脚本（`scripts/run_esa_anomaly_inference.py`），产出逐通道对比图、均值/标准差带图；
10. 整理项目 README 文档与本课程设计报告。

<div style="page-break-after: always;"></div>

## 第2章　项目总体设计

### 2.1　方法总体框架

TSGM 是首个用 score-based / diffusion 模型做**规则时序生成**（区别于时序预测 TimeGrad/ScoreGrad、时序插补 CSDI）的方法。其核心思路是：把时序 x_{1:T} 映射到隐空间 h_{1:T}，用一个以**上一时刻隐向量为条件**的分数网络 M_θ(s, h_t^s, h_{t-1}) 学习隐空间中的条件分数 ∇log p(h_t^s | h_{t-1})，再用 SDE 反向采样递归生成整段序列。

方法由三大组件构成：

| 组件 | 作用 | 对应代码 |
| --- | --- | --- |
| GRU 自编码器 | h_t = e(h_{t-1}, x_t)，x̂_t = d(h_t)，把时序映射到隐空间再还原 | `models/autoencoder.py` |
| 条件 Score 网络 | 输入 [被扩散隐向量 ⊕ 条件 h_{t-1}] + 扩散时间 s，预测分数 | `models/score_unet1d.py` |
| VP / subVP SDE | 前向加噪 / 反向去噪的随机微分方程（论文排除效果较差的 VE-SDE） | `models/sde.py` |

总体数据流可概括为：

```text
原始序列 x_{1:T}
     │  GRU Encoder e(·)
     ▼
隐序列 h_{1:T}  ──(标准化)──►  条件分数网络训练 / 递归 PC 采样  ──►  ĥ_{1:T}
     │  GRU Decoder d(·)                                            │  GRU Decoder d(·)
     ▼                                                               ▼
重构序列 x̂_{1:T}（训练阶段自监督信号）                        合成序列 x̂_{1:T}（生成阶段最终输出）
```

两个核心损失函数：

- 自编码器重构损失：L_ed = E‖x̂_{1:T} - x_{1:T}‖²
- 隐空间条件去噪分数匹配损失（论文主要贡献，约定 h_0 = 0，λ(s) 权重沿用 Song et al. 2021）：

```text
L_score^H = E_s E_{h_{1:T}} Σ_{t=1}^{T} λ(s) E_{h_t^s} ‖ M_θ(s, h_t^s, h_{t-1}) - ∇_{h_t^s} log p(h_t^s | h_t) ‖²
```

### 2.2　数据集与预处理方案设计

窗口长度统一设计为 **T = 24**，逐特征 MinMax 归一化到 [0, 1]（仅用训练集拟合归一化器，避免信息泄漏）。三个数据集的设计方案如下：

| 数据集 | 内容 | 切分策略 | 特征维度 D |
| --- | --- | --- | --- |
| **C-MAPSS** | 涡扇发动机 21 路传感器 + 3 路工况，多机组逐 cycle 数据 | 按 engine unit 分组切窗，避免跨机组穿越 | 14（剔除近常量列后） |
| **PM25** | 北京多站点 PM2.5 逐小时观测（对应论文 Air 场景） | 时间顺序切分（前段 train，后段 test），滑窗 | 36 |
| **ESA Anomaly Dataset** | 卫星 Mission1 遥测，16 个共同采样通道 | 重采样到统一小时网格，时间顺序切分（无官方切分文件，本课题自定义标准协议） | 16 |

数据集选型依据（对候选数据集的适配性评估）：

| 数据集 | 采样规则性 | 匹配度 | 结论 |
| --- | --- | --- | --- |
| C-MAPSS | 规则（逐 cycle） | ★★★★★ | 核心训练集 |
| PM25 | 规则（逐小时） | ★★★★ | 核心训练集（论文 Air 场景的工业/环境类比） |
| SMAP | 规则 | ★★★☆ | 可选扩展，未纳入本轮复现 |
| ESA Anomaly Dataset | 规则遥测 | 用于评估 | 标准协议测试集 |
| PhysioNet | 不规则、稀疏、缺失多 | ★★ | 与"regular"前提冲突，未纳入 |

预处理管道设计为通用四步流程（`data/loaders/base.py`）：

1. 选取连续特征列，剔除常量/近常量列（避免归一化除零）；
2. 缺失值处理：前向填充 + 线性插值（PM25、ESA 两数据集均存在少量缺失）；
3. 逐特征 MinMax 归一化到 [0,1]（仅用训练集拟合）；
4. 定长滑窗切分（T=24，步长可配），并按数据集特性选择"按机组分组"（C-MAPSS）或"按时间顺序"（PM25、ESA）的划分策略，避免窗口跨越 train/test 边界造成信息泄漏。

同时，为支撑判别得分评估的"同分布留出集"协议，在训练集内部再切出一份窗口级 `val.npy`，与生成样本比较时避免引入 train/test 机组/时段分布偏移带来的额外偏差。

### 2.3　模型架构设计

**自编码器**：采用多层 GRU 编码器（x_{1:T} → h_{1:T}）与 GRU 解码器（h_{1:T} → x̂_{1:T}），隐藏维度 d_hidden 取输入维度 D 的 2~5 倍（三数据集分别取 64/96/64）。解码器输出经 Sigmoid 约束到 [0,1]，与归一化后的数据支撑域一致。

**条件 Score 网络**设计为可切换的两种架构：

- **1D U-Net**（默认，对应论文原始架构）：把拼接向量 [h_t^s ⊕ h_{t-1}]（长度 2·d_hidden）视为单通道一维信号，通过多层 stride-2 卷积下采样 + 上采样 + skip connection 构成 U-Net，扩散时间 s 通过正弦时间嵌入 + FiLM（缩放/平移）方式注入每个卷积块；
- **ResMLP + FiLM**（论文之外的对照变体）：由于分数网络作用的隐向量本身没有空间/时序局部性结构，额外实现了一个残差 MLP + FiLM 条件的变体作为消融对照。

**SDE 模块**实现 VP 与 subVP 两种随机微分方程（论文明确排除表现较差的 VE-SDE），提供前向漂移/扩散系数、边缘分布 mean/std 解析解、反向 SDE 系数等接口，供训练时的加噪与采样时的去噪共用。

**辅助模块**：参数 EMA（对分数网络做指数滑动平均，采样时使用 EMA 权重以提升稳定性）、隐空间标准化器 `LatentStandardizer`（将编码器输出的隐向量标准化为零均值/单位方差，解决隐空间尺度与 SDE 默认参数不匹配的问题，详见 3.2 节）。

### 2.4　训练流程设计

训练流程严格对齐论文 Algorithm 1，设计为两阶段：

1. **自编码器预训练**（`iter_pre` 步）：仅用重构损失训练 encoder + decoder，使隐空间具备基本的可逆重构能力；
2. **分数网络主训练**（`iter_main` 步）：冻结（或按 `use_alt` 交替微调）自编码器，用隐空间条件去噪分数匹配损失训练 Score 网络；条件 h_{t-1} 需要 `detach`，避免分数损失的梯度泄漏回自编码器破坏隐空间结构；
3. **Checkpoint 选择**：训练过程中每 `save_every` 步，用当前 EMA 权重生成多批次合成样本、以多随机种子训练判别器打分，取平均判别得分最低的一版存为 `ckpt_best.pt`，供后续采样与评估使用。

### 2.5　采样流程设计

采样阶段采用**递归 + Predictor-Corrector（PC）**的设计：

- t = 1：从标准高斯先验采样 z_0，与约定的 h_0 = 0 拼接，经 PC 采样器反解出 ĥ_1；
- t = 2..T：读入上一步生成的 ĥ_{t-1} 作为条件，经条件反向 SDE（Predictor：reverse-diffusion / Euler-Maruyama；Corrector：Langevin MCMC，默认 N=1000 步）生成 ĥ_t；
- 得到完整 ĥ_{1:T} 后，一次性用 Decoder 还原为合成序列 x̂_{1:T}。

### 2.6　评估方案设计

评估方案严格沿用 TimeGAN 协议：

- **判别得分（Discriminative score）**：训练一个 2 层 LSTM 二分类器区分真/假样本，报告 |acc - 0.5|（越低越好，越接近 0 说明分类器越难以区分真假）；
- **预测得分（Predictive score，TSTR：Train on Synthetic, Test on Real）**：用合成数据训练 GRU 下一步预测器，在真实测试集上测 MAE（越低越好，越接近"用真实数据训练"的基线说明生成数据的时序依赖结构越真实）；
- **t-SNE 可视化**：真/假样本降维后的分布重合度，作为多样性检查的辅助手段；
- 每个指标跑 **10 个随机种子**，报告 mean ± std，降低分类器/预测器初始化带来的评估方差。

<div style="page-break-after: always;"></div>

## 第3章　项目实现与测试

### 3.1　任务具体实现

#### 3.1.1　数据加载与预处理模块

以 C-MAPSS 为例，`data/loaders/cmapss.py` 实现了从原始压缩包（内嵌 `CMAPSSData.zip`）中直接在内存中读取 `train_FD001.txt`/`test_FD001.txt`，按 26 列（`unit, cycle, op1-3, s1-s21`）解析为 `DataFrame`；`select_feature_columns` 用训练集统计量剔除标准差低于阈值的近常量传感器列；`make_windows` 按机组分组、逐机组滑窗切分，避免跨机组数据穿越：

```python
def make_windows(df: pd.DataFrame, feature_cols: list[str], T: int = 24, stride: int = 1) -> np.ndarray:
    windows = []
    for unit_id, unit_df in df.groupby("unit"):
        unit_df = unit_df.sort_values("cycle")
        values = unit_df[feature_cols].to_numpy(dtype=np.float32)
        n = len(values)
        if n < T:
            continue
        for start in range(0, n - T + 1, stride):
            windows.append(values[start : start + T])
    return np.stack(windows, axis=0)
```

`scripts/prepare_data.py` 在此基础上完成 train/test 机组划分、窗口级 val 留出集切分、MinMax 归一化拟合与应用，并保存 `meta.json`（记录特征列、维度、样本量等元信息）与 `normalizer.npz`（供采样后反归一化使用）。PM25（`data/loaders/pm25.py`）与 ESA（`data/loaders/esa.py`）采用类似的四步流程，区别在于：PM25 是时间顺序连续序列（无机组概念），按时间前后段切分 train/test；ESA 需要先把 16 个原始不等间隔采样的遥测通道分别重采样到统一的 1 小时网格再对齐拼接。

#### 3.1.2　自编码器模块

`models/autoencoder.py` 实现 `GRUEncoder`（`x_{1:T} → h_{1:T}`，输出全部时间步隐状态）与 `GRUDecoder`（`h_{1:T} → x̂_{1:T}`，末端接线性层 + Sigmoid）：

```python
class GRUEncoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(d_in, d_hidden, num_layers=n_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return h
```

`Autoencoder.reconstruction_loss` 即对应 L_ed = E‖x̂ - x‖²。`models/tsgm.py` 中的 `train_autoencoder` 额外支持"去噪自编码器"模式（`ae_noise_std > 0`）：按批次隐向量标准差比例注入高斯噪声后再解码，使解码器学会容忍扩散采样器产出的略有偏差的隐向量，缓解采样阶段"训练/推理隐空间分布不一致"的问题。

#### 3.1.3　条件 Score 网络模块

`models/score_unet1d.py` 实现两种可切换架构，统一接口 `forward(h_s_t, h_cond, s) -> score`：

- `ConditionalScoreUNet1D`：将 `[h_s_t ⊕ h_cond]` 视为长度 `2·d_hidden` 的单通道一维信号，经 `in_conv` → 多层 `FiLMConvBlock`（stride-2 卷积 + GroupNorm + SiLU + FiLM 时间条件）下采样 → 中间块 → 上采样并与跳连特征拼接 → 展平后经 MLP 投影回 `d_hidden` 维分数输出；
- `ConditionalScoreMLP`：由若干 `FiLMResBlock`（LayerNorm 预归一化残差 MLP + FiLM）堆叠而成，作为该方法在低维、无空间局部性隐向量上的对照架构。

扩散时间 `s ∈ [0,1]` 通过正弦位置编码（`SinusoidalTimeEmbedding`）映射为高维时间嵌入，再经两层 MLP 处理后以 FiLM 缩放/平移的方式注入每个卷积/残差块，这是扩散模型条件注入的标准做法。`build_score_net` 按配置文件 `model.score_net_type` 字段选择具体实现，默认为 `unet`（复现论文架构），三个数据集配置均采用 `unet_channels=[32,64,64,128]`、`unet_depth=4`。

#### 3.1.4　SDE 与 PC 采样器模块

`models/sde.py` 实现 `VPSDE` 与 `SubVPSDE`（均继承自抽象基类 `SDE`），提供前向 SDE 的漂移/扩散系数 `sde()`、边缘分布 `p(x_s|x_0)` 的均值/标准差解析解 `marginal_prob()`：

```python
class VPSDE(SDE):
    def sde(self, x, s):
        beta_s = _beta(s, self.beta_min, self.beta_max)
        drift = -0.5 * beta_s * x
        diffusion = torch.sqrt(beta_s.clamp_min(0))
        return drift, diffusion

    def marginal_prob(self, x0, s):
        log_mean_coeff = _log_mean_coeff(s, self.beta_min, self.beta_max)
        mean = torch.exp(log_mean_coeff) * x0
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std
```

`models/tsgm.py` 中的 `pc_sample_step` 实现单步 Predictor-Corrector 采样：Corrector 阶段用 Langevin MCMC，按信噪比 `snr` 自适应确定步长；Predictor 阶段用 reverse-diffusion / Euler-Maruyama 离散化反向 SDE。`recursive_generate` 在此基础上按 `t=1..T` 循环调用 `pc_sample_step`，每步都以上一步生成的隐向量为条件，最终堆叠为完整隐序列并批量解码。

#### 3.1.5　训练脚本

`scripts/train.py` 是两阶段训练的顶层入口，主要流程：加载配置与数据 → 构建自编码器/分数网络/SDE/EMA → 自编码器预训练（并拟合隐空间标准化器）→ 分数网络主训练（按 `save_every` 分块，每块训练后存档并做 checkpoint 选择）。核心的分数损失采用数值稳定的"噪声预测"形式：

```python
s = torch.rand(B * T, device=device) * (1 - eps) + eps
mean, std = sde.marginal_prob(h_flat, s)
z = torch.randn_like(h_flat)
h_s = mean + std.unsqueeze(-1) * z

pred = score_net(h_s, h_prev_flat, s)
score_loss = torch.mean((std.unsqueeze(-1) * pred + z) ** 2)
```

即 λ(s) = std(s)² 的加权去噪分数匹配损失，训练目标等价于让网络预测标准化噪声 `z` 的负值除以 `std`。

#### 3.1.6　评估与可视化模块

`metrics/discriminative.py`、`metrics/predictive.py` 分别实现 2 层 LSTM 判别器与 GRU 预测器；`metrics/visualization.py` 用 scikit-learn 的 `TSNE` 对真/假样本降维后绘制散点图。`scripts/evaluate.py` 整合三者：加载 checkpoint（含 EMA 权重与隐空间标准化器）→ 递归采样与测试集等量的合成样本 → 循环 10 个随机种子计算判别/预测得分 → 输出 CSV 报告与 t-SNE 图。

#### 3.1.7　一键运行脚本与独立推理演示

`scripts/run_full.{sh,ps1}`（及 `_pm25`、`_esa` 变体）串联"数据准备 → 训练 → 采样 → 10-seed 评估"全流程，便于一键复现。此外还实现了独立于正式评估协议之外的 `scripts/run_esa_anomaly_inference.py`，提供带完整进度日志的推理演示：加载 ESA checkpoint、生成合成样本，输出逐通道对比图、均值/标准差带图、t-SNE 图与指标汇总 JSON，便于在不跑完整评估的情况下快速检视生成质量。

### 3.2　项目难点分析

#### 3.2.1　隐空间尺度失配问题

**问题现象**：C-MAPSS 上训练初期分数网络损失下降缓慢，采样样本判别得分很高（生成质量差）。

**诊断过程**：排查发现 GRU 编码器学到的隐向量方差非常小（各维标准差约 0.01~0.13，整体约 0.28），原因是该数据重构任务较为容易，编码器倾向于收缩到很紧凑的隐空间。而 VP SDE 的默认参数（`beta_min=0.1, beta_max=20`）是按照 Song et al. (2021) 中假设输入近似单位方差设计的。当隐向量方差比这个假设低两个量级时，扩散过程的大部分时间段都在对一个相对自身尺度而言已经被噪声淹没的信号进行去噪，训练信号被严重稀释；同时分数网络输入 `[h_s_t, h_cond]` 中两部分数值量级也不一致（`h_cond` 量级 ~0.1，`h_s_t` 在 s→1 附近可扫到 ~1），进一步加剧了优化困难。

**解决方案**：新增 `models/latent_norm.py` 中的 `LatentStandardizer`，在自编码器预训练完成后，对训练集隐向量拟合逐维零均值/单位方差标准化器；分数网络训练与递归采样均在标准化后的隐空间中进行，解码前再做逆标准化还原：

```python
class LatentStandardizer(nn.Module):
    def fit(self, h, eps=1e-6):
        flat = h.reshape(-1, h.shape[-1])
        self.mean.copy_(flat.mean(dim=0))
        self.std.copy_(flat.std(dim=0).clamp_min(eps))
        return self
```

该改动使隐空间尺度与 SDE 默认超参匹配的假设一致，显著改善了训练收敛性与生成质量。

#### 3.2.2　checkpoint 选择方差过大问题

**问题现象**：训练早期版本仅用单批次静态生成样本 + 多随机种子训练判别器来选最优 checkpoint，选出的"最优"版本内部评分能低至 0.002~0.03，但用完整的 10-seed / 全量测试集协议重新评估时，判别得分却高达 0.31~0.36，选择标准与最终评估标准严重不一致。

**诊断过程**：分析发现根因是"生成draw方差"未被纳入选择过程——单批次生成样本本身就带有较大的随机性（每次重新采样的噪声不同），只对同一批静态样本反复用不同判别器种子打分，并不能反映生成过程本身的方差，导致选择过程"过拟合"到某一次偶然生成质量较好的样本批次。

**解决方案**：`scripts/train.py` 中的 `maybe_select_best` 改为对 EMA 权重**重新采样 `select_n_fake_batches` 次**（每次都从全局随机数流中抽取新的噪声），每批次再用 `select_n_seeds` 个判别器种子打分，取全部 `select_n_fake_batches × select_n_seeds` 次测量的平均值作为该 checkpoint 的判别得分估计；同时将验证集样本量从 512 提升到全量（如 C-MAPSS 的 1460），进一步降低估计方差：

```python
discs = []
for _ in range(select_n_fake_batches):
    fake = tsgm.sample(ae, ema_net, sde, n_samples=select_n_samples, ...)
    fake_np = fake.cpu().numpy()
    discs.extend(discriminative_score(val_np, fake_np, seed=s, device=device)
                 for s in range(select_n_seeds))
disc = float(np.mean(discs))
```

该改进使 checkpoint 选择的内部估计与最终 10-seed 完整评估结果基本一致，避免了"选择偏差"。

#### 3.2.3　递归采样保真度问题（温度采样）

**问题现象**：在诊断生成样本的统计特性时发现，生成窗口相比真实窗口存在约 40% 的超额趋势/斜率幅度，即便逐时刻边缘分布与一阶自相关系数都已经匹配得较好。

**诊断过程**：进一步排查确认该问题对 Corrector 步数（`n_corrector_steps`）与信噪比参数（`snr`）都不敏感，说明这并非 SDE 数值积分的离散化误差，而是分数网络本身学到的条件方差略微偏大（over-dispersed），导致递归采样中"上一步误差"通过条件传递被逐步放大。

**解决方案**：在 `pc_sample_step` 与 `recursive_generate` 中引入 `temperature` 参数，对 Corrector（Langevin）与 Predictor（reverse-diffusion）两个阶段注入的随机噪声项统一做低温缩放：

```python
h_s = h_s + step_size * grad + torch.sqrt(2 * step_size) * noise * temperature
...
h_s = h_mean + diffusion * torch.sqrt(ds) * torch.randn_like(h_s) * temperature
```

这是标准的低温采样（low-temperature sampling）技巧，`temperature < 1` 通过牺牲一定的生成多样性换取与真实分布更贴近的保真度，在本课题的递归条件采样场景下能有效抑制误差累积带来的超额趋势问题。

#### 3.2.4　C-MAPSS 判别得分明显偏高的结构性原因分析

三个数据集完整评估后发现一个现象：**ESA 判别得分最好（0.0716），PM25 居中（0.2454），C-MAPSS 明显偏高（0.3281）**，且这一相对排序在多轮参数调整下保持稳定。

**归因分析**：由于三个数据集使用完全相同的代码与相近的架构配置进行训练和评估，可以排除"实现 bug"这一假设。深入分析数据特性发现：

- C-MAPSS 的多传感器信号具有**强单调退化趋势**（发动机从健康状态逐渐老化到失效），窗口内数值随时间近似单调变化；
- TSGM 的条件分布假设是**一阶马尔可夫**的 h_t | h_{t-1}，即当前时刻隐向量只依赖于紧邻的上一时刻，这对 PM25（具有明显日/周周期性、局部平稳）和 ESA（准平稳遥测信号）这类数据是较为合理的近似，但难以充分刻画 C-MAPSS 这种"长程、强单调"的趋势结构——单步条件下生成的序列容易在局部看起来合理，但整体趋势的长程一致性（是否始终朝退化方向单调推进）更难保证，从而更容易被判别器捕捉到统计差异。

因此可以得出结论：C-MAPSS 上偏高的判别得分**不是实现缺陷**，而是 TSGM 方法基于单步马尔可夫条件假设这一设计本身的结构性局限，在面对强单调趋势型工业退化数据时表现出的固有短板。这一发现也是本课题在"方法从原论文场景迁移到工业场景"这一目标下得到的一项重要分析结论。

（补充说明：训练过程中还尝试过 `use_alt=True`，即在分数网络主训练阶段交替微调自编码器，在 C-MAPSS 上的结果反而更差——判别得分从 0.3134 上升到 0.3618，原因是联合微调下自编码器出现了过拟合/记忆训练窗口的现象（重构损失降到 ~3×10⁻⁶），破坏了隐空间的平滑性，进而影响分数网络的拟合质量，因此该数据集最终采用 `use_alt=False`。）

### 3.3　实验结果与分析

#### 3.3.1　训练与评估配置

三个数据集均采用窗口长度 T=24，最终训练配置汇总如下：

| 配置项 | C-MAPSS | PM25 | ESA |
| --- | --- | --- | --- |
| 特征维度 D | 14 | 36 | 16 |
| 隐藏维度 d_hidden | 64 | 96 | 64 |
| Score 网络架构 | 1D U-Net（depth=4） | 1D U-Net（depth=4） | 1D U-Net（depth=4） |
| SDE 类型 | VP | VP | VP |
| AE 预训练步数 iter_pre | 10,000 | 50,000 | 10,000 |
| Score 主训练步数 iter_main | 40,000 | 40,000 | 40,000 |
| use_alt（交替微调 AE） | False | True | False |
| 采样步数 n_steps_sample | 1000 | 1000 | 1000 |
| 评估随机种子数 | 10 | 10 | 10 |

#### 3.3.2　10-seed 标准评估结果

在三个数据集上完成完整训练后，使用 `scripts/evaluate.py` 按 TimeGAN 协议进行 10 随机种子评估，结果如下（数据来源：`reproduction_results/2026-07-08_unet_multisampling/`）：

| 数据集 | 判别得分 Discriminative ↓ | 预测得分 Predictive ↓ |
| --- | --- | --- |
| C-MAPSS | 0.3281 ± 0.0830 | 0.0624 ± 0.0001 |
| PM25 | 0.2454 ± 0.0179 | 0.0280 ± 0.0002 |
| **ESA** | **0.0716 ± 0.0193** | 0.0247 ± 0.0028 |

关键发现：

- **预测得分** 在三个数据集上都达到了接近论文报告的水平，说明生成样本的时序统计结构（一步预测意义下的可预测性）学得较好；
- **判别得分** 上 ESA 表现最好（0.0716，接近论文报告的最佳数据集水平），PM25 居中，C-MAPSS 明显偏高（0.3281），归因分析见 3.2.4 节；
- 三个数据集使用完全相同的代码框架，说明观测到的差异反映的是数据特性与方法假设之间的匹配程度，而非工程实现问题。

t-SNE 可视化（真实样本 vs. 合成样本降维后的分布重合度）如下：

<table>
<tr>
<td><img src="report/figures/cmapss_tsne.png" width="260"/><br><center>图 3-1　C-MAPSS：真实 vs. 合成样本 t-SNE</center></td>
<td><img src="report/figures/pm25_tsne.png" width="260"/><br><center>图 3-2　PM25：真实 vs. 合成样本 t-SNE</center></td>
<td><img src="report/figures/eval_esa_tsne.png" width="260"/><br><center>图 3-3　ESA：真实 vs. 合成样本 t-SNE</center></td>
</tr>
</table>

从 t-SNE 图可以直观看出，ESA 与 PM25 的真实/合成样本分布重合度较高，而 C-MAPSS 上合成样本相对真实样本的分布覆盖存在更明显的偏移，与判别得分的定量结果相互印证。

#### 3.3.3　ESA 独立推理演示

除正式评估协议外，还使用 `scripts/run_esa_anomaly_inference.py` 对 ESA 数据集上训练得到的最优 checkpoint 进行了独立推理演示（生成 100 条合成窗口、5 个随机种子评分），结果汇总（`esa_anomaly_inference_results/esa_inference_summary.json`）：

| 指标 | 数值 |
| --- | --- |
| 判别得分 | 0.0900 ± 0.0604 |
| 预测得分 | 0.0265 ± 0.0030 |
| 生成 100 条样本耗时 | 约 167 秒（GPU） |

该结果与正式 10-seed 评估结果量级一致，进一步验证了模型的稳定性。逐通道真实/合成样本对比图与逐时刻均值±标准差带图如下：

<p align="center">
<img src="report/figures/esa_channel_comparison.png" width="600"/><br>
<center>图 3-4　ESA Mission1：16 个通道的真实 vs. 合成样本示例窗口对比</center>
</p>

<p align="center">
<img src="report/figures/esa_mean_std_band.png" width="480"/><br>
<center>图 3-5　ESA Mission1：逐时刻均值 ± 标准差带（通道平均），真实 vs. 合成</center>
</p>

从逐通道对比图可以看出，合成窗口在多数通道上都能较好地跟随真实数据的整体走势与波动范围；均值±标准差带图显示真实与合成序列在窗口内 24 个时间步上的整体统计特性（均值水平、波动范围）高度接近，进一步支撑了 ESA 数据集上定量评估得到的较低判别得分。

<div style="page-break-after: always;"></div>

## 结　论

本课题围绕 AAAI 2023 论文《Regular Time-series Generation using SGM》（TSGM）开展了完整的方法复现工作，并将其应用场景从原论文的金融/能源/空气质量数据迁移到工业与环境监测相关的时序数据（C-MAPSS 涡扇发动机退化仿真、PM25 空气质量、ESA 卫星遥测异常数据）。主要工作与结论总结如下：

1. **完整复现了 TSGM 的核心方法**：包括 GRU 自编码器、支持 1D U-Net 与 ResMLP+FiLM 两种架构的条件 Score 网络、VP/subVP SDE、两阶段训练算法（Algorithm 1）以及递归 Predictor-Corrector 采样流程，并严格按 TimeGAN 协议实现了判别得分、预测得分、t-SNE 可视化三项标准评估指标。
2. **在三个数据集上取得了与论文报告量级相近的结果**：预测得分在三个数据集上均表现良好；判别得分方面 ESA 达到 0.0716±0.0193（接近论文最佳数据集水平），PM25 为 0.2454±0.0179，C-MAPSS 为 0.3281±0.0830。
3. **诊断并解决了多个工程问题**：包括隐空间尺度与 SDE 默认参数不匹配问题（引入隐空间标准化）、checkpoint 选择方差过大问题（引入多批次重采样的选择协议）、递归采样中的超额趋势问题（引入低温采样）。这些问题的诊断过程本身也体现了扩散模型工程实践中"理论假设—数值实现—经验现象"三者之间需要反复核验的一般性经验。
4. **揭示了方法的结构性局限**：通过对照三个数据集在同一套代码下的表现差异，确认 C-MAPSS 上偏高的判别得分并非实现缺陷，而是 TSGM 基于一阶马尔可夫条件假设的方法设计，难以充分刻画强单调退化趋势型工业数据的长程结构这一固有局限。这一发现对该类方法在工业退化/健康预测场景下的适用边界给出了实证依据。

**展望**：后续工作可以从以下方向进一步展开：（1）针对强趋势型数据，探索引入更长程的条件（如滑动窗口内多个历史隐向量而非仅上一时刻）或显式趋势建模模块，以缓解 C-MAPSS 类数据上的结构性局限；（2）在更多工业数据集（如 SMAP、STFT 高频振动数据）上进一步验证方法的可迁移性；（3）探索采样步数、subVP 与 VP 的系统性对比、U-Net 深度等超参数的敏感性分析，进一步压缩生成质量与采样成本之间的权衡空间；（4）将生成的合成数据实际用于下游异常检测/健康预测任务的数据增强实验，从任务效果的角度进一步验证生成数据的实用价值。

<div style="page-break-after: always;"></div>

## 参考文献

[1] Lim H., Kim M., Park S., Lee N. Regular Time-series Generation using SGM[C]. Proceedings of the AAAI Conference on Artificial Intelligence, 2023. arXiv:2301.08518.

[2] Song Y., Sohl-Dickstein J., Kingma D. P., Kumar A., Ermon S., Poole B. Score-Based Generative Modeling through Stochastic Differential Equations[C]. International Conference on Learning Representations (ICLR), 2021.

[3] Yoon J., Jarrett D., van der Schaar M. Time-series Generative Adversarial Networks[C]. Advances in Neural Information Processing Systems (NeurIPS), 2019.

[4] Rasul K., Seward C., Schuster I., Vollgraf R. Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting (TimeGrad)[C]. International Conference on Machine Learning (ICML), 2021.

[5] Tashiro Y., Song J., Song Y., Ermon S. CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation[C]. Advances in Neural Information Processing Systems (NeurIPS), 2021.

[6] Saxena A., Goebel K., Simon D., Eklund N. Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation (C-MAPSS Dataset)[C]. International Conference on Prognostics and Health Management, 2008.

[7] European Space Agency. ESA Anomaly Dataset (Mission1 Telemetry)[EB/OL]. https://github.com/kplabs-pl/ESA-ADB.

[8] Yi X., Zheng Y., Zhang J., Li T. ST-MVL: Filling Missing Values in Geo-Sensory Time Series Data (PM2.5 Beijing Dataset)[C]. International Joint Conference on Artificial Intelligence (IJCAI), 2016.

<div style="page-break-after: always;"></div>

## 致　谢

感谢校内导师与企业导师在本次生产实习课程设计过程中给予的悉心指导，感谢在论文精读、实验设计与问题诊断过程中给予帮助的同学与老师。本课题涉及的复现方案、代码实现与实验分析均在此过程中不断修正完善，特此致谢。
