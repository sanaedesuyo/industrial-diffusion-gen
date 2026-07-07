# TSGM 论文复现方案

> **论文**：*Regular Time-series Generation using SGM* (TSGM, Lim et al., AAAI 2023, arXiv:2301.08518)
> **项目**：`industrial-diffusion-gen`（基于扩散/Score-based 模型的工业时序数据生成）
> **本文档目标**：给出可落地的 TSGM 复现方案——论文精读、数据集适配评估、代码架构、训练/采样/评估流程、超参、里程碑与风险。

## 决策基线（已确认）

- **评估口径**：ESA Anomaly Dataset 作为测试集，采用**标准协议**——在 ESA 自身 train/test 划分内计算 discriminative / predictive / t-SNE。
- **首轮复现范围（核心）**：**C-MAPSS + PM25**。其余数据集（SMAP / STFT / HallThruster / PhysioNet）列为可选扩展。
- **交付**：本方案文档（不在本阶段搭建代码骨架）。

---

## 一、论文核心解读

### 1.1 定位
TSGM 是**首个用 SGM（score-based / diffusion）做时序"生成"**的方法，区别于已有的 SGM 时序"预测"（TimeGrad / ScoreGrad）与"插补"（CSDI）。它面向**规则采样（regular，等间隔、可切定长窗口）的多元时序**。核心区别（论文 Table 1）：目标 score 为**隐空间中、以上一时刻隐向量为条件**的条件分数 $\nabla \log p(h^s_t \mid h_{t-1})$。

### 1.2 三大组件
1. **Encoder / Decoder（RNN 自编码器）**：把时序 $x_{1:T}$ 递归映射到隐空间 $h_{1:T}$ 再还原。$h_t=e(h_{t-1},x_t)$，$\hat x_t=d(h_t)$。用 GRU/LSTM 实现。
2. **Conditional Score Network $M_\theta(s, h^s_t, h_{t-1})$**：把图像用的 2D U-Net 改成 **1D U-Net**；输入为 [被扩散隐向量 $h^s_t$ ⊕ 条件 $h_{t-1}$] + 扩散时间 $s$ 的时间嵌入。
3. **前向/反向 SDE**：采用 **VP** 与 **subVP** 两种（论文明确排除 VE，因性能差）。

### 1.3 两个损失
- 自编码器重构损失（式3）：$\mathcal{L}_{ed}=\mathbb{E}\|\hat x_{1:T}-x_{1:T}\|_2^2$
- 隐空间条件去噪分数匹配（主贡献，Theorem 1 + Corollary 1，式5），约定 $h_0=0$，$\lambda(s)$ 沿用 Song et al. (2021)：

$$\mathcal{L}^H_{score}=\mathbb{E}_s\mathbb{E}_{h_{1:T}}\sum_{t=1}^{T}\lambda(s)\,\mathbb{E}_{h^s_t}\big\|M_\theta(s,h^s_t,h_{t-1})-\nabla_{h^s_t}\log p(h^s_t\mid h_t)\big\|_2^2$$

### 1.4 训练算法（论文 Algorithm 1）
1. **预训练**：迭代 `iter_pre` 次，仅用 $\mathcal{L}_{ed}$ 训练 Encoder+Decoder。
2. **主训练**：迭代 `iter_main` 次，用 $\mathcal{L}^H_{score}$ 训练 score 网络；若 `use_alt=True`，每步再用 $\mathcal{L}_{ed}$ 交替微调 AE。
3. `use_alt` 为逐数据集超参（部分数据集只训 score 更好）。

### 1.5 采样流程（递归 + Predictor-Corrector）
- $t=1$：从高斯先验采 $z_0$，与 $h_0=0$ 拼接，用 PC sampler 反解出 $\hat h_1$。
- $t=2..T$：读入上一步生成的 $\hat h_{t-1}$，采 $z_{t-1}$，拼接后经条件反向 SDE（PC，默认 N=1000 步）生成 $\hat h_t$。
- 得到全部 $\hat h_{1:T}$ 后，用 Decoder 一次性还原 $\hat x_{1:T}$。

### 1.6 评估指标（严格沿用 TimeGAN 协议，越低越好）
- **Discriminative score**：训练 2 层 LSTM 二分类器区分真/假，报 $|\text{acc}-0.5|$。
- **Predictive score（TSTR）**：用合成数据训练 LSTM 预测器，在真实测试集测 MAE。
- **t-SNE 可视化**：真/假样本重合度（多样性）。
- 每方法跑 **10 个随机种子**，报 mean±std；每 5000 iter 选一次最优模型。

### 1.7 关键超参（论文 Table 2，窗口长度 T=24）
| 数据集 | d_in | d_t | use_alt | iter_pre | iter_main |
|---|---|---|---|---|---|
| Stocks | 24 | 96 | True | 50000 | 40000 |
| Energy | 56 | 56 | False | — | 100000 |
| Air | 40 | 80 | True | 50000 | — |

`d_hidden` 取输入维度的 2~5 倍；其余沿用 TimeGAN + VPSDE 默认。采样步数可 1000/500/250/100（subVP 100 步仍可用）。

---

## 二、数据集分析与适配性评估

TSGM 前提：**规则（等间隔）、定长可窗口化的连续多元时序**。逐一评估：

| 数据集 | 内容 | 采样规则性 | 匹配度 | 结论 |
|---|---|---|---|---|
| **C-MAPSS** | 涡扇发动机退化仿真，21 传感器+3 工况，多机组逐 cycle | 规则（逐 cycle） | ★★★★★ | **核心训练集** |
| **PM25** | 北京 PM2.5 逐小时多站点（STMVL 数据） | 规则（逐小时） | ★★★★ | **核心训练集**（对应论文 Air 场景） |
| **SMAP** | NASA 航天器遥测，连续通道+指令 one-hot | 规则 | ★★★☆ | 可选扩展，取连续遥测通道 |
| **STFT** | 高频振动/声学 CSV，3.2GB | 规则（高采样率） | ★★★ | 可选/进阶，需降采样+分段 |
| **1D-HallThruster** | **Julia 仿真器代码**（HallThruster.jl），非现成数据 | — | ★★★（作数据源） | 进阶数据生成源，需 Julia 环境 |
| **PhysioNet** | ICU 记录，`Time,Parameter,Value` 稀疏格式 | **不规则、稀疏、缺失多** | ★★ | 与"regular"前提冲突；论文列为 future work，需重采样，低优先级 |
| **ESA Anomaly** | 卫星遥测多通道+异常标注，~12GB | 规则遥测 | 用于评估 | **测试集**（标准协议） |

**关键提醒**：
- `1D-HallThruster` 是仿真代码而非数据，需运行 Julia 才能产出时序。
- `PhysioNet` 不规则，直接使用违背论文假设。
- `C-MAPSS / SMAP` 存在恒定传感器列，须剔除以免归一化除零。

---

## 三、数据集选型与划分

- **核心训练/开发集**：C-MAPSS（FD001 起步）、PM25。
- **测试/评估集**：ESA Anomaly Dataset（标准协议：ESA 自身 train/test 内算 disc/pred/t-SNE）。
- **可选扩展**：SMAP、STFT、HallThruster 仿真、PhysioNet（不规则扩展实验）。

---

## 四、代码复现方案

### 4.1 目录结构（对齐 README 规划）
```
industrial-diffusion-gen/
├── configs/                 # 每数据集一份 YAML + 模型/SDE 配置
│   ├── cmapss.yaml
│   └── pm25.yaml
├── data/
│   ├── raw/                 # 解压后的原始数据（git 忽略）
│   ├── processed/           # 预处理后的 .npy 窗口张量
│   └── loaders/
│       ├── base.py          # MinMax 归一化 + 滑窗 + 划分（通用）
│       ├── cmapss.py
│       ├── pm25.py
│       └── esa.py
├── models/
│   ├── autoencoder.py       # GRU Encoder/Decoder
│   ├── score_unet1d.py      # 1D U-Net 条件分数网络 + 时间嵌入
│   ├── sde.py               # VP / subVP SDE
│   ├── ema.py               # 参数 EMA
│   └── tsgm.py              # 组装：训练/采样 API
├── scripts/
│   ├── prepare_data.py      # 解压 + 预处理
│   ├── train.py             # 两阶段训练（Algorithm 1）
│   ├── sample.py            # 递归 PC 采样
│   └── evaluate.py          # disc/pred/t-SNE
├── metrics/
│   ├── discriminative.py
│   ├── predictive.py
│   └── visualization.py
├── outputs/                 # checkpoints / 样本 / 图表（git 忽略）
├── requirements.txt
└── README.md
```

### 4.2 数据管道（`data/`）
通用流程（`base.py`）：
1. 选连续特征列，剔除常量/近常量列。
2. 缺失处理：PM25 前向填充+线性插值；PhysioNet 重采样到小时网格（如启用）。
3. **MinMax 归一化到 [0,1]**（逐特征，沿用 TimeGAN）。
4. **滑窗**：窗口长度 `T=24`，步长可配；C-MAPSS 按机组分别切窗，避免跨机组穿越。
5. 打乱、划分 train/test，存 `data/processed/<name>/{train,test}.npy`，形状 `[N, T, D]`。
6. 保存归一化器（用于反归一化与评估）。

各数据集要点：
- **C-MAPSS**：26 列 = unit + cycle + 3 op settings + 21 sensors；按 unit 分组切窗，取非常量 sensor（±op settings），D≈14~17。
- **PM25**：逐小时 PM2.5（多站点为多变量），插补后切窗。
- **ESA**：解压对应 mission，读取 `channels/channel_*.zip`，用 `channels.csv`/`anomaly_types.csv` 选正常段构建 test 参照。

### 4.3 模型实现（`models/`）
- **autoencoder.py**：多层 GRU Encoder（`x_{1:T}→h_{1:T}`）+ GRU Decoder（`h_{1:T}→x̂_{1:T}`）；隐维 `d_hidden = k·D, k∈[2,5]`。
- **sde.py**：VP、subVP 的 `f(s,·)`、`g(s)`、边缘条件分布 `p(x^s|x^0)` 的 mean/std（解析）、`λ(s)` 权重、反向 SDE 系数。
- **score_unet1d.py**：1D U-Net（depth=4，可切 3 做敏感性实验）；输入通道 = [被扩散隐向量 ⊕ 条件 `h_{t-1}`]，注入 `s` 的正弦时间嵌入（FiLM/加性）。
- **tsgm.py**：
  - `train_autoencoder()` → $\mathcal{L}_{ed}$
  - `train_score()` → $\mathcal{L}^H_{score}$（对 t 求和，`h_0=0`，条件 `h_{t-1}` 需 detach）
  - `sample()` → 递归 PC 采样器（Predictor: reverse-diffusion / Euler-Maruyama；Corrector: Langevin），N 步可配。
  - 采样使用 EMA 权重。

### 4.4 训练流程（`scripts/train.py`，对齐 Algorithm 1）
1. 读 config → 载数据 → 建模型。
2. 预训练 AE `iter_pre` 步。
3. 主训练 score `iter_main` 步；`use_alt=True` 时交替微调 AE。
4. 每 5000 步保存 checkpoint + 采样一批 + 记录 disc/pred（用于选最优模型）。
5. 日志 TensorBoard/CSV；支持断点续训。

### 4.5 采样流程（`scripts/sample.py`）
- 递归 + PC 生成 `h_{1:T}` → Decoder 还原 `x̂_{1:T}` → 反归一化 → 存 `.npy`。
- 参数：`n_samples`、`sde_type`（vp/subvp）、`n_steps`。

### 4.6 评估（`metrics/` + `scripts/evaluate.py`）
- **discriminative.py**：2 层 LSTM 分类器，输出 `|acc−0.5|`。
- **predictive.py**：TSTR，合成训练 LSTM 预测器，真实测试集测 MAE。
- **visualization.py**：真/假样本 t-SNE（+ PCA 备选）。
- 每指标跑 10 seeds，输出 mean±std 表 + 图到 `outputs/reports/`。

---

## 五、环境与依赖

```
python>=3.9
torch>=2.0
numpy, pandas, scipy
scikit-learn
matplotlib
pyyaml, tqdm
tensorboard
h5py                  # 读 SMAP .h5（若启用）
# 可选：julia + HallThruster.jl（进阶仿真数据源）
```
硬件：单张 GPU（论文用 RTX 2080Ti 11GB；TSGM 训练显存约 4GB）。SGM 采样较慢，调试期先用小规模/小步数验证。

---

## 六、里程碑（推进顺序）

1. **M1 数据打通**：`prepare_data.py` 跑通 C-MAPSS FD001 → `[N,24,D]`，健全性检查。
2. **M2 AE 复现**：GRU 自编码器重构误差收敛（`x̂≈x`）。
3. **M3 Score + SDE**：VP 版 score 网络训练、单窗口 PC 采样跑通。
4. **M4 递归采样**：完整 `h_{1:T}` 递归生成 + Decoder 还原。
5. **M5 评估管线**：disc/pred/t-SNE 在 C-MAPSS 出第一版数值。
6. **M6 扩展到 PM25**。
7. **M7 ESA 测试基准**：标准协议在 ESA 上评估。
8. **M8 消融/敏感性**：w/o 预训练、U-Net depth 3、采样步 500/250/100、subVP vs VP；补齐 mean±std 表。

---

## 七、风险与注意事项

- **规则性前提**：PhysioNet 不规则，若纳入须先重采样。
- **常量列**：C-MAPSS/SMAP 有恒定传感器，需剔除否则归一化除零。
- **采样速度**：SGM 采样慢，调试期先 `n_steps=100`（subVP 论文显示 100 步仍可用）。
- **条件泄漏**：score 训练时条件 `h_{t-1}` 需 detach。
- **官方代码缺失**：论文 under review 未附仓库，按公式 + TimeGAN/VPSDE 默认实现；VPSDE 的 SDE/PC sampler 可参考 Song et al. (2021) `score_sde`。
- **数据体量**：ESA/STFT 十几 GB，解压与预处理需规划磁盘/时间，先抽子集验证。

---

## 附：论文与数据速查

- **方法**：TSGM = RNN 自编码器 + 隐空间条件 score 网络（1D U-Net）+ VP/subVP SDE + 递归 PC 采样。
- **窗口长度**：T=24。
- **核心创新**：面向时序生成的去噪分数匹配（Theorem 1）与递归条件采样。
- **评估**：discriminative / predictive（TSTR）/ t-SNE，10 seeds，越低越好。
