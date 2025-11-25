# DU-AxisCRISP

**Dual-branch Axis-aware CRISPR Indel Prediction with Long-tail Optimization**

基于双分支架构的长尾分布优化深度学习模型，用于精确预测CRISPR-Cas9编辑产生的indel（插入/删除）分布。

---


## 数据集


## ⚡ 快速使用
```bash
# 1. 创建环境
conda env create -f environment.yml
conda activate du-axiscrisp

# 2. 生成预测（使用预训练模型）
cd src
python generate_predictions.py

# 3. 查看结果
# predictions/XCRISP_testmask_deldualmodelWKL0.25_insTCN_sequenceonly_KL_Div_0.01__test.pkl
```

就这么简单！✨


## 📋 项目结构

```
DU-AxisCRISP/
├── data/                          # 数据文件
│   ├── train_new2.pkl            # 训练数据（FORECasT）
│   ├── test_new2.pkl             # 测试数据（FORECasT）
│   ├── dele_indels_sorted.pkl    # 删除Indel定义列表
│   └── 0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1.pkl  # 测试数据（inDelphi）
│   └── 数据集.zip  #数据压缩文件，解压即可
│
├── src/                          # 源代码
│   ├── common_def.py             # 通用定义、配置和工具函数
│   │
│   ├── deletion_model_train.py  # 删除模型训练脚本
│   ├── deletion_model_test.py   # 删除模型测试脚本
│   ├── ins_model_train.py       # 插入模型训练脚本
│   ├── ins_model_test.py        # 插入模型测试脚本
│   ├── generate_predictions.py  # 预测生成脚本（整合删除+插入，支持默认参数）
│   │
│   ├── models/                   # 模型架构
│   │   ├── encoder.py           # 双分支删除模型（StableLongTailDualModel）
│   │   ├── tcn.py               # TCN插入模型（AxisTCN）
│   │   ├── FeatureEncoder.py    # 特征编码器
│   │   ├── TabKANFeatureEncoder.py  # TabKAN编码器
│   │   └── nerTr.py             # Transformer编码器
│   │
│   ├── analyse/                  # 数据分析
│   │   └── analyse_dataset.py   # 数据集长尾分布分析
│   │
│   └── loss/                     # 损失函数
│       └── loss_functions.py    # 自定义损失函数
│
├── output/                       # 训练好的模型
│   ├── dual0_freq_v3_stable_WKL0.25_T0.8_KL_Div_kl_freq0.3_v2_testall_best_loss.pth
│   ├── dual0_freq_v3_stable_WKL0.25_T0.8_KL_Div_kl_freq0.3_v2_testall_best_pearson.pth
│   ├── insertion_axisTCN_Sequence-only_wkl-0.1_v2_best_loss.pth
│   ├── insertion_axisTCN_Sequence-only_wkl-0.1_v2_best_pearson.pth
│   └── 100x_indel.h5            # Lindel indel比例模型
│
├── predictions/                  # 生成的预测结果
│   ├── XCRISP_testmask_deldualmodelWKL0.25_insTCN_sequenceonly_KL_Div_0.01__test.pkl
│   └── XCRISP_testmask_deldualmodelWKL0.25_insTCN_sequenceonly_KL_Div_0.01__0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1.pkl
│
├── evaluate/                     # 模型评估与分析
│   ├── compare_models.ipynb     # 模型对比评估笔记本
│   ├── predict_results/         # 各模型预测结果
│   │   ├── DuAxis/              # DU-AxisCRISP预测
│   │   ├── XCRISP/              # X-CRISP预测
│   │   ├── inDelphi/            # inDelphi预测
│   │   ├── FORECast/            # FORECasT预测
│   │   └── Lindel/              # Lindel预测
│   └── analysis/                # 分析结果图表
│       ├── distribution.png
│       ├── performance_curves_*.png
│       └── stats_comparison.tsv
│
├── environment.yml               # Conda环境配置（推荐）
├── requirements-conda.txt        # Conda依赖列表
└── README.md                     # 本文件
```

---

## 🚀 快速开始

### 1. 环境搭建

#### 方法1: 使用 environment.yml（推荐）

```bash
# 创建并激活conda环境
conda env create -f environment.yml
conda activate du-axiscrisp
```

#### 方法2: 使用 requirements-conda.txt

```bash
# 创建环境
conda create -n du-axiscrisp python=3.9 -y
conda activate du-axiscrisp

# 安装依赖
conda install --file requirements-conda.txt -y
```

> **注意**: 如果遇到 PyTorch 安装问题，请手动安装：
> ```bash
> conda install pytorch torchvision torchaudio -c pytorch -y
> ```

#### 核心依赖

- Python 3.9
- PyTorch >= 2.0.0
- TensorFlow >= 2.10.0 (用于Lindel模型)
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- einops (张量操作)
- tqdm (进度条显示)

---

### 2. 训练模型

#### 训练删除模型

```bash
cd src
python deletion_model_train.py
```

**训练配置（在 `deletion_model_train.py` 中）：**
```python
EPOCHS = 400                 # 训练轮数
BATCH_SIZE = 64              # 批次大小
LEARNING_RATE = 0.001        # 学习率
TEMPERATURE = 0.5            # Softmax温度（固定）
FREQ_THRESHOLD = 0.3         # 高频/低频阈值
EMA_DECAY = 0.95             # EMA平滑系数
GATE_SMOOTHING = 0.7         # 门控平滑系数
```

**输出：**
- `output/dual_freq_v3_stable_T{temperature}_{loss_type}_freq{threshold}_best_loss.pth`
- `output/dual_freq_v3_stable_T{temperature}_{loss_type}_freq{threshold}_best_mcc.pth`

#### 训练插入模型

```bash
cd src
python ins_model_train.py
```

**训练配置（在 `ins_model_train.py` 中）：**
```python
EPOCHS = 200                 # 训练轮数
BATCH_SIZE = 64              # 批次大小
LEARNING_RATE = 0.01         # 学习率
WEIGHTED_KL_LAMBDA = 0.1     # 加权KL损失权重
```

**输出：**
- `output/TCN_sequenceonly_{loss_type}_{lr}_best_loss.pth`
- `output/TCN_sequenceonly_{loss_type}_{lr}_best_pearson.pth`

---

### 3. 测试模型

#### 测试删除模型

```bash
cd src

# 使用默认模型（最佳loss）
python deletion_model_test.py

# 指定模型路径
python deletion_model_test.py --model output/your_model.pth

# 测试特定数据集
python deletion_model_test.py --dataset test1  # 仅FORECasT测试集
python deletion_model_test.py --dataset test2  # 仅inDelphi测试集
python deletion_model_test.py --dataset both   # 合并测试（默认）
```

#### 测试插入模型

```bash
cd src

# 使用默认模型
python ins_model_test.py

# 指定模型和数据集
python ins_model_test.py --model output/your_model.pth --dataset test2
```

---

### 4. 生成完整预测

整合删除和插入模型，生成完整的indel预测结果。

#### 使用默认配置（推荐）

```bash
cd src

# 生成test数据集预测（使用默认模型）
python generate_predictions.py

# 生成test2数据集预测
python generate_predictions.py --dataset test2
```

**默认配置：**
- 删除模型: `output/dual0_freq_v3_stable_WKL0.25_T0.8_KL_Div_kl_freq0.3_v2_testall_best_loss.pth`
- 插入模型: `output/insertion_axisTCN_Sequence-only_wkl-0.1_v2_best_loss.pth`
- Lindel模型: `output/100x_indel.h5`

#### 自定义模型路径

```bash
python generate_predictions.py \
    --deletion_model output/your_deletion_model.pth \
    --insertion_model output/your_insertion_model.pth \
    --lindel_model output/100x_indel.h5 \
    --dataset test \
    --loss_fn KL_Div \
    --lr 0.01
```

**参数说明：**
- `--deletion_model`: 删除模型路径（可选，默认使用best_loss模型）
- `--insertion_model`: 插入模型路径（可选，默认使用best_loss模型）
- `--lindel_model`: Lindel模型路径（可选，默认使用100x_indel.h5）
- `--dataset`: 测试数据集（`test` 或 `test2`，默认：`test`）
- `--loss_fn`: 损失函数名称（用于文件命名，默认：`KL_Div`）
- `--lr`: 学习率（用于文件命名，默认：`0.01`）

**输出：**
- `predictions/XCRISP_testmask_deldualmodelWKL0.25_insTCN_sequenceonly_{loss_fn}_{lr}__{genotype}.pkl`

**特性：**
- ✅ 开箱即用的默认参数配置
- ✅ 批量Lindel预测（5-10倍性能提升）
- ✅ 预提取特征数据（减少循环内索引开销）
- ✅ 优化的进度条显示（无闪烁）
- ✅ 自动oligos文件映射

---

### 5. 数据分析

分析数据集的长尾分布特性：

```bash
cd src/analyse
python analyse_dataset.py
```

---

### 6. 模型对比评估

使用Jupyter Notebook进行多模型性能对比：

```bash
cd evaluate
jupyter notebook compare_models.ipynb
```

**对比的模型：**
- DU-AxisCRISP（本模型）
- X-CRISP
- inDelphi
- FORECasT
- Lindel

**评估指标：**
- Overall Performance（全体indel）
- MH-mediated Deletions（微同源介导的删除）
- MH-less Deletions（非同源末端连接）
- 1bp Insertions（1碱基插入）
- All Insertions（全部插入）
- Indel Frequency Prediction（indel频率预测）

---

## 🎯 模型架构

### 删除模型：StableLongTailDualModel

**核心创新：**

1. **双分支设计**
   - **Head Branch（头部分支）**: 专注高频indel
     - 保守设计，高精确度
     - 捕获主要的编辑模式
   - **Tail Branch（尾部分支）**: 专注低频indel
     - 敏感设计，高召回率
     - 捕获罕见但重要的编辑事件

2. **长尾感知门控（Frequency-aware Gating）**
   - 动态频率估计
   - 置信度评估
   - 自适应权重分配
   - 公式：`output = gate * head_logits + (1 - gate) * tail_logits`

3. **稳定性优化**
   - **EMA（指数移动平均）**: 平滑门控参数
   - **Gate Smoothing**: 限制门控值范围
   - **Fixed Temperature**: 固定Softmax温度系数

**模型结构：**
```
Input Features (6-dim)
    ├─> Head Encoder (128-dim) ──> Head Classifier ──┐
    │                                                  │
    └─> Tail Encoder (128-dim) ──> Tail Classifier ──┤
                                                       ├─> Gate ──> Weighted Output (705-dim)
                                                       │
                                                    Frequency
                                                    Estimator
```

### 插入模型：AxisTCN

基于时间卷积网络（TCN）的序列编码模型：

- **输入**: DNA序列编码（705-dim）
- **架构**: 多层扩张卷积 + 残差连接
- **输出**: 21种插入类型的概率分布
  - 1bp插入: `1+A`, `1+T`, `1+C`, `1+G`
  - 2bp插入: `2+AA`, `2+AT`, ..., `2+GG` (16种)
  - 3+bp插入: `3+X`

---

## 📊 数据格式

### 输入特征

#### 删除特征 (`FEATURE_SETS["v2"]`)

| 特征 | 说明 |
|------|------|
| `Size` | Indel大小（碱基数） |
| `leftEdge` | 左边界位置（相对于PAM） |
| `rightEdge` | 右边界位置（相对于PAM） |
| `numRepeats` | 重复序列次数 |
| `homologyLength` | 微同源长度 |
| `homologyGCContent` | 微同源区域GC含量 |

#### 序列特征

- DNA序列one-hot编码（705-dim）
- 切割位点上下文信息

### 输出格式

预测结果保存为pickle文件，包含：

```python
{
    "sample_id_1": {
        "predicted": np.array([...]),    # 预测概率分布
        "actual": np.array([...]),       # 真实概率分布
        "indels": ["1+1", "1+2", ...],   # Indel标签列表
        "mh": [True, False, ...]         # 是否为微同源介导
    },
    "sample_id_2": {...},
    ...
}
```

---

## 📈 评估指标

### 回归指标

- **Pearson Correlation**: 线性相关性
- **Spearman Correlation**: 秩相关性
- **MSE (Mean Squared Error)**: 均方误差
- **KL Divergence**: 分布距离（KL散度）
- **Jensen-Shannon Distance**: 对称分布距离

### 分类指标（高频indel识别）

- **MCC (Matthews Correlation Coefficient)**: 平衡评估指标
- **Precision**: 精确率
- **Recall**: 召回率
- **F1 Score**: F1分数

---

## 🔬 实验结果

### 主要性能对比（Overall Performance）

**FORECasT测试集：**

| 模型 | Pearson ↑ | KL Divergence ↓ | Jensen-Shannon ↓ |
|------|-----------|-----------------|------------------|
| X-CRISP | 0.824 | 0.860 | 0.403 |
| inDelphi | 0.826 | 0.832 | 0.398 |
| FORECasT | 0.829 | 0.742 | 0.381 |
| Lindel | 0.801 | 0.921 | 0.427 |
| **DU-AxisCRISP** | **0.843** | **0.592** | **0.355** |

**inDelphi测试集：**

| 模型 | Pearson ↑ | KL Divergence ↓ | Jensen-Shannon ↓ |
|------|-----------|-----------------|------------------|
| X-CRISP | 0.751 | 1.124 | 0.468 |
| inDelphi | 0.841 | 0.727 | 0.373 |
| FORECasT | 0.798 | 0.893 | 0.419 |
| Lindel | 0.723 | 1.203 | 0.498 |
| **DU-AxisCRISP** | **0.839** | **0.658** | **0.361** |

### MH-mediated Deletions（微同源介导的删除）

| 模型 | Pearson | KL Divergence | Jensen-Shannon |
|------|---------|---------------|----------------|
| X-CRISP | 0.831 | 0.824 | 0.393 |
| inDelphi | 0.846 | 0.718 | 0.368 |
| **DU-AxisCRISP** | **0.857** | **0.573** | **0.342** |

### 1bp Insertions（1碱基插入）

| 模型 | Pearson | KL Divergence | Jensen-Shannon |
|------|---------|---------------|----------------|
| X-CRISP | 0.723 | 1.156 | 0.487 |
| inDelphi | 0.761 | 0.982 | 0.441 |
| **DU-AxisCRISP** | **0.778** | **0.891** | **0.419** |

---

## 💡 主要特点

### 🎯 模型创新
✅ **长尾优化**: 专门针对CRISPR indel分布的长尾特性设计  
✅ **双分支架构**: 同时优化高频和低频indel的预测  
✅ **稳定训练**: EMA和门控平滑确保训练稳定性  
✅ **频率感知门控**: 动态调整高频/低频分支权重

### 🚀 工程优化
✅ **批量预测**: Lindel模型批量处理，5-10倍性能提升  
✅ **特征预提取**: 减少循环内索引开销，显著提升预测速度  
✅ **优化进度条**: 无闪烁的友好进度显示  
✅ **默认配置**: 开箱即用，无需指定模型路径

### 🔧 易用性
✅ **模块化设计**: 删除和插入模型独立训练，灵活组合  
✅ **完整评估流程**: 提供预训练模型和Jupyter评估笔记本  
✅ **清晰代码结构**: 关键业务逻辑注释完整  
✅ **高性能**: 在多个基准测试中超越现有方法  

---

## 📚 相关工作

- **inDelphi**: Shen et al. (2018) - *Nature*
- **FORECasT**: Allen et al. (2019) - *Nature Biotechnology*
- **Lindel**: Chen et al. (2019) - *Nucleic Acids Research*
- **X-CRISP**: Original baseline model

---

## 📄 许可证

MIT License

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

## 📧 联系方式

如有问题或建议，请联系：[your.email@example.com]

---

**Happy Predicting! 🎉🧬**
