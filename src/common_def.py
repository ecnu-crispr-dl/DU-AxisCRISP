import os, sys
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import kl_div, mse_loss
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef, f1_score, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from tqdm import trange
import time
from scipy.stats import pearsonr
import math
import pickle as pkl
from tqdm import tqdm

#cpu/gpu win/mac
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if DEVICE.type == 'cpu': DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 

# 动态路径配置 - 基于 common_def.py 所在位置计算项目根目录
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)  # DU-AxisCRISP/

#FORECast DATA
train_file_path = os.path.join(_PROJECT_ROOT, "data", "train_new2.pkl")
test_file_path = os.path.join(_PROJECT_ROOT, "data", "test_new2.pkl")
t1_path = test_file_path  # 别名

#inDelphi DATA
t2_path = os.path.join(_PROJECT_ROOT, "data", "0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1.pkl")
indels_sorted_path = os.path.join(_PROJECT_ROOT, "data", "dele_indels_sorted.pkl")

output_dir = "output/"
FEATURE_SETS = {
    "insv1": ["Size", "Start", "leftEdge", "rightEdge", "leftEdgeMostDownstream", "rightEdgeMostUpstream", "InsSize",
              "InsGC", "InsSeqLen", "InsIsTemplated", "InsertPositionRelative", "IsPalindrome", "LeftGC", "RightGC",
              "InsSeqEncoded"],
    "full+numrepeat": ["Size", "Start", "homologyLength", "numRepeats", "homologyGCContent", "homologyDistanceRank",
                       "homologyLeftEdgeRank", "homologyRightEdgeRank", "homologyLengthRank"],
    "full": ["Size", "Start", "homologyLength", "homologyGCContent", "homologyDistanceRank", "homologyLeftEdgeRank",
             "homologyRightEdgeRank", "homologyLengthRank"],
    "v2": ["Size", "leftEdge", "rightEdge", "numRepeats", "homologyLength", "homologyGCContent"],
    "v3": ["Size", "leftEdge", "rightEdge", "homologyLength", "homologyGCContent"],
    "v4": ["Gap", "leftEdge", "rightEdge", "homologyLength", "homologyGCContent"],
    "v5": ["leftEdge", "rightEdge", "homologyLength", "homologyGCContent"],
    "v6": ["leftEdge", "Gap", "homologyLength", "homologyGCContent"],
    "v7": ["leftEdgeMostDownstream", "rightEdgeMostUpstream", "homologyLength", "homologyGCContent"],
    "ranked": ["Size", "numRepeats", "homologyLength", "homologyGCContent", "homologyLeftEdgeRank",
               "homologyRightEdgeRank"],
    "v2+ranked": ["Size", "leftEdge", "rightEdge", "numRepeats", "homologyLength", "homologyGCContent",
                  "homologyLeftEdgeRank", "homologyRightEdgeRank"]
}

def _to_tensor(arr):
    if isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series):
        arr = arr.to_numpy()
    return torch.tensor(arr).to(DEVICE).float()


def merge_tests(
    test1_path,
    test2_path,
    indel_list,
    mode="union",            # 'union' 或 'intersection'
    tag=("T1","T2"),         # 给样本加后缀，避免同名冲突
    fractions=True,
    assert_same_seq_on_intersection=True
):
    # 复用你已有的加载函数
    X1, y1, s1, seq1 = load_delete_data(test1_path, num_samples=None, fractions=True, indel_list=indel_list)
    X2, y2, s2, seq2 = load_delete_data_inDelphi(test2_path, num_samples=None, fractions=True, indel_list=indel_list)

    # 安全检查：列一致
    assert list(X1.columns) == list(X2.columns), "特征列不一致，请先对齐"
    # MultiIndex的两个层级名也应一致
    assert X1.index.names == X2.index.names, "MultiIndex 名称不一致，请先对齐"

    # 当前索引是 MultiIndex: (sample, indel)
    # 先把 sample 层拿出来
    sample_level = 0  # (sample, indel) 中 sample 在第0层

    def add_tag_to_samples(obj, tag_str, sample_level=0):
        """
        obj: DataFrame 或 Series，索引为 MultiIndex: (sample, indel)
        """
        import pandas as pd
        idx = obj.index
        if not isinstance(idx, pd.MultiIndex):
            raise ValueError("期望 MultiIndex 索引 (sample, indel)")

        # 复制一份 levels，修改第 sample_level 层后整体设置回去
        new_levels = list(idx.levels)
        new_levels[sample_level] = new_levels[sample_level].astype(str) + f"@{tag_str}"

        new_index = idx.set_levels(new_levels)  # 注意：是对 idx 调用，而不是 df
        obj = obj.copy()
        obj.index = new_index
        return obj

    # 两种模式：并集 或 交集
    if mode == "union":
        X1_u = add_tag_to_samples(X1, tag[0])
        y1_u = add_tag_to_samples(y1, tag[0])
        X2_u = add_tag_to_samples(X2, tag[1])
        y2_u = add_tag_to_samples(y2, tag[1])

        X = pd.concat([X1_u, X2_u], axis=0).sort_index()
        y = pd.concat([y1_u, y2_u], axis=0).sort_index()

        # seq_features 的行索引是 sample 维，按同样规则加后缀再拼
        seq1_u = seq1.copy()
        seq1_u.index = seq1_u.index.astype(str) + f"@{tag[0]}"
        seq2_u = seq2.copy()
        seq2_u.index = seq2_u.index.astype(str) + f"@{tag[1]}"
        seq = pd.concat([seq1_u, seq2_u], axis=0)

        samples = list(seq.index)

    elif mode == "intersection":
        # 找到两测试集的同名样本交集
        common_samples = sorted(set(s1).intersection(set(s2)))
        if assert_same_seq_on_intersection and len(common_samples) > 0:
            # 确认交集样本的序列特征一致
            a = seq1.loc[common_samples].sort_index()
            b = seq2.loc[common_samples].sort_index()
            if not a.equals(b):
                raise ValueError("交集样本的序列特征不一致，请不要直接并用 'intersection'。")

        # 直接用同名样本（不加后缀），只保留交集样本的 (sample, indel)
        idx1 = X1.index.get_level_values(0).isin(common_samples)
        idx2 = X2.index.get_level_values(0).isin(common_samples)

        X = pd.concat([X1[idx1], X2[idx2]], axis=0).sort_index()
        y = pd.concat([y1[idx1], y2[idx2]], axis=0).sort_index()
        # 也可以选择“取平均/取一份”，这里保留两份记录（来自不同来源），评估时每条样本独立

        # 合并交集的 seq_features（两侧一致）
        seq = seq1.loc[common_samples].copy()
        samples = common_samples
    else:
        raise ValueError("mode 只能是 'union' 或 'intersection'")

    # 最终返回：拼接后的特征/标签/样本名/序列特征
    return X, y, samples, seq

#data load
def load_delete_data_inDelphi(filepath = None, num_samples=None, fractions=True, indel_list=None):
    data = pd.read_pickle(filepath)
    counts = data["counts"]
    del_features = data["del_features"]
    seq_features = data["ins_features"]

    # 筛选样本
    samples = counts.index.levels[0]
    if num_samples is not None:
        samples = samples[:num_samples]
        counts = counts.loc[samples]
        del_features = del_features.loc[samples]
    if seq_features is not None:
        # 重要：顺序对齐到 samples
        seq_features = seq_features.reindex(samples)

    # 选择 DELETION 数据
    y_df = counts.loc[counts.Type == "DELETION"]
    y = y_df.fraction if fractions else y_df.countEvents

    # 填充 Gap 特征
    del_features["Gap"] = del_features["Size"] - del_features["homologyLength"]

    # 构建完整的 MultiIndex 索引
    # 使用样本和 indel_list 的笛卡尔积，构造完整索引
    index = pd.MultiIndex.from_product([samples, indel_list], names=del_features.index.names)

    # reindex del_features，补全缺失的 sample-indel 对
    X_full = del_features.reindex(index, fill_value=0)

    # 生成 mask：原始数据存在则为1，否则为0
    mask = (~X_full["Size"].eq(0)).astype(float)
    X_full = X_full.copy()
    X_full["mask"] = mask

    # 对 y 也 reindex 补全
    y_full = y.reindex(index, fill_value=0)
    '''with open(ALL_FEAT_PKL, "rb") as f:
        bundle = pkl.load(f)
    bio_features: pd.DataFrame = bundle["features"]  # index=Oligo_ID, columns=全部特征
    # 缺失已经在生成 pkl 时置 0，这里再保险一次
    bio_features = bio_features.reindex(samples).fillna(0.0)'''

    return X_full, y_full, samples, seq_features
def load_delete_data(filepath = None, num_samples=None, fractions=True, indel_list=None):
    data = pd.read_pickle(filepath)
    counts = data["counts"]
    del_features = data["del_features"]
    
    # 安全获取序列特征：优先使用 seq_features，如果不存在则使用 ins_features
    seq_features = data.get("seq_features", None)
    if seq_features is None and "ins_features" in data:
        seq_features = data["ins_features"]  # 第二个数据集中，ins_features 就是 seq_features

    # 筛选样本
    samples = counts.index.levels[0]
    if num_samples is not None:
        samples = samples[:num_samples]
        counts = counts.loc[samples]
        del_features = del_features.loc[samples]
    if seq_features is not None:
        # 重要：顺序对齐到 samples
        seq_features = seq_features.reindex(samples)

    # 选择 DELETION 数据
    y_df = counts.loc[counts.Type == "DELETION"]
    y = y_df.fraction if fractions else y_df.countEvents

    # 填充 Gap 特征
    del_features["Gap"] = del_features["Size"] - del_features["homologyLength"]

    # 构建完整的 MultiIndex 索引
    # 使用样本和 indel_list 的笛卡尔积，构造完整索引
    index = pd.MultiIndex.from_product([samples, indel_list], names=del_features.index.names)

    # reindex del_features，补全缺失的 sample-indel 对
    X_full = del_features.reindex(index, fill_value=0)

    # 生成 mask：原始数据存在则为1，否则为0
    mask = (~X_full["Size"].eq(0)).astype(float)
    X_full = X_full.copy()
    X_full["mask"] = mask

    # 对 y 也 reindex 补全
    y_full = y.reindex(index, fill_value=0)
    '''with open(ALL_FEAT_PKL, "rb") as f:
        bundle = pkl.load(f)
    bio_features: pd.DataFrame = bundle["features"]  # index=Oligo_ID, columns=全部特征
    # 缺失已经在生成 pkl 时置 0，这里再保险一次
    bio_features = bio_features.reindex(samples).fillna(0.0)'''

    return X_full, y_full, samples, seq_features
def load_insert_data(filepath = None, num_samples=None, fractions=True, indel_list=None):
    data = pd.read_pickle(filepath)
    counts = data["counts"]
    ins_features = data["ins_features"]
    seq_features = data.get("seq_features", None)  # 安全获取，可能不存在

    # 筛选样本
    samples = counts.index.levels[0]
    if num_samples is not None:
        samples = samples[:num_samples]
        counts = counts.loc[samples]
        ins_features = ins_features.loc[samples]
    if seq_features is not None:
        # 重要：顺序对齐到 samples
        seq_features = seq_features.reindex(samples)

    # 选择 DELETION 数据
    y_df = counts.loc[counts.Type == "INSERTION"]
    y = y_df.fraction if fractions else y_df.countEvents

    # 填充 Gap 特征
    ins_features["Gap"] = ins_features["Size"] - ins_features["homologyLength"]

    # 构建完整的 MultiIndex 索引
    # 使用样本和 indel_list 的笛卡尔积，构造完整索引
    index = pd.MultiIndex.from_product([samples, indel_list], names=ins_features.index.names)

    # reindex del_features，补全缺失的 sample-indel 对
    X_full = ins_features.reindex(index, fill_value=0)

    # 生成 mask：原始数据存在则为1，否则为0
    mask = (~X_full["Size"].eq(0)).astype(float)
    X_full = X_full.copy()
    X_full["mask"] = mask

    # 对 y 也 reindex 补全
    y_full = y.reindex(index, fill_value=0)
    '''with open(ALL_FEAT_PKL, "rb") as f:
        bundle = pkl.load(f)
    bio_features: pd.DataFrame = bundle["features"]  # index=Oligo_ID, columns=全部特征
    # 缺失已经在生成 pkl 时置 0，这里再保险一次
    bio_features = bio_features.reindex(samples).fillna(0.0)'''
    #first_6_nt_feature_indices = list(range(0, 56)) + list(range(80, 304))
    return X_full, y_full, samples, seq_features#.iloc[:,first_6_nt_feature_indices]
def load_insdel_data(filepath = None, num_samples=None, fractions=True):
    data = pd.read_pickle(filepath)
    counts = data["counts"]
    ins_features = data["ins_features"]
    seq_features = data["seq_features"]

    # 筛选样本
    samples = counts.index.levels[0]
    if num_samples is not None:
        samples = samples[:num_samples]
        counts = counts.loc[samples]
        ins_features = ins_features.loc[samples]
    if seq_features is not None:
        # 重要：顺序对齐到 samples
        seq_features = seq_features.reindex(samples)
    # 选择度量
    val_col = "fraction" if fractions else "countEvents"
    # 分家族聚合到样本级
    ins_sum = counts[counts["Type"] == "INSERTION"].groupby(level=0)[val_col].sum()
    del_sum = counts[counts["Type"] == "DELETION"].groupby(level=0)[val_col].sum()

    # 填充 Gap 特征
    ins_features["Gap"] = ins_features["Size"] - ins_features["homologyLength"]

    # 构建完整的 MultiIndex 索引
    # 使用样本和 indel_list 的笛卡尔积，构造完整索引
    #index = pd.MultiIndex.from_product([samples, indel_list], names=ins_features.index.names)

    # reindex del_features，补全缺失的 sample-indel 对
    X_full = ins_features#.reindex(index, fill_value=0)

    # 生成 mask：原始数据存在则为1，否则为0
    mask = (~X_full["Size"].eq(0)).astype(float)
    X_full = X_full.copy()
    X_full["mask"] = mask

    total = (ins_sum + del_sum).replace(0, np.finfo(float).eps)
    y_ins = (ins_sum / total).rename("INS")
    y_del = (del_sum / total).rename("DEL")
    y2_df = pd.concat([y_ins, y_del], axis=1)

    # 数值稳定（保证和为1）
    y2_df["INS"] = y2_df["INS"].clip(0, 1)
    y2_df["DEL"] = 1.0 - y2_df["INS"]
    '''with open(ALL_FEAT_PKL, "rb") as f:
        bundle = pkl.load(f)
    bio_features: pd.DataFrame = bundle["features"]  # index=Oligo_ID, columns=全部特征
    # 缺失已经在生成 pkl 时置 0，这里再保险一次
    bio_features = bio_features.reindex(samples).fillna(0.0)'''
    #first_6_nt_feature_indices = list(range(0, 56)) + list(range(80, 304))
    return X_full, y2_df, samples, seq_features#.iloc[:,first_6_nt_feature_indices]

def load_delete_data_include_dnaFeatures(filepath = None, dna_feature_path = None, num_samples=None, fractions=True, indel_list=None):
    data = pd.read_pickle(filepath)
    counts = data["counts"]
    del_features = data["del_features"]
    seq_features = data["seq_features"]    

    # 筛选样本
    samples = counts.index.levels[0]
    if num_samples is not None:
        samples = samples[:num_samples]
        counts = counts.loc[samples]
        del_features = del_features.loc[samples]
    
    # ✅ 从dna_feature_path读取DNA features
    if dna_feature_path is not None:
        # 读取DNA features文件
        dna_features = pd.read_pickle(dna_feature_path)
        # DNA features的索引应该是Oligo_ID，需要对齐到samples
        seq_features = dna_features.reindex(samples, fill_value=0.0)
    elif seq_features is not None:
        # 原有逻辑：从data中读取的seq_features
        # 重要：顺序对齐到 samples
        seq_features = seq_features.reindex(samples)

    # 选择 DELETION 数据
    y_df = counts.loc[counts.Type == "DELETION"]
    y = y_df.fraction if fractions else y_df.countEvents

    # 填充 Gap 特征
    del_features["Gap"] = del_features["Size"] - del_features["homologyLength"]

    # 构建完整的 MultiIndex 索引
    # 使用样本和 indel_list 的笛卡尔积，构造完整索引
    index = pd.MultiIndex.from_product([samples, indel_list], names=del_features.index.names)

    # reindex del_features，补全缺失的 sample-indel 对
    X_full = del_features.reindex(index, fill_value=0)

    # 生成 mask：原始数据存在则为1，否则为0
    mask = (~X_full["Size"].eq(0)).astype(float)
    X_full = X_full.copy()
    X_full["mask"] = mask

    # 对 y 也 reindex 补全
    y_full = y.reindex(index, fill_value=0)
    '''with open(ALL_FEAT_PKL, "rb") as f:
        bundle = pkl.load(f)
    bio_features: pd.DataFrame = bundle["features"]  # index=Oligo_ID, columns=全部特征
    # 缺失已经在生成 pkl 时置 0，这里再保险一次
    bio_features = bio_features.reindex(samples).fillna(0.0)'''

    return X_full, y_full, samples, seq_features


def batch_model(X, seq_full, Y, samples, model, loss_fn, optimizer, lr_scheduler):
    model.train()
    loss = torch.zeros(1).to(DEVICE)

    # 构造 batch 输入 (B, N, F)
    x_batch = torch.stack([_to_tensor(X.loc[s]) for s in samples])  # (batch, num_indels, num_features)
    y_batch = torch.stack([_to_tensor(Y.loc[s]) for s in samples])  # (batch, num_indels)
    seq_batch = torch.stack([_to_tensor(seq_full.loc[s]) for s in samples])

    y_batch = y_batch / (y_batch.sum(dim=1, keepdim=True) + 1e-10)

    # Forward pass
    y_pred_batch= model(x_batch, seq_batch)


    # 逐个样本计算 loss 并加和
    for i in range(len(samples)):
        y_pred = y_pred_batch[i]
        y_true = y_batch[i]

        if loss_fn == "KL_Div":
            # PyTorch的kl_div需要第一个参数是log概率
            y_pred_clamped = torch.clamp(y_pred, min=1e-8, max=1-1e-8)
            loss += kl_div(torch.log(y_pred_clamped), y_true, reduction='batchmean')
        else:
            loss += loss_fn(y_pred, y_true)

    loss = torch.div(loss, len(samples))  # loss = loss / len(samples)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    lr_scheduler.step()

    return loss.cpu().detach().numpy().item()


def analyze_sample_zeros(test_y):
    sample_stats = test_y.groupby(level='Sample_Name').agg({
        'total_indels': 'size',
        'zero_indels': lambda x: (x == 0).sum(),
        'non_zero_indels': lambda x: (x != 0).sum(),
        'all_zeros': lambda x: np.all(x == 0),
        'max_fraction': 'max',
        'min_fraction': 'min'
    })

    # 重命名列
    sample_stats = sample_stats.rename(columns={
        'total_indels': '总Indel数',
        'zero_indels': '零值Indel数',
        'non_zero_indels': '非零Indel数',
        'all_zeros': '是否全为零',
        'max_fraction': '最大频率',
        'min_fraction': '最小频率'
    })

    return sample_stats

def test_model(model, X, Y, samples, seq_features, indel_list,A):
    """
    测试模型性能
    Args:
        model: 训练好的模型
        X: 特征数据
        Y: 标签数据
        samples: 样本列表
        seq_features: 序列特征
        indel_list: indel列表
    
    Returns:
        regression_metrics, classification_metrics: 回归和分类指标
    """
    model.eval()
    
    # 回归指标
    correlations = []
    kl_divergences = []
    mse=[]

    # 分类指标数据
    y_true_list = []
    y_pred_list = []
    THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    with torch.no_grad():
        for i in range(len(samples)):
            sample = samples[i]

            x_data = X.loc[sample]
            y_data = Y.loc[sample]    

            x = _to_tensor(x_data)  
            y = _to_tensor(y_data)  
                
            # 添加batch维度
            x = x.unsqueeze(0)  # (1, indel_num, feature_num)
                
            # 处理序列特征
            if seq_features is not None:
                seq_data = seq_features.loc[sample]
                xs = _to_tensor(seq_data).unsqueeze(0)  # (1, seq_len, 4)
            else:
                xs = torch.zeros(1, 79, 4).to(DEVICE)  # 默认序列特征
                
            # 模型预测
            y_pred = model(x, xs)  # (1, indel_num)
            #y_pred = torch.softmax((y_pred) , dim=1)
            y_pred = y_pred.squeeze(0)  # (indel_num,)

            ''' head_logits, tail_logits, mask_logits, embout = model(x, xs, None,
                                                                  mask=None)  # .squeeze(-1)  # (batch, num_indels)
            mask_hard = (torch.sigmoid(mask_logits) > 0.3)
            fusion, weight_head = model.ensemble(embout.detach(), head_logits.detach(), tail_logits.detach(),
                                                 mask_hard.detach(), A)'''

            #head_prob = torch.softmax((head_logits), dim=1)
            #y_pred=head_prob.squeeze(0)
            #print(tail_prob)
            
            # 检查预测结果
            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                print(f"⚠️  样本 {sample} 预测结果包含NaN或Inf")
                continue
                
            # 归一化
            y_sum = y.sum()
            y_pred_sum = y_pred.sum()
            
            if y_sum <= 1e-10:
                y_sum+=1e-10
                #print(f"⚠️  样本 {sample} 标签和为零")
                #continue
                
            if y_pred_sum <= 1e-10:
                y_pred_sum+=1e-10
                #print(f"⚠️  样本 {sample} 预测和为零")
                #continue
                
            y = y / y_sum
            y_pred = y_pred / y_pred_sum
            
            # 转换为numpy
            y_np = y.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            
            # 最终检查
            if np.isnan(y_np).any() or np.isnan(y_pred_np).any():
                print(f"⚠️  样本 {sample} 归一化后包含NaN")
                continue
            
            # 计算回归指标    
            if len(y_np) > 1 and np.var(y_np) > 0:
                corr = pearsonr(y_pred_np, y_np)[0]
                if not np.isnan(corr):
                    correlations.append(corr)
                
                y_pred_clamped = torch.clamp(y_pred, min=1e-8, max=1-1e-8)
                kl_value = kl_div(torch.log(y_pred_clamped), y, reduction='batchmean').cpu().item()
                mse_value=mse_loss(y_pred_clamped, y, reduction='mean').cpu().item()
                if not np.isnan(kl_value) and not np.isinf(kl_value) and kl_value >= 0:
                    kl_divergences.append(kl_value)
                mse.append(mse_value)
                
            # 保存用于分类指标计算
            y_true_list.append(y_np)
            y_pred_list.append(y_pred_np)
                
    
    # 计算回归指标
    regression_metrics = {
        'avg_correlation': np.mean(correlations) if correlations else 0.0,
        'avg_kl_divergence': np.mean(kl_divergences) if kl_divergences else float('inf'),
        'avg_mse': np.mean(mse) if mse else float('inf'),
        'num_samples': len(correlations)
    }
    #print("当前全局温度：",model.current_temperature())
    # 计算分类指标
    classification_metrics = calculate_classification_metrics(y_true_list, y_pred_list, THRESHOLDS)
    
    return regression_metrics, classification_metrics

def print_results(regression_metrics, classification_metrics):
    """
    打印测试结果
    
    Args:
        regression_metrics: 回归指标
        classification_metrics: 分类指标
    """
    print(f"\n" + "="*80)
    print(f"📊 模型测试结果")
    print(f"="*80)
    
    # 回归指标
    print(f"\n🔍 回归指标:")
    print(f"   - 平均Pearson相关系数: {regression_metrics['avg_correlation']:.4f}")
    print(f"   - 平均KL散度: {regression_metrics['avg_kl_divergence']:.6f}")
    print(f"   - 平均mse损失: {regression_metrics['avg_mse']:.6f}")
    print(f"   - 有效样本数: {regression_metrics['num_samples']}")
    
    # 分类指标
    print(f"\n🎯 分类指标 (各阈值详细结果):")
    print(f"-"*80)
    for i, threshold in enumerate(classification_metrics['thresholds']):
        tp = classification_metrics['tp'][i]
        fp = classification_metrics['fp'][i]
        tn = classification_metrics['tn'][i]
        fn = classification_metrics['fn'][i]
        total = tp + fp + tn + fn
        
        print(f"阈值 {threshold:.1f}: "
              f"Precision={classification_metrics['precision'][i]:.4f}, "
              f"Recall={classification_metrics['recall'][i]:.4f}, "
              f"MCC={classification_metrics['mcc'][i]:.4f}, "
              f"F1={classification_metrics['f1_score'][i]:.4f}")
        print(f"         TP={tp}, FP={fp}, TN={tn}, FN={fn} | 总样本={total}")
        print()
    
    # 总体统计
    total_tp = sum(classification_metrics['tp'])
    total_fp = sum(classification_metrics['fp'])
    total_tn = sum(classification_metrics['tn'])
    total_fn = sum(classification_metrics['fn'])
    
    print(f"📈 总体分类统计:")
    print(f"   - 总TP: {total_tp}")
    print(f"   - 总FP: {total_fp}")
    print(f"   - 总TN: {total_tn}")
    print(f"   - 总FN: {total_fn}")
    
    # 平均指标
    avg_precision = np.mean(classification_metrics['precision'])
    avg_recall = np.mean(classification_metrics['recall'])
    avg_mcc = np.mean(classification_metrics['mcc'])
    avg_f1 = np.mean(classification_metrics['f1_score'])
    
    print(f"\n📊 平均分类指标:")
    print(f"   - 平均Precision: {avg_precision:.4f}")
    print(f"   - 平均Recall: {avg_recall:.4f}")
    print(f"   - 平均MCC: {avg_mcc:.4f}")
    print(f"   - 平均F1-Score: {avg_f1:.4f}")
    
    print(f"\n" + "="*80)


def calculate_classification_metrics(y_true_list, y_pred_list, thresholds):
    """
    计算分类指标
    
    Args:
        y_true_list: 真实标签列表
        y_pred_list: 预测概率列表
        thresholds: 阈值列表
    
    Returns:
        metrics_dict: 包含各阈值下指标的字典
    """
    results = {
        'thresholds': thresholds,
        'precision': [], 'recall': [], 'mcc': [], 'f1_score': [],
        'tp': [], 'fp': [], 'tn': [], 'fn': []
    }
    
    print(f"\n📊 计算分类指标 (样本数: {len(y_true_list)})")
    
    for threshold in thresholds:
        threshold_preds = []
        threshold_labels = []
        
        for y_true, y_pred in zip(y_true_list, y_pred_list):
            true_label = 1 if np.sum(y_true > threshold) == 1 else 0
            
            # 预测标签：是否预测有删除发生
            pred_label = 1 if np.sum(y_pred > threshold) == 1 else 0
            
            threshold_labels.append(true_label)
            threshold_preds.append(pred_label)
        
        # 计算混淆矩阵
        if len(set(threshold_labels)) > 1:  # 确保有两个类别
            tn, fp, fn, tp = confusion_matrix(threshold_labels, threshold_preds).ravel()
            
            # 计算指标
            prec = precision_score(threshold_labels, threshold_preds, zero_division=0)
            rec = recall_score(threshold_labels, threshold_preds, zero_division=0)
            mcc = matthews_corrcoef(threshold_labels, threshold_preds)
            f1 = f1_score(threshold_labels, threshold_preds, zero_division=0)
            
            results['precision'].append(prec)
            results['recall'].append(rec)
            results['mcc'].append(mcc)
            results['f1_score'].append(f1)
            results['tp'].append(int(tp))
            results['fp'].append(int(fp))
            results['tn'].append(int(tn))
            results['fn'].append(int(fn))
        else:
            # 只有一个类别的情况
            results['precision'].append(0.0)
            results['recall'].append(0.0)
            results['mcc'].append(0.0)
            results['f1_score'].append(0.0)
            results['tp'].append(0)
            results['fp'].append(0)
            results['tn'].append(len(threshold_labels))
            results['fn'].append(0)
    
    return results
def compute_prior_stable(
    y_full: pd.Series,
    indel_list=None,
    alpha: float = 1.0,         # 拉普拉斯平滑
    lam: float = 0.05,          # 与均匀分布插值系数
    pi_min: float = 1e-10,       # 地板，避免过小
    pi_max: float = 0.4         # 天花板，避免过大（可适当放宽）
):
    # 1) 只统计真实存在的类（求每个 indel 的全局计数/质量）
    y = y_full.clip(lower=0)
    c = y.groupby(level=1).sum()   # Series: indel -> mass

    # 对齐顺序
    if indel_list is not None:
        c = c.reindex(indel_list, fill_value=0.0)

    # 2) 拉普拉斯平滑
    c = c.to_numpy(dtype=np.float64)
    c_smooth = c + alpha

    # 3) 归一化 + 与均匀分布插值
    pi = c_smooth / c_smooth.sum()
    K = len(pi)
    pi = (1 - lam) * pi + lam * (1.0 / K)

    # 4) 裁剪避免极端
    pi = np.clip(pi, pi_min, pi_max)
    pi = pi / pi.sum()

    return pi.astype(np.float32)
def make_logit_adjustment(pi: np.ndarray, tau: float = 1.0, device: str = "cpu"):
    # A = -tau * log(pi)
    A = tau * np.log(np.clip(pi, 1e-12, 1.0)).astype(np.float32)
    return torch.tensor(A, dtype=torch.float32, device=device)
def weighted_kl_loss(
    pred: torch.Tensor,   # (B, C)
    p_true: torch.Tensor,   # (B, C) 非负，可未归一化
    gamma: float = 0.5,     # 头部放大强度: 0.5~2 常用
    mask: torch.Tensor = None,  # (B, C) 可选无效类掩码
    eps: float = 1e-8
) -> torch.Tensor:
    """
    仅头部放大 p^gamma 的加权 KL（无阈值窗/焦点/间隔项）。
    返回标量 loss。
    """
    #q = F.softmax(logits, dim=-1)
    p = p_true.clamp_min(0.0)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)

    # 头部放大权重 w = p^gamma
    w = (p + eps) ** gamma

    # 加权 forward-KL
    kl_elem = w * p * (torch.log(p + eps) - torch.log(pred + eps))

    if mask is not None:
        m = mask.to(dtype=kl_elem.dtype)
        kl_elem = kl_elem * m
        valid = m.sum(dim=-1).clamp_min(1.0)
        loss = kl_elem.sum(dim=-1) / valid
        return loss.mean()

    return kl_elem.mean()


def continuous_weighted_kl_loss(pred, p_true, gamma_func="linear",eps: float = 1e-8):
    """
    连续变化的gamma值
    """
    p = p_true.clamp_min(0.0)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)

    if gamma_func == "linear":
        # gamma从-0.1线性变化到-0.5
        gamma = -0.25 - 0.1 * p  # p越大，gamma越负
    elif gamma_func == "sigmoid":
        # S形变化
        gamma = -0.3 - 0.4 * torch.sigmoid(10 * (p - 0.05))
    elif gamma_func == "piecewise_linear":
        # 分段线性
        gamma = torch.where(
            p < 0.01,
            -0.1,
            torch.where(
                p < 0.05,
                -0.1 - 2.0 * (p - 0.01),  # 快速变化
                -0.3 - 0.5 * (p - 0.05)  # 慢速变化
            )
        )

    weights = (p + eps) ** gamma
    kl_elem = weights * p * (torch.log(p + eps) - torch.log(pred + eps))
    return kl_elem.mean()


def adaptive_gamma_weighted_kl(pred, p_true, base_gamma=-0.3, eps=1e-8):
    """
    根据当前模型表现动态调整gamma
    """
    # 计算模型在当前数据上的平均置信度 - 需要detach！
    # 计算平均置信度 - pred已经是概率
    with torch.no_grad():
        avg_confidence = pred.max(dim=-1)[0].mean()

    # 如果模型整体置信度高，更关注尾部；置信度低，更关注头部
    dynamic_gamma = base_gamma + 0.3 * (avg_confidence - 0.5)

    w = (p_true + eps) ** dynamic_gamma
    kl_elem = w * p_true * (torch.log(p_true + eps) - torch.log(pred + eps))
    return kl_elem.mean()


def curriculum_weighted_kl(pred, p_true, epoch, total_epochs=200, base_gamma=0,eps=1e-8):
    """
    随着训练进行，逐步增加对尾部的关注
    """
    # 线性课程：从关注头部逐步转移到关注尾部
    progress = min(epoch / total_epochs,1)
    curriculum_gamma = base_gamma - 0.4 * progress  # 从-0.3逐步到-0.7

    w = (p_true + eps) ** curriculum_gamma
    kl_elem = w * p_true * (torch.log(p_true + eps) - torch.log(pred + eps))
    return kl_elem.mean()