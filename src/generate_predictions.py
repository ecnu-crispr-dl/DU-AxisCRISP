"""
DU-AxisCRISP 预测生成脚本
整合删除模型和插入模型，生成完整的indel预测结果
"""

from common_def import *
from deletion_model_train import StableLongTailDualModel
from models.tcn import AxisTCN
from ins_model_train import INSERTION_INDELS
from tqdm import tqdm
import argparse


# =============================================================================
# 配置
# =============================================================================
class Config:
    """全局配置类"""
    PREDICTIONS_DIR = "./predictions/"
    MIN_NUM_READS = 100  # 样本过滤阈值
    
    # 默认模型路径
    DEFAULT_DEL_MODEL = "output/dual0_freq_v3_stable_WKL0.25_T0.8_KL_Div_kl_freq0.3_v2_testall_best_loss.pth"
    DEFAULT_INS_MODEL = "output/insertion_axisTCN_Sequence-only_wkl-0.1_v2_best_loss.pth"
    DEFAULT_LINDEL_MODEL = "output/100x_indel.h5"
    
    # 数据集配置映射: (名称, oligos文件路径, genotype)
    DATASET_MAPPING = {
        "test": ("test", "evaluate/predict_results/FORECast/test.fasta", "test"),
        "test2": ("LibA", "evaluate/predict_results/inDelphi/LibA.fasta", "0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1")
    }


# =============================================================================
# 模型加载
# =============================================================================
def load_deletion_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    config = checkpoint.get('config', {})
    
    model = StableLongTailDualModel(
        num_features=config.get('num_features', 6),
        seq_feature_dim=config.get('seq_feature_dim', 705),
        hidden_dim=config.get('hidden_dim', 128),
        out_dim=config.get('out_dim', 96),
        temperature=config.get('temperature', 0.8),
        freq_threshold=config.get('freq_threshold', 0.3),
        tail_alpha=0.8,
        ema_decay=config.get('ema_decay', 0.95),
        gate_smoothing=config.get('gate_smoothing', 0.7)
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model


def load_insertion_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = AxisTCN().to(DEVICE)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model


def load_lindel_model(model_path):
    from tensorflow import keras
    return keras.models.load_model(model_path)


# =============================================================================
# DNA序列编码
# =============================================================================
def onehotencoder(seq):
    """单核苷酸 + 双核苷酸编码，20bp guide序列 -> 384维"""
    nt = ['A', 'T', 'C', 'G']
    head = []
    l = len(seq)
    
    for k in range(l):
        for i in range(4):
            head.append(nt[i] + str(k))
    
    for k in range(l - 1):
        for i in range(4):
            for j in range(4):
                head.append(nt[i] + nt[j] + str(k))
    
    head_idx = {key: idx for idx, key in enumerate(head)}
    encode = np.zeros(len(head_idx))
    
    for j in range(l):
        encode[head_idx[seq[j] + str(j)]] = 1.
    
    for k in range(l - 1):
        encode[head_idx[seq[k:k+2] + str(k)]] = 1.
    
    return encode


def read_oligos(oligo_file_path):
    """
    读取oligo信息（支持FASTA、PKL和JSON格式）
    用于提取guide序列和PAM位置信息
    """
    if oligo_file_path is None:
        return None
    
    # 处理相对路径
    if not os.path.isabs(oligo_file_path):
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        oligo_file_path = os.path.join(project_root, oligo_file_path)
    
    if not os.path.exists(oligo_file_path):
        print(f"⚠️ 文件不存在: {oligo_file_path}")
        return None
    
    try:
        if oligo_file_path.endswith(('.pkl', '.pickle')):
            with open(oligo_file_path, 'rb') as f:
                oligos = pkl.load(f)
        elif oligo_file_path.endswith(('.fasta', '.fa')):
            # FASTA格式: >ID PAM_Index ORIENTATION
            #           Sequence
            oligos = []
            with open(oligo_file_path, 'r') as f:
                current_id = None
                current_pam_index = None
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        parts = line[1:].split()
                        if len(parts) >= 2:
                            current_id = parts[0]
                            current_pam_index = int(parts[1])
                    elif line and current_id:
                        oligos.append({
                            "ID": current_id,
                            "TargetSequence": line,
                            "PAM Index": current_pam_index
                        })
                        current_id = None
            
            if len(oligos) == 0:
                return None
        else:
            # JSON格式
            import json
            oligos = []
            with open(oligo_file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        oligos.append(json.loads(line.strip()))
        
        return oligos
    except Exception as e:
        print(f"⚠️ 读取oligos失败: {e}")
        return None


# =============================================================================
# 预测生成核心逻辑
# =============================================================================
def _batch_predict_ratios(indel_model, oligos, all_samples):
    """
    批量预测del/ins比例（性能优化：批量预测比逐个预测快5-10倍）
    使用Lindel模型预测deletion/insertion比例
    """
    dratio_dict = {}
    insratio_dict = {}
    valid_guides = []
    valid_sample_ids = []
    pam_dict = {'AGG', 'TGG', 'CGG', 'GGG'}
    
    for o in oligos:
        sample_id = o["ID"]
        if sample_id not in all_samples:
            continue
        
        target_seq = o["TargetSequence"]
        pam_index = o["PAM Index"]
        
        # 根据序列长度选择提取方式（与test_3.py保持一致）
        if len(target_seq) >= 60:
            # 长序列（如test.fasta，80+bp）
            seq = target_seq[pam_index-33:pam_index + 27]
            guide = seq[13:33]
            pam_seq = seq[33:36]
        else:
            # 短序列（如LibA.fasta，55bp）
            guide = target_seq[pam_index-20:pam_index]
            pam_seq = target_seq[pam_index:pam_index+3]
        
        # 验证guide和PAM有效性
        if len(guide) == 20 and len(pam_seq) == 3 and pam_seq in pam_dict:
            valid_guides.append(guide)
            valid_sample_ids.append(sample_id)
        else:
            # 无效样本使用固定比例
            dratio_dict[sample_id] = 0.7
            insratio_dict[sample_id] = 0.3
    
    # 批量预测所有有效样本
    if len(valid_guides) > 0:
        try:
            encoded_guides = np.array([onehotencoder(g) for g in valid_guides])
            batch_predictions = indel_model.predict(encoded_guides, verbose=0, batch_size=128)
            
            for i, sample_id in enumerate(valid_sample_ids):
                dratio_dict[sample_id] = batch_predictions[i, 0]
                insratio_dict[sample_id] = batch_predictions[i, 1]
        except Exception as e:
            print(f"   ⚠️ 批量预测失败: {e}")
            dratio_dict.clear()
            insratio_dict.clear()
    
    return dratio_dict, insratio_dict


def _preextract_features(X_del, seq_full, samples):
    """
    预提取特征数据，避免循环内频繁索引（性能优化）
    将DataFrame数据提前转换为dict，减少重复的.loc索引操作
    """
    X_del_dict = {}
    seq_dict = {}
    homology_dict = {}
    
    for sample_id in samples:
        # 提取v2特征集（deletion模型输入）
        X_del_dict[sample_id] = X_del.loc[sample_id, FEATURE_SETS["v2"]].to_numpy()
        # 提取序列特征（insertion模型输入）
        if seq_full is not None:
            seq_dict[sample_id] = seq_full.loc[sample_id].to_numpy()
        # 提取microhomology长度
        homology_dict[sample_id] = X_del.loc[sample_id, "homologyLength"]
    
    return X_del_dict, seq_dict, homology_dict


def generate_predictions(deletion_model, insertion_model, indel_model,
                        test_dataset, ALL_INDELS, output_path, oligos=None):
    """
    生成预测结果
    整合deletion和insertion模型的输出，生成完整的indel预测
    """
    print(f"\n{'='*80}")
    print(f"🔮 生成预测: {test_dataset}")
    print(f"{'='*80}\n")
    
    # 加载数据（与test_3.py保持一致的两步加载策略）
    data_path = test_file_path if test_dataset == 'test' else t2_path
    with open(data_path, 'rb') as f:
        data = pkl.load(f)
    y_original = data["counts"]  # 原始counts，用于获取实际存在的indel
    
    # 加载填充后的特征数据（用于模型预测）
    X_del, y, _, seq_full = load_delete_data(
        filepath=data_path, num_samples=None, fractions=True, indel_list=ALL_INDELS
    )
    
    # 样本过滤：保持原始样本顺序 + 过滤低读数样本
    all_samples = y_original.index.get_level_values(0).unique()
    sample_reads = y_original.groupby(level=0)["countEvents"].sum()
    common_samples = sample_reads[sample_reads >= Config.MIN_NUM_READS].index
    samples = [s for s in all_samples if s in common_samples]
    
    print(f"样本数: {len(samples)} (过滤: min_reads >= {Config.MIN_NUM_READS})")
    print(f"删除特征数: {X_del.shape[1]}")
    print(f"序列特征维度: {seq_full.shape[1] if seq_full is not None else 0}")
    
    profiles = {}
    use_oligos = oligos is not None
    
    if use_oligos:
        print(f"✅ 使用oligos提取guide序列")
        oligos_filtered = [o for o in oligos if o["ID"] in common_samples]
        iteration_list = oligos_filtered
    else:
        print(f"⚠️ 未提供oligos，使用固定比例 (0.7/0.3)")
        iteration_list = samples
    
    # 性能优化：批量预测del/ins比例
    if use_oligos and indel_model is not None:
        dratio_dict, insratio_dict = _batch_predict_ratios(indel_model, iteration_list, all_samples)
        print(f"   预计算完成，{len(dratio_dict)} 个样本")
    
    # 性能优化：预提取特征数据
    print(f"\n⚡ 预提取特征数据...")
    X_del_dict, seq_dict, homology_dict = _preextract_features(X_del, seq_full, samples)
    print(f"   完成，{len(X_del_dict)} 个样本\n")
    
    pbar = tqdm(iteration_list, desc="预测进度", ncols=100, position=0, leave=True)
    
    for item in pbar:
        # 确定sample_id
        if use_oligos:
            o = item
            sample_id = o["ID"]
            if sample_id not in all_samples:
                continue
        else:
            sample_id = item
        
        pbar.set_postfix_str(f"样本: {sample_id[:30]}...")
        
        # 模型预测
        with torch.no_grad():
            x_del = torch.tensor(X_del_dict[sample_id]).float().unsqueeze(0)
            
            if sample_id in seq_dict:
                seq = torch.tensor(seq_dict[sample_id]).float().unsqueeze(0)
            else:
                seq = torch.zeros(1, 705).float()
                tqdm.write(f"⚠️ 样本 {sample_id} 缺少序列特征")
            
            # Deletion预测
            ds = deletion_model(x_del, None).squeeze(0)
            ds = (ds / ds.sum()).detach().cpu().numpy()
            
            # Insertion预测
            ins = insertion_model(None, seq).squeeze(0).detach().cpu().numpy()
        
        # 获取实际存在的deletion类型（关键：与test_3.py保持一致）
        y_obs_original = y_original.loc[sample_id]
        valid_indels = list(y_obs_original.index.intersection(ALL_INDELS))
        
        # 获取del/ins比例（优先使用预计算的比例）
        if sample_id in dratio_dict:
            dratio, insratio = dratio_dict[sample_id], insratio_dict[sample_id]
        else:
            dratio, insratio = 0.7, 0.3
        
        # 过滤deletion预测：只保留实际存在的deletion类型
        indel_index_map = {indel: i for i, indel in enumerate(ALL_INDELS)}
        ds_filtered = np.array([ds[indel_index_map[i]] for i in valid_indels])
        ds_filtered = ds_filtered / ds_filtered.sum() if ds_filtered.sum() > 0 else ds_filtered
        
        # Insertion预测：保留所有21种类型（与其他模型保持一致）
        ins_normalized = ins / ins.sum() if ins.sum() > 0 else ins
        
        # 合并预测：deletion + insertion
        y_hat = np.concatenate((ds_filtered * dratio, ins_normalized * insratio))
        
        # 构建完整的indel列表和真实标签
        all_indels = valid_indels + list(INSERTION_INDELS)
        y_obs_selected = y_obs_original.loc[all_indels]
        y_obs_normalized = y_obs_selected["countEvents"].values / y_obs_selected["countEvents"].sum()
        
        # 计算microhomology标记
        hl = homology_dict[sample_id]
        mh = [bool(hl.get(ind, 0) > 0) for ind in all_indels]
        
        # 保存结果
        profiles[sample_id] = {
            "predicted": y_hat,
            "actual": y_obs_normalized,
            "indels": all_indels,
            "mh": mh
        }
    
    print(f"\n✅ 预测完成，共 {len(profiles)} 个样本")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pkl.dump(profiles, f)
    
    print(f"💾 已保存到: {output_path}")
    return profiles


# =============================================================================
# 主程序
# =============================================================================
def main(deletion_model_path=None, insertion_model_path=None, lindel_model_path=None,
         test_dataset='test', loss_fn='KL_Div', lr='0.01'):
    """主函数：生成预测结果"""
    
    deletion_model_path = deletion_model_path or Config.DEFAULT_DEL_MODEL
    insertion_model_path = insertion_model_path or Config.DEFAULT_INS_MODEL
    lindel_model_path = lindel_model_path or Config.DEFAULT_LINDEL_MODEL
    
    if test_dataset not in Config.DATASET_MAPPING:
        print(f"❌ 错误：未知的测试数据集 '{test_dataset}'")
        print(f"   可用的数据集: {list(Config.DATASET_MAPPING.keys())}")
        return None
    
    dataset_name, oligos_file, genotype = Config.DATASET_MAPPING[test_dataset]
    
    print("=" * 80)
    print("🔮 DU-AxisCRISP 预测生成")
    print("=" * 80)
    print(f"设备: {DEVICE}")
    print(f"删除模型: {deletion_model_path}")
    print(f"插入模型: {insertion_model_path}")
    print(f"Lindel模型: {lindel_model_path}")
    print(f"数据集: {dataset_name} (genotype: {genotype})")
    print("=" * 80 + "\n")
    
    print("🤖 加载模型...")
    deletion_model = load_deletion_model(deletion_model_path)
    insertion_model = load_insertion_model(insertion_model_path)
    
    indel_model = None
    if lindel_model_path and os.path.exists(lindel_model_path):
        indel_model = load_lindel_model(lindel_model_path)
    else:
        print("⚠️ 未提供Lindel模型，使用固定比例\n")
    
    with open(indels_sorted_path, "rb") as f:
        ALL_INDELS = pkl.load(f)
    
    print(f"📂 加载oligos: {oligos_file}")
    oligos = read_oligos(oligos_file)
    if not oligos:
        print(f"❌ 无法加载oligos文件")
        return None
    print(f"✅ 成功加载 {len(oligos)} 个oligos")
    
    output_path = os.path.join(
        Config.PREDICTIONS_DIR,
        f"XCRISP_testmask_deldualmodelWKL0.25_insTCN_sequenceonly_{loss_fn}_{lr}__{genotype}.pkl"
    )
    
    profiles = generate_predictions(
        deletion_model, insertion_model, indel_model,
        test_dataset, ALL_INDELS, output_path, oligos=oligos
    )
    
    print("\n" + "=" * 80)
    print("🎉 预测生成完成!")
    print("=" * 80)
    
    return profiles


# =============================================================================
# 命令行入口
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DU-AxisCRISP 预测生成')
    parser.add_argument('--deletion_model', type=str, default=None,
                        help=f'删除模型路径（默认: {Config.DEFAULT_DEL_MODEL}）')
    parser.add_argument('--insertion_model', type=str, default=None,
                        help=f'插入模型路径（默认: {Config.DEFAULT_INS_MODEL}）')
    parser.add_argument('--lindel_model', type=str, default=None,
                        help=f'Lindel模型路径（默认: {Config.DEFAULT_LINDEL_MODEL}）')
    parser.add_argument('--dataset', type=str, default='test',
                        choices=['test', 'test2'],
                        help='测试数据集（默认: test）')
    parser.add_argument('--loss_fn', type=str, default='KL_Div',
                        help='损失函数名称（默认: KL_Div）')
    parser.add_argument('--lr', type=str, default='0.01',
                        help='学习率（默认: 0.01）')
    
    args = parser.parse_args()
    
    main(
        deletion_model_path=args.deletion_model,
        insertion_model_path=args.insertion_model,
        lindel_model_path=args.lindel_model,
        test_dataset=args.dataset,
        loss_fn=args.loss_fn,
        lr=args.lr
    )
