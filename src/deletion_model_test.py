"""
模型测试脚本 - 加载训练好的模型并评估
"""
from deletion_model_train import (
    StableLongTailDualModel, ModelConfig, TrainingConfig,
    get_output_path
)
from common_def import *
import argparse


def load_trained_model(model_path, num_features, seq_feature_dim):
    """加载训练好的模型"""
    print(f"📂 加载模型: {model_path}")
    

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    config = checkpoint.get('config', {})
    
    # 初始化模型
    model = StableLongTailDualModel(
        num_features=num_features,
        seq_feature_dim=seq_feature_dim,
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
    
    print(f"✅ 模型加载成功")
    print(f"   训练配置: {config.get('loss_type', 'N/A')}")
    print(f"   最佳Epoch: {config.get('best_epoch', 'N/A')}")
    
    return model, config


def main(model_path=None, test_dataset='both'):
    """
    主测试函数
    
    Args:
        model_path: 模型路径，默认使用最佳loss模型
        test_dataset: 'test1', 'test2', 'both'
    """
    print("="*80)
    print("🧪 DU-AxisCRISP 模型测试")
    print("="*80)
    
    # 初始化配置
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    # 设置默认模型路径
    if model_path is None:
        output_path = get_output_path(model_config, train_config)
        model_path = output_path.replace(".pth", "_best_loss.pth")
    
    print(f"\n📊 测试配置:")
    print(f"   设备: {DEVICE}")
    print(f"   模型: {model_path}")
    print(f"   测试集: {test_dataset}")
    print(f"   特征集: {train_config.FEATURES}")
    
    # 加载数据
    print(f"\n📂 加载数据...")
    with open(indels_sorted_path, "rb") as f:
        ALL_INDELS = pkl.load(f)
    
    X, y, samples, sequences = load_delete_data(
        filepath=train_file_path, num_samples=1, fractions=True, indel_list=ALL_INDELS
    )
    X = X.loc[:, FEATURE_SETS[train_config.FEATURES]]
    
    # 加载模型
    model, config = load_trained_model(model_path, X.shape[1], sequences.shape[1])
    
    # 计算先验
    prior = compute_prior_stable(y)
    A = make_logit_adjustment(prior, tau=1, device=DEVICE)
    
    # 加载测试集
    print(f"\n📊 加载测试集...")
    if test_dataset == 'test1':
        test_X, test_y, test_samples, test_sequences = load_delete_data(
            filepath=test_file_path, num_samples=None, fractions=True, indel_list=ALL_INDELS
        )
        test_X = test_X.loc[:, FEATURE_SETS[train_config.FEATURES]]
        print(f"   Test1 样本数: {len(test_samples)}")
    elif test_dataset == 'test2':
        test_X, test_y, test_samples, test_sequences = load_delete_data_inDelphi(
            filepath=t2_path, num_samples=None, fractions=True, indel_list=ALL_INDELS
        )
        test_X = test_X.loc[:, FEATURE_SETS[train_config.FEATURES]]
        print(f"   Test2 样本数: {len(test_samples)}")
    else:  # both
        test_X, test_y, test_samples, test_sequences = merge_tests(
            t1_path, t2_path, indel_list=ALL_INDELS, mode="union"
        )
        test_X = test_X.loc[:, FEATURE_SETS[train_config.FEATURES]]
        print(f"   合并测试集样本数: {len(test_samples)}")
    
    # 评估模型
    print(f"\n{'='*80}")
    print("🎯 开始评估")
    print(f"{'='*80}\n")
    
    test_reg, test_cls = test_model(
        model, test_X, test_y, test_samples, test_sequences, ALL_INDELS, A
    )
    
    # 打印结果
    print(f"\n{'='*80}")
    print("📈 评估结果")
    print(f"{'='*80}")
    print_results(test_reg, test_cls)
    
    # 打印关键指标
    print(f"\n{'='*80}")
    print("🏆 关键指标总结")
    print(f"{'='*80}")
    print(f"   Pearson相关系数: {test_reg['avg_correlation']:.4f}")
    print(f"   KL散度: {test_reg['avg_kl_divergence']:.6f}")
    print(f"   平均MSE: {test_reg['avg_mse']:.6f}")
    
    if test_cls and 'thresholds' in test_cls:
        threshold_idx = test_cls['thresholds'].index(0.3) if 0.3 in test_cls['thresholds'] else 0
        print(f"\n   @ 阈值 0.3:")
        print(f"     MCC: {test_cls['mcc'][threshold_idx]:.4f}")
        print(f"     Precision: {test_cls['precision'][threshold_idx]:.4f}")
        print(f"     Recall: {test_cls['recall'][threshold_idx]:.4f}")
        print(f"     F1-Score: {test_cls['f1_score'][threshold_idx]:.4f}")
    
    print(f"\n{'='*80}")
    print("✅ 测试完成!")
    print(f"{'='*80}\n")
    
    return test_reg, test_cls


#python deletion_test.py --model output/dual0_freq_v3_stable_WKL0.25_T0.8_KL_Div_freq0.3_v2_testall_best_loss.pth --dataset test1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DU-AxisCRISP 模型测试')
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径 (默认: best_loss模型)')
    parser.add_argument('--dataset', type=str, default='both',
                        choices=['test1', 'test2', 'both'],
                        help='测试数据集: test1, test2, 或 both')
    
    args = parser.parse_args()
    main(model_path=args.model, test_dataset=args.dataset)

