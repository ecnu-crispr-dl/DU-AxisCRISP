"""
DU-AxisCRISP 插入模块训练脚本
基于AxisTCN模型的CRISPR插入预测
"""

# ============================================================================
# 导入依赖
# ============================================================================
from common_def import *
from common_def import _to_tensor as _to_tensor_explicit
from models.tcn import AxisTCN
import torch.nn.functional as F


# ============================================================================
# 全局配置
# ============================================================================
class TrainingConfig:
    """训练配置"""
    # 数据配置
    NUM_SAMPLE = None           # 使用全部数据
    RANDOM_STATE = 42
    FEATURES = "insv1"
    
    # 训练超参数
    EPOCHS = 41
    BATCH_SIZE = 64
    LEARING_RATE = 0.0002
    
    # 训练策略
    USE_WARMUP = True
    WARMUP_EPOCHS = 15
    EVAL_FREQUENCY = 10


class LossConfig:
    """损失函数配置"""
    LOSS_TYPE = "weighted_kl"
    ALPHA = -0.1  # weighted KL loss的gamma参数


# 插入indel列表（21类）
INSERTION_INDELS = [
    '1+A', '1+T', '1+C', '1+G',
    '2+AA', '2+AT', '2+AC', '2+AG', '2+TA', '2+TT', '2+TC', '2+TG',
    '2+CA', '2+CT', '2+CC', '2+CG', '2+GA', '2+GT', '2+GC', '2+GG',
    '3+X'
]


# ============================================================================
# 路径和输出
# ============================================================================
def get_output_path(config: LossConfig) -> str:
    """生成输出模型路径"""
    return output_dir + f"insertion_axisTCN_Sequence-only_wkl{config.ALPHA}_v2.pth"


# ============================================================================
# 初始化函数
# ============================================================================
def init_env(config: TrainingConfig):
    """初始化训练环境"""
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    os.makedirs(output_dir, exist_ok=True)


def init_model(train_config: TrainingConfig):
    """初始化模型、优化器和学习率调度器"""
    model = AxisTCN().to(DEVICE)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        train_config.LEARING_RATE, 
        betas=(0.9, 0.999)
    )
    
    if train_config.USE_WARMUP:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    
    return model, optimizer, scheduler


# ============================================================================
# 辅助函数
# ============================================================================
def build_model_config(train_config: TrainingConfig, loss_config: LossConfig,
                       X_shape: tuple, seq_shape: tuple, 
                       best_pearson: float = None, best_epoch: int = None) -> dict:
    """构建模型保存配置"""
    config = {
        "model_type": "insertion_axisTCN",
        "loss_type": loss_config.LOSS_TYPE,
        "alpha": loss_config.ALPHA,
        "num_features": X_shape[1],
        "seq_feature_dim": seq_shape[1],
    }
    
    if best_pearson is not None:
        config["best_pearson"] = best_pearson
    if best_epoch is not None:
        config["best_epoch"] = best_epoch
    
    return config


def save_model(model, optimizer, lr_scheduler, samples, X_shape, seq_shape,
               train_config: TrainingConfig, loss_config: LossConfig,
               filepath: str, best_pearson: float = None, best_epoch: int = None):
    """保存模型"""
    torch.save({
        "random_state": train_config.RANDOM_STATE,
        "model": model.state_dict(),
        "samples": samples,
        "loss_type": loss_config.LOSS_TYPE,
        "optimiser": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "feature_sets": FEATURE_SETS[train_config.FEATURES],
        "config": build_model_config(
            train_config, loss_config, X_shape, seq_shape,
            best_pearson, best_epoch
        )
    }, filepath)


# ============================================================================
# 训练函数
# ============================================================================
def train_model(X_arrays, Sequences_arrays, y_arrays, samples,
                model, optimizer, lr_scheduler,
                train_config: TrainingConfig, loss_config: LossConfig,
                output_path: str, A):
    """训练模型主函数"""
    
    # 打印训练配置
    print("\n" + "=" * 80)
    print("🔥 Insertion Model Training: AxisTCN Sequence-only")
    print("=" * 80)
    print(f"📊 训练配置:")
    print(f"   - 模型: AxisTCN (序列特征专用)")
    print(f"   - Loss: weighted_kl_loss (alpha={loss_config.ALPHA})")
    print(f"   - 样本数: {'全部' if train_config.NUM_SAMPLE is None else train_config.NUM_SAMPLE}")
    print(f"   - 学习率: {train_config.LEARING_RATE}")
    print(f"   - Batch大小: {train_config.BATCH_SIZE}")
    print("=" * 80 + "\n")

    train_samples, val_samples = train_test_split(samples, test_size=100, random_state=train_config.RANDOM_STATE)

    test_X, test_y, test_samples, test_sequences = load_insert_data(
        filepath=t1_path, num_samples=None, fractions=True, indel_list=INSERTION_INDELS
    )
    test_X = test_X.loc[:, FEATURE_SETS[train_config.FEATURES]]


    # 初始化跟踪变量
    best_loss = float('inf')
    best_pearson = 0.0
    patience_loss = 0
    patience_pearson = 0

    # 训练循环
    for epoch in range(train_config.EPOCHS):
        epoch_start = time.time()

        # Warmup学习率调整
        if train_config.USE_WARMUP and epoch < train_config.WARMUP_EPOCHS:
            warmup_lr = train_config.LEARING_RATE * (epoch + 1) / train_config.WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        # 打乱训练样本
        train_samples_shuffled = np.random.permutation(train_samples)

        # 批次训练
        epoch_losses = []
        for i in range(0, len(train_samples_shuffled), train_config.BATCH_SIZE):
            batch_samples = train_samples_shuffled[i:i + train_config.BATCH_SIZE]
            model.train()

            # 准备批次数据
            y_batch = torch.stack([_to_tensor_explicit(y_arrays.loc[s]) for s in batch_samples])
            seq_batch = torch.stack([_to_tensor_explicit(Sequences_arrays.loc[s]) for s in batch_samples])
            y_batch = y_batch / (y_batch.sum(dim=1, keepdim=True) + 1e-10)

            # 前向传播（AxisTCN只使用序列特征）
            y_pred_batch = model(None, seq_batch)

            # 计算损失
            loss_total = torch.zeros(1).to(DEVICE)
            for bi in range(len(batch_samples)):
                y_true = y_batch[bi]
                y_pred = y_pred_batch[bi]
                loss_wkl = weighted_kl_loss(y_pred, y_true, gamma=loss_config.ALPHA)
                loss_total += loss_wkl

            loss_total = loss_total / len(batch_samples)

            # 反向传播
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            epoch_losses.append(loss_total.detach().cpu().item())

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        # 评估和保存
        should_eval = (epoch % train_config.EVAL_FREQUENCY == 0 and epoch >= 30) or \
                      (epoch == train_config.EPOCHS - 1)
        
        current_pearson = 0.0
        if should_eval:
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch}/{train_config.EPOCHS}")
            print(f"{'=' * 80}")
            
            print("\n📊 测试集评估:")
            test_reg, test_cls = test_model(
                model, test_X, test_y, test_samples, test_sequences, INSERTION_INDELS, A
            )
            print_results(test_reg, test_cls)
            
            current_pearson = test_reg['avg_correlation']
            print(f"\n🎯 当前Pearson: {current_pearson:.4f}, 最佳Pearson: {best_pearson:.4f}")
            print("=" * 80)
            
            # 保存最佳Pearson模型
            if current_pearson > best_pearson:
                best_pearson = current_pearson
                patience_pearson = 0
                print(f"\n🎉 新的最佳Pearson模型! Pearson={best_pearson:.4f}")
                
                save_model(
                    model, optimizer, lr_scheduler, samples,
                    X_arrays.shape, Sequences_arrays.shape,
                    train_config, loss_config,
                    output_path.replace(".pth", "_best_pearson.pth"),
                    best_pearson, epoch
                )
            else:
                patience_pearson += 1
                print(f"Pearson未改善: {patience_pearson}/15")
        else:
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.6f}, LR={current_lr:.6f}, Time={epoch_time:.2f}s")
        
        # 保存最佳Loss模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_loss = 0
            print(f"\n💾 新的最佳Loss模型! loss={best_loss:.6f}")
            
            save_model(
                model, optimizer, lr_scheduler, samples,
                X_arrays.shape, Sequences_arrays.shape,
                train_config, loss_config,
                output_path.replace(".pth", "_best_loss.pth")
            )
        else:
            patience_loss += 1

        # 早停检查
        if patience_loss >= 30:
            print(f"\n🛑 Loss早停! 最佳Loss: {best_loss:.6f}")
            break

    print(f"\n✅ 训练完成!")
    print(f"🎯 最佳Pearson: {best_pearson:.4f}")
    return best_pearson


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    # 初始化配置
    train_config = TrainingConfig()
    loss_config = LossConfig()
    output_path = get_output_path(loss_config)
    
    # 打印配置信息
    print("=" * 80)
    print("DU-AxisCRISP 插入模块训练")
    print("=" * 80)
    print(f"配置:")
    print(f"   设备: {DEVICE}")
    print(f"   样本数: {'全部' if train_config.NUM_SAMPLE is None else train_config.NUM_SAMPLE}")
    print(f"   特征集: {train_config.FEATURES}")
    print(f"   学习率: {train_config.LEARING_RATE}")
    print(f"   Epoch数: {train_config.EPOCHS}")
    print(f"   Batch大小: {train_config.BATCH_SIZE}")
    print(f"   Loss Alpha: {loss_config.ALPHA}")
    print("=" * 80 + "\n")

    # 初始化环境
    init_env(train_config)

    # 加载数据
    print("📂 加载数据...")
    X, y, samples, sequences = load_insert_data(
        filepath=train_file_path,
        num_samples=train_config.NUM_SAMPLE,
        fractions=True,
        indel_list=INSERTION_INDELS
    )
    X = X.loc[:, FEATURE_SETS[train_config.FEATURES]]
    prior = compute_prior_stable(y)
    A = make_logit_adjustment(prior, tau=1, device=DEVICE)
    
    print(f"✅ 数据加载完成")
    print(f"   样本数: {len(samples)}")
    print(f"   特征数: {X.shape[1]}")
    print(f"   Indel数: {len(INSERTION_INDELS)}")
    print(f"   序列特征维度: {sequences.shape[1]}")

    # 初始化模型
    print("\n🤖 初始化模型...")
    model, optimizer, lr_scheduler = init_model(train_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 模型初始化完成")
    print(f"   总参数数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")

    # 训练模型
    best_pearson = train_model(
        X, sequences, y, samples,
        model, optimizer, lr_scheduler,
        train_config, loss_config, output_path, A
    )

    # 保存最终模型
    print("\n💾 保存最终模型...")
    save_model(
        model, optimizer, lr_scheduler, samples,
        X.shape, sequences.shape,
        train_config, loss_config,
        output_path
    )

    print(f"✅ 模型已保存到: {output_path}")
    print("\n" + "=" * 80)
    print("🎉 训练完成!")
    print("=" * 80)
