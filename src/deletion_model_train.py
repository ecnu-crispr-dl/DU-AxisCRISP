"""
DU-AxisCRISP 删除模块训练脚本
基于稳定长尾分布双分支模型的CRISPR indel预测
"""

# ============================================================================
# 导入依赖
# ============================================================================
from common_def import *
from common_def import _to_tensor as _to_tensor_explicit
from models.FeatureEncoder import FeatureEncoder
import torch.nn.functional as F
from types import SimpleNamespace


# ============================================================================
# 全局配置
# ============================================================================
class TrainingConfig:
    """训练配置"""
    # 数据配置
    NUM_SAMPLE = None           # 使用全部数据
    RANDOM_STATE = 42
    FEATURES = "v2"
    
    # 训练超参数
    EPOCHS = 700
    BATCH_SIZE = 64
    LEARING_RATE = 0.001
    
    # 损失函数配置
    LOSS_TYPE = "KL_Div"
    LOSS_PARAMS = {
        "alpha": 0.20,
    }
    
    # 评估配置
    USE_WARMUP = True
    WARMUP_EPOCHS = 15
    EVAL_FREQUENCY = 10


class ModelConfig:
    """模型配置"""
    # 特征编码器
    FEATURE_ENCODER_PARAMS = {
        "embedding_dim": 128,
        "output_dim": 96,
    }
    
    # 长尾分布优化
    FREQ_THRESHOLD = 0.3
    TOP_K = 4
    WEIGHT_RATIO = 35.0
    TEMPERATURE = 0.8           # 固定温度，避免波动
    
    # 稳定性参数
    EMA_DECAY = 0.95            # 参数平滑系数
    GATE_SMOOTHING = 0.7        # 门控平滑系数
    THRESHOLD_SMOOTHING = 0.8   # 阈值平滑系数


# 生成输出模型路径
def get_output_path(config: ModelConfig, train_config: TrainingConfig) -> str:
    """生成输出模型路径"""
    if train_config.LOSS_TYPE == "mse":
        return output_dir + f"dual_freq_v3_stable_WKL0_T{config.TEMPERATURE}_{train_config.LOSS_TYPE}_freq{config.FREQ_THRESHOLD}_{train_config.FEATURES}.pth"
    else:
        return output_dir + f"dual0_freq_v3_stable_WKL0.25_T{config.TEMPERATURE}_{train_config.LOSS_TYPE}_kl_freq{config.FREQ_THRESHOLD}_{train_config.FEATURES}_testall.pth"


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


def get_loss_function(loss_type: str):
    """获取损失函数"""
    if loss_type == "mse":
        return lambda y_pred, y_true: mse_loss(y_pred, y_true)
    else:
        return "KL_Div"


def init_model(num_features: int, seq_feature_dim: int, 
               model_config: ModelConfig, train_config: TrainingConfig):
    """初始化模型、优化器和学习率调度器"""
    model = StableLongTailDualModel(
        num_features=num_features,
        seq_feature_dim=seq_feature_dim,
        hidden_dim=model_config.FEATURE_ENCODER_PARAMS["embedding_dim"],
        out_dim=model_config.FEATURE_ENCODER_PARAMS["output_dim"],
        temperature=model_config.TEMPERATURE,
        freq_threshold=model_config.FREQ_THRESHOLD,
        tail_alpha=0.8,
        ema_decay=model_config.EMA_DECAY,
        gate_smoothing=model_config.GATE_SMOOTHING
    ).to(DEVICE)

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

    loss_fn = get_loss_function(train_config.LOSS_TYPE)
    return model, optimizer, scheduler, loss_fn


# ============================================================================
# 模型定义
# ============================================================================
class StableLongTailDualModel(nn.Module):
    """
    长尾分布双频模型：减少训练过程中的参数波动
    
    核心设计：
    - 头部（高频）：极保守设计，专注Precision
    - 尾部（低频）：极敏感设计，专注Recall
    - 长尾感知门控：基于频率分布特征动态调整
    """
    
    def __init__(self, num_features: int, seq_feature_dim: int, 
                 hidden_dim: int = 128, out_dim: int = 96, 
                 temperature: float = 1.0, freq_threshold: float = 0.3, 
                 tail_alpha: float = 0.8, ema_decay: float = 0.95, 
                 gate_smoothing: float = 0.7):
        super(StableLongTailDualModel, self).__init__()
        
        self.temperature = temperature  
        self.freq_threshold = freq_threshold
        self.tail_alpha = tail_alpha
        self.ema_decay = ema_decay
        self.gate_smoothing = gate_smoothing

        # 共享特征编码
        self.feature_encoder = FeatureEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            output_dim=out_dim
        )

        # 头部（高频）分支：极保守设计
        self.head_branch = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(0.4),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(out_dim // 2, 1)
        )
        
        self.head_residual = nn.Linear(out_dim, 1)
        self.head_residual_weight = nn.Parameter(torch.tensor(0.01))
        self.head_enhancer = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Linear(out_dim // 2, 1)
        )

        # 尾部（低频）分支：极敏感设计
        self.tail_branch = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(0.005),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.002),
            nn.Linear(out_dim // 2, 1)
        )
        
        self.tail_attention = nn.MultiheadAttention(out_dim, num_heads=2, batch_first=True)
        self.tail_attn_norm = nn.LayerNorm(out_dim)
        self.tail_enhancement = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Linear(out_dim // 2, 1)
        )

        # 长尾感知门控（保守版）
        self.longtail_gate = nn.Sequential(
            nn.Linear(out_dim + 4, out_dim // 2),  # +4: confidence, pred_diff, freq_est, tail_est
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(out_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 评估器
        self.frequency_estimator = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.tail_estimator = nn.Sequential(
            nn.Linear(out_dim, out_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # EMA缓存（用于平滑门控权重）
        self.register_buffer('gate_ema', None)
        self.log_t = nn.Parameter(torch.log(torch.tensor(1.0)))

    def current_temperature(self):
        """获取当前温度（动态调整范围）"""
        return torch.exp(self.log_t).clamp(0.1, 2.0)
    
    def _compute_head_logits(self, feat):
        """计算头部分支logits"""
        logit_main = self.head_branch(feat).squeeze(-1)
        logit_residual = self.head_residual(feat).squeeze(-1)
        logit_enhance = self.head_enhancer(feat).squeeze(-1)
        return logit_main + torch.sigmoid(self.head_residual_weight) * logit_residual + 0.05 * logit_enhance
    
    def _compute_tail_logits(self, feat):
        """计算尾部分支logits"""
        attn_feat, _ = self.tail_attention(feat, feat, feat)
        attn_feat = self.tail_attn_norm(feat + attn_feat)
        logit_main = self.tail_branch(attn_feat).squeeze(-1)
        logit_enhance = self.tail_enhancement(feat).squeeze(-1)
        return logit_main + 0.6 * logit_enhance
    
    def _compute_gate(self, feat, logit_head, logit_tail):
        """计算长尾感知门控权重"""
        confidence = self.confidence_estimator(feat).squeeze(-1)
        frequency_est = self.frequency_estimator(feat).squeeze(-1)
        tail_est = self.tail_estimator(feat).squeeze(-1)
        pred_diff = torch.abs(logit_head - logit_tail)
        
        gate_input = torch.cat([
            feat, 
            confidence.unsqueeze(-1),
            pred_diff.unsqueeze(-1),
            frequency_est.unsqueeze(-1),
            tail_est.unsqueeze(-1)
        ], dim=-1)
        
        gate_w_raw = self.longtail_gate(gate_input).squeeze(-1)
        gate_w = 0.3 + 0.4 * gate_w_raw  # 范围从[0,1]缩小到[0.3,0.7]
        
        # 训练时使用EMA平滑
        if self.training and self.gate_ema is not None:
            if self.gate_ema.shape == gate_w.shape:
                gate_w = self.ema_decay * self.gate_ema + (1 - self.ema_decay) * gate_w
            self.gate_ema = gate_w.detach()
        elif self.training:
            self.gate_ema = gate_w.detach()
        
        # 长尾感知融合
        head_tail_weight = frequency_est * (1 - tail_est) 
        adaptive_gate = gate_w * (1 + 0.3 * head_tail_weight)  
        adaptive_gate = torch.clamp(adaptive_gate, 0.2, 0.8)
        
        return adaptive_gate, gate_w_raw, gate_w, confidence, frequency_est, tail_est, pred_diff
    
    def forward(self, x, sequences):
        """前向传播"""
        feat = self.feature_encoder(x)
        logit_head = self._compute_head_logits(feat)
        logit_tail = self._compute_tail_logits(feat)
        adaptive_gate, _, _, _, _, _, _ = self._compute_gate(feat, logit_head, logit_tail)
        
        logits = adaptive_gate * logit_head + (1.0 - adaptive_gate) * logit_tail
        probs = torch.softmax(logits / (self.temperature * self.current_temperature()), dim=1)
        return probs

    def forward_with_details(self, x, sequences):
        """前向传播（返回详细信息）"""
        feat = self.feature_encoder(x)
        logit_head = self._compute_head_logits(feat)
        logit_tail = self._compute_tail_logits(feat)
        adaptive_gate, gate_w_raw, gate_w, confidence, frequency_est, tail_est, pred_diff = \
            self._compute_gate(feat, logit_head, logit_tail)
        
        logits = adaptive_gate * logit_head + (1.0 - adaptive_gate) * logit_tail
        
        probs_fused = torch.softmax(logits / (self.temperature * self.current_temperature()), dim=1)
        probs_head = torch.softmax(logit_head / self.temperature, dim=1)
        probs_tail = torch.softmax(logit_tail / self.temperature, dim=1)
        
        return {
            "logits": logits / self.temperature,
            "probs": probs_fused,
            "probs_head": probs_head,
            "probs_tail": probs_tail,
            "gate": adaptive_gate,
            "gate_raw": gate_w_raw,
            "gate_smoothed": gate_w,
            "confidence": confidence,
            "frequency_est": frequency_est,
            "tail_est": tail_est,
            "pred_diff": pred_diff,
            "head_residual_weight": torch.sigmoid(self.head_residual_weight),
        }


# ============================================================================
# 辅助函数
# ============================================================================
def build_model_config(model_config: ModelConfig, train_config: TrainingConfig,
                       X_shape: tuple, seq_shape: tuple, 
                       best_pearson: float = None, best_epoch: int = None) -> dict:
    """构建模型保存配置"""
    config = {
        "model_type": "stable_longtail_dual",
        "loss_type": train_config.LOSS_TYPE,
        "freq_threshold": model_config.FREQ_THRESHOLD,
        "top_k": model_config.TOP_K,
        "weight_ratio": model_config.WEIGHT_RATIO,
        "num_features": X_shape[1],
        "seq_feature_dim": seq_shape[1],
        "hidden_dim": model_config.FEATURE_ENCODER_PARAMS["embedding_dim"],
        "out_dim": model_config.FEATURE_ENCODER_PARAMS["output_dim"],
        "temperature": model_config.TEMPERATURE,
        "ema_decay": model_config.EMA_DECAY,
        "gate_smoothing": model_config.GATE_SMOOTHING,
    }
    
    if best_pearson is not None:
        config["best_pearson"] = best_pearson
    if best_epoch is not None:
        config["best_epoch"] = best_epoch
    
    return config


def save_model(model, optimizer, lr_scheduler, samples, X_shape, seq_shape, 
               model_config: ModelConfig, train_config: TrainingConfig,
               filepath: str, best_pearson: float = None, best_epoch: int = None):
    """保存模型"""
    torch.save({
        "random_state": train_config.RANDOM_STATE,
        "model": model.state_dict(),
        "samples": samples,
        "loss_type": train_config.LOSS_TYPE,
        "optimiser": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "feature_sets": FEATURE_SETS[train_config.FEATURES],
        "config": build_model_config(
            model_config, train_config, X_shape, seq_shape, 
            best_pearson, best_epoch
        )
    }, filepath)


# ============================================================================
# 训练函数
# ============================================================================
def train_model(X_arrays, Sequences_arrays, y_arrays, samples, 
                model, loss_fn, optimizer, lr_scheduler,
                model_config: ModelConfig, train_config: TrainingConfig,
                output_path: str, A):
    """训练模型主函数"""
    
    # 打印训练配置
    print("\n" + "="*80)
    print("🔥 Stable LongTail Dual Model: 稳定版长尾分布优化")
    print("="*80)
    print(f"📊 训练配置:")
    print(f"   - 模型: StableLongTailDualModel (稳定版长尾分布优化)")
    print(f"   - 编码维度: {model_config.FEATURE_ENCODER_PARAMS['embedding_dim']}→{model_config.FEATURE_ENCODER_PARAMS['output_dim']}")
    print(f"   - Temperature: {model_config.TEMPERATURE} (固定)")
    print(f"   - KLD权重: {train_config.LOSS_PARAMS['alpha']*100:.0f}%")
    print(f"   - 高频阈值: {model_config.FREQ_THRESHOLD}")
    print(f"   - 样本数: {'全部' if train_config.NUM_SAMPLE is None else train_config.NUM_SAMPLE}")
    print(f"   - 稳定性优化:")
    print(f"     • EMA衰减: {model_config.EMA_DECAY}")
    print(f"     • 门控平滑: {model_config.GATE_SMOOTHING}")
    print(f"     • 门控范围: [0.2, 0.8]（限制极端值）")
    print(f"     • 固定温度（无自适应波动）")
    print("="*80 + "\n")

    # 数据划分
    train_samples, val_samples = train_test_split(
        samples, test_size=100, random_state=train_config.RANDOM_STATE
    )
    print(f"📊 训练样本数: {len(train_samples)}, 验证样本数: {len(val_samples)}")

    # 加载测试数据
    test_X, test_y, test_samples, test_sequences = merge_tests(
        t1_path, t2_path, indel_list=ALL_INDELS, mode="union"
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
            batch_samples = train_samples_shuffled[i:i+train_config.BATCH_SIZE]
            model.train()

            # 准备批次数据
            x_batch = torch.stack([_to_tensor_explicit(X_arrays.loc[s]) for s in batch_samples])
            y_batch = torch.stack([_to_tensor_explicit(y_arrays.loc[s]) for s in batch_samples])
            seq_batch = torch.stack([_to_tensor_explicit(Sequences_arrays.loc[s]) for s in batch_samples])
            y_batch = y_batch / (y_batch.sum(dim=1, keepdim=True) + 1e-10)

            # 前向传播
            details = model.forward_with_details(x_batch, seq_batch)
            logits_batch = details['logits']

            # 计算损失
            loss_total = torch.zeros(1).to(DEVICE)
            for bi in range(len(batch_samples)):
                logits = logits_batch[bi]
                y_true = y_batch[bi]

                if loss_fn == "KL_Div":
                    y_pred = torch.softmax(logits, dim=0).clamp_min(1e-12)
                    y_pred_temp = torch.softmax(
                        logits.detach() / model.current_temperature(), dim=0
                    ).clamp_min(1e-12)
                    
                    loss_wkl = weighted_kl_loss(y_pred, y_true, gamma=-0.25)
                    loss_temp = mse_loss(y_pred_temp, y_true, reduction='mean')
                    loss_total += (loss_wkl + loss_temp)
                else:
                    loss_total += loss_fn(y_pred, y_true)

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
        should_eval = (epoch % train_config.EVAL_FREQUENCY == 0 and epoch >= 10) or \
                      (epoch == train_config.EPOCHS - 1)
        
        current_pearson = 0.0
        if should_eval:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{train_config.EPOCHS}")
            print(f"{'='*80}")
            
            print("\n📊 测试集评估:")
            test_reg, test_cls = test_model(
                model, test_X, test_y, test_samples, test_sequences, ALL_INDELS, A
            )
            print_results(test_reg, test_cls)
            
            current_pearson = test_reg['avg_correlation']
            print(f"\n🎯 当前P: {current_pearson:.4f}, 最佳P: {best_pearson:.4f}")
            print("="*80)
            
            # 保存最佳Pearson模型
            if current_pearson > best_pearson:
                best_pearson = current_pearson
                patience_pearson = 0
                print(f"\n🎉 新的最佳Pearson模型! Pearson={best_pearson:.4f}")
                
                save_model(
                    model, optimizer, lr_scheduler, samples,
                    X_arrays.shape, Sequences_arrays.shape,
                    model_config, train_config,
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
                model_config, train_config,
                output_path.replace(".pth", "_best_loss.pth")
            )
        else:
            patience_loss += 1

        # 早停检查
        if patience_loss >= 50:
            print(f"\n🛑 Loss早停! 最佳Loss: {best_loss:.4f}")
            break

    print(f"训练完成!")
    return best_pearson


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    # 初始化配置
    model_config = ModelConfig()
    train_config = TrainingConfig()
    output_path = get_output_path(model_config, train_config)
    
    # 打印配置信息
    print("="*80)
    print("DU-AxisCRISP 删除模块训练")
    print("="*80)
    print(f"配置:")
    print(f"   设备: {DEVICE}")
    print(f"   样本数: {'全部' if train_config.NUM_SAMPLE is None else train_config.NUM_SAMPLE}")
    print(f"   特征集: {train_config.FEATURES}")
    print(f"   Temperature: {model_config.TEMPERATURE} (固定)")
    print(f"   学习率: {train_config.LEARING_RATE}")
    print(f"   Epoch数: {train_config.EPOCHS}")
    print(f"   Batch大小: {train_config.BATCH_SIZE}")
    print(f"   稳定性参数:")
    print(f"     - EMA衰减: {model_config.EMA_DECAY}")
    print(f"     - 门控平滑: {model_config.GATE_SMOOTHING}")
    print(f"     - 阈值平滑: {model_config.THRESHOLD_SMOOTHING}")
    print("="*80 + "\n")

    # 初始化环境
    init_env(train_config)

    # 加载数据
    print("📂 加载数据...")
    with open(indels_sorted_path, "rb") as f:
        ALL_INDELS = pkl.load(f)

    X, y, samples, sequences = load_delete_data(
        filepath=train_file_path, 
        num_samples=train_config.NUM_SAMPLE, 
        fractions=True, 
        indel_list=ALL_INDELS
    )

    X = X.loc[:, FEATURE_SETS[train_config.FEATURES]]
    prior = compute_prior_stable(y)
    A = make_logit_adjustment(prior, tau=1, device=DEVICE)
    
    print(f"   数据加载完成")
    print(f"   样本数: {len(samples)}")
    print(f"   特征数: {X.shape[1]}")
    print(f"   Indel数: {len(ALL_INDELS)}")
    print(f"   序列特征维度: {sequences.shape[1]}")

    # 初始化模型
    print("\n🤖 初始化模型...")
    model, optimizer, lr_scheduler, loss_fn = init_model(
        X.shape[1], sequences.shape[1], model_config, train_config
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 模型初始化完成")
    print(f"   总参数数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")

    # 训练模型
    best_pearson = train_model(
        X, sequences, y, samples, 
        model, loss_fn, optimizer, lr_scheduler,
        model_config, train_config, output_path, A
    )

    # 保存最终模型
    print("\n💾 保存最终模型...")
    save_model(
        model, optimizer, lr_scheduler, samples,
        X.shape, sequences.shape,
        model_config, train_config,
        output_path
    )

    print(f"✅ 模型已保存到: {output_path}")
    print("\n" + "="*80)
    print("🎉 训练完成!")
    print("="*80)
