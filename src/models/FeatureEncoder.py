import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEncoder(nn.Module):
    """
    可插拔的特征编码器
    用于处理删除特征 (B, N, num_features)
    """
    def __init__(self, num_features, hidden_dim=64, output_dim=64):
        super(FeatureEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, N, num_features) - batch, num_indels, features
        Returns:
            features: (B, N, output_dim)
        """
        B, N, F = x.shape
        
        # 展平处理
        x_flat = x.view(B * N, F)
        
        # 编码
        features = self.encoder(x_flat)  # (B*N, output_dim)
        
        # 恢复形状
        features = features.view(B, N, -1)  # (B, N, output_dim)
        
        return features
