import torch
import torch.nn as nn
import torch.nn.functional as F
from .tabkanet import NumEncoderTransformer


class TabKANFeatureEncoder(nn.Module):
    """
    🔥 TabKAN特征编码器 - 使用NumEncoderTransformer
    参考deletion_3.py的实现
    """
    def __init__(self, num_features, embedding_dim=64, output_dim=64):
        super(TabKANFeatureEncoder, self).__init__()
        
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        # NumEncoderTransformer: 将num_features编码成embedding_dim
        self.num_encoder = NumEncoderTransformer(num_features, embedding_dim)
        
        '''# 后处理层
        self.post_encoder = nn.Sequential(
            nn.LayerNorm(num_features * embedding_dim),
            nn.Dropout(0.1),
            nn.Linear(num_features * embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, output_dim)
        )'''
    
    def forward(self, x):
        """
        Args:
            x: (B, N, num_features) - batch, num_indels, features
        Returns:
            features: (B, N, output_dim)
        """
        B, N, F = x.shape
        
        # 展平处理: (B, N, F) → (B*N, F)
        x_flat = x.view(B * N, F)
        
        # NumEncoderTransformer编码
        # (B*N, F) → (B*N, F*embedding_dim)
        x_encoded = self.num_encoder(x_flat)
        
        # 后处理
        # (B*N, F*embedding_dim) → (B*N, output_dim)
        #features = self.post_encoder(x_encoded)
        
        # 恢复形状: (B*N, output_dim) → (B, N, output_dim)
        features = x_encoded.view(B, N, -1)
        
        return features
