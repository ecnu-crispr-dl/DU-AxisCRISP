import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn as nn
import pickle as pkl
import numpy as np
import math
from types import SimpleNamespace
import torch.nn.functional as F
from torch.distributions.normal import Normal
#from dispatcher.Other import SparseDispatcher, FourierLayer, series_decomp_multi, MLP
from sklearn.metrics import precision_score
import numpy as np


import numpy as np

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class EPABlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm2 = nn.BatchNorm2d(dim)

        # Simple Pixel Attention
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            #nn.Dropout(0.3),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1),
            #nn.Dropout(0.3)
        )

    def forward(self, x):
        identity = x
        x = self.norm2(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x
class pFFN1(nn.Module):
    def __init__(self, configs,patch):
        super(pFFN1, self).__init__()
        self.patch = patch

        # Patch 内建模（局部特征增强）
        '''self.local_conv = nn.Sequential(
            nn.Conv1d(configs.d_model, configs.d_model, kernel_size=3, padding=1, groups=configs.d_model),  # depthwise conv
            nn.GELU(),
            nn.Conv1d(configs.d_model, configs.d_model, kernel_size=1),
            nn.Dropout(0.2)
        )'''
        self.ffn1pw1 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_ff, kernel_size=1)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=configs.d_ff, out_channels=configs.d_model, kernel_size=1)
        self.ffn1drop1 = nn.Dropout(configs.dropout)
        self.ffn1drop2 = nn.Dropout(configs.dropout)


    def forward(self, x):
        # Patch内局部建模
        '''out, balance_loss1 = self.patchmoe(out)
        local_feat = out.permute(0, 3, 1, 2).contiguous()'''
        B, n_patch,patch, N=x.shape
        #print(B, n_patch,patch,N)
        x_patch = x.permute(0, 1, 3, 2).contiguous()  # [B, n_patch, D, patch]
        x_patch = x_patch.view(B * n_patch, N, patch)  # [B*n_patch, D, patch]
        # local_feat = self.local_conv(x_patch)  # [B*n_patch, D, patch]
        local_feat = self.ffn1drop1(self.ffn1pw1(x_patch))
        local_feat = self.ffn1act(local_feat)
        local_feat = self.ffn1drop2(self.ffn1pw2(local_feat))

        local_feat = local_feat.view(B, n_patch, N, patch).permute(0, 2, 1,3).contiguous()  # [B, D, n_patch, patch]


        # out = self.TCN_block(out)
        #out = self.epa(x_merge + local_feat)
        return local_feat
class pFFN2(nn.Module):
    def __init__(self, configs,patch):
        super(pFFN2, self).__init__()
        self.patch = patch

        # Patch 间建模：类 FFN2（跨 patch 通道建模）
        self.ffn2pw1 = nn.Conv1d(in_channels=configs.d_model * patch, out_channels=configs.d_ff, kernel_size=1,
                                 groups=configs.d_model)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=configs.d_ff, out_channels=configs.d_model * patch, kernel_size=1,
                                 groups=configs.d_model)
        self.ffn2drop1 = nn.Dropout(0.3)
        self.ffn2drop2 = nn.Dropout(0.3)

        #self.epa = EPABlock(configs.d_model)
    def forward(self, x):
        # Patch内局部建模
        '''out, balance_loss1 = self.patchmoe(out)
        local_feat = out.permute(0, 3, 1, 2).contiguous()'''
        B, N, n_patch,patch=x.shape


        # Patch间建模（类 FFN2）
        x_merge = x.permute(0, 1, 3, 2).contiguous().view(B, N * patch,n_patch)  # [B, D*patch, n_patch]
        x_merge = self.ffn2pw1(x_merge)
        x_merge = self.ffn2drop1(x_merge)
        x_merge = self.ffn2act(x_merge)
        x_merge = self.ffn2pw2(x_merge)
        x_merge = self.ffn2drop2(x_merge)
        out = x_merge.view(B, N, patch, n_patch).permute(0, 1, 3, 2).contiguous()  # [B, D, n_patch, patch]

        '''out, balance_loss1 = self.patchmoe((x_merge+local_feat).permute(0, 2, 3, 1).contiguous())
        out = out.permute(0, 3, 1, 2).contiguous()'''
        #out = self.epa(x_merge + local_feat)
        return out
class expert_Layer(nn.Module):
    def __init__(self, configs, seq_len, pred_len,top_k,patch):
        super(expert_Layer, self).__init__()
        #self.cpa=CrossPatchAttention(configs.d_model*patch)
        self.seq_len=seq_len
        self.pred_len=pred_len
        self.k=top_k
        self.patch=patch
        if self.seq_len % patch != 0:
            self.length = (((self.seq_len + self.pred_len) // patch) + 1) * patch
            #self.padding = torch.zeros([128, (length - (self.seq_len + self.pred_len)), configs.d_model])
            #out = torch.cat([x, padding], dim=1)
        else:
            self.length = (self.seq_len + self.pred_len)
            #out = x

        self.norm = nn.LayerNorm(configs.d_model)



        self.pffn1=pFFN1(configs,patch)
        #self.pffn2=pFFN2(configs,patch)
        #self.epa=EPABlock(configs.d_model)

    def forward(self,x):
        B, T, N = x.size()
        # print(B,T,N)

        # padding
        if self.seq_len % self.patch != 0:
            padding = torch.zeros([x.shape[0], (self.length - (self.seq_len + self.pred_len)), N]).to(x.device)
            out1 = torch.cat([x, padding], dim=1)
        else:
            out1 = x
            # reshape
        out1 = out1.reshape(B, self.length// self.patch, self.patch,N)#.permute(0, 3, 1, 2).contiguous()

        #out1=self.cpa(out1, out1)
        # 2D conv: from 1d Variation to 2d Variation
        #out1 = self.convs(out1)  #
        out1 = self.pffn1(out1)

        # out=self.TCN_block1d_2(out)
        out1 = out1.reshape(B, N, self.length// self.patch*self.patch)
        out1 = out1.permute(0, 2, 1)
        out1=out1[:, :(self.seq_len + self.pred_len), :]


        #stride = self.patch // 2  # 步长为 patch 大小的一半

        '''# x: [B, T, N]
        if (self.seq_len - self.patch)% stride != 0:
            pad_len = stride - (self.seq_len + self.pred_len - self.patch) % stride
            padding = torch.zeros([x.shape[0], pad_len, x.shape[2]], device=x.device)
            x_reshape = torch.cat([x, padding], dim=1)
        else:
            pad_len=0
            x_reshape = x'''

        '''# 更新长度
        total_len = self.seq_len  # 原始序列长度 + padding
        num_patches = (total_len - self.patch) // stride + 1

        # 使用 unfold 构造 patch: [B, N, num_patches, patch]
        x1 = out1.transpose(1, 2)  # [B, N, T]
        patches = x1.unfold(dimension=2, size=self.patch, step=stride)  # [B, N, num_patches, patch]
        #patches = patches.permute(0, 2, 3, 1).contiguous()  # [B, num_patches, patch, N]

        # add
        out2 = self.pffn2(patches).permute(0, 2, 3, 1).contiguous()
        #out2 = out2.reshape(B, -1, N)
        #out2 = out2[:, :(self.tolength-pad_len), :]
        reconstructed = torch.zeros(B, total_len, N, device=out2.device)
        count = torch.zeros(B, total_len, N, device=out2.device)

        for i in range(num_patches):
            start = i * stride
            end = start + self.patch
            reconstructed[:, start:end, :] += out2[:, i, :, :]  # [B, patch, N]
            count[:, start:end, :] += 1

        # 避免除以0
        count = torch.clamp(count, min=1)
        reconstructed = reconstructed / count
        out2 = reconstructed[:, :self.seq_len + self.pred_len, :]  # [B, original_len, N]'''

        out=out1#+out2
        # padding
        '''if self.seq_len % self.patch != 0:
            padding = torch.zeros([out.shape[0], (self.length - (self.seq_len + self.pred_len)), N]).to(x.device)
            out = torch.cat([out, padding], dim=1)
        out=out.reshape(B, self.length // self.patch, self.patch, N).permute(0, 3, 1, 2).contiguous()
        #out=self.epa(out)
        #out=torch.cat([out1, out2], dim=-1)
        #out=self.fuse_linear(out.reshape(-1,2*N)).reshape(B,T,N)+ x
        #out = self.fuse_linear(out)+x
        #out=self.norm(out)
        out = out.reshape(B, N, self.length// self.patch*self.patch)
        out = out.permute(0, 2, 1)
        out=out[:, :(self.seq_len + self.pred_len), :]'''



        return out+x#,balance_loss1

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1, bidirectional=True):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,     # 输入形状为 [B, T, N]
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)

    def forward(self, x):
        # x shape: [B, T, N]
        #print(x.shape)
        output, (hn, cn) = self.lstm(x)  # output: [B, T, hidden_dim * num_directions]
        return output#, hn, cn
class Model_v0(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, enc_in, seq_len,d_model, embed, freq,dropout,class_num=21, epnum=3):
        super(Model_v0, self).__init__()
        self.seq_len = seq_len//4

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq,
                                           dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.bilstm=BiLSTMEncoder(input_dim=d_model, hidden_dim=int(d_model/2))

        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        #self.projection = nn.Linear(d_model * configs.seq_len, configs.num_class)
        #self.input_linear = nn.Linear(configs.enc_in, configs.d_model)
        #self.labelmoe=label_moe(configs,configs.num_class,3)
        self.linear_main = nn.Sequential(
            nn.Linear(d_model * self.seq_len, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, class_num),)
    def classification(self, x_enc,features=None):
        B=x_enc.shape[0]
        x_enc=x_enc.reshape(B,-1,4)
        balance_loss=0
        # embedding
        #print(x_enc.shape)
        if torch.isinf(x_enc).any() or torch.isnan(x_enc).any():
            print("wrong input of embedding")
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        #enc_out=nn.Sigmoid()(self.input_linear(x_enc))
        '''if torch.isinf(enc_out).any() or torch.isnan(enc_out).any():
            print("wrong input of bilstm")
        '''
        enc_out = self.bilstm(enc_out)  # bilstm
        #enc_out, attn_output_weights=self.attn(enc_out,enc_out,enc_out)
        # TimesNet

        enc_out=self.layer_norm(enc_out)
        enc_out=self.linear_main(enc_out.reshape(B,-1))
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        #output = self.act(enc_out)
        #output = self.dropout(output)

        # (batch_size, seq_length * d_model)
        output=torch.softmax(enc_out, dim=1)
        #output = enc_out.reshape(enc_out.shape[0], -1)
        '''features_out=self.NeuralNetwork(features)
        combined_out = torch.cat([output, features_out], dim=-1)'''
        #output = self.projection(output)  # (batch_size, num_classes)
        #???
        #output = nn.Softmax()(output)
        #output,beloss=self.labelmoe(output)
        #balance_loss+=beloss
        return output

    def forward(self, x,x_enc,features=None):
        dec_out = self.classification(x_enc,features)
        return dec_out  # [B, N]


'''configs = SimpleNamespace(
        seq_len=18,
        pred_len=0,  # 分类任务无预测长度
        enc_in=4,  # 输入特征数量
        d_model=64,
        c_out=1,  # 分类任务类别数量
        num_class=1,  # 分类任务类别数量
        e_layers=10,  # 3,  # TimesBlock 的层数
        top_k=2,
        num_kernels=6,
        d_ff=128,
        dropout=0.1,
        embed='fixed',
        freq='h'
    )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=Model_moe(configs, 2).to(device)'''
