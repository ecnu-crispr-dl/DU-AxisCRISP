import torch
from torch.nn.utils import weight_norm
import torch.nn as nn
import torch.nn.functional as F
from models.nerTr import NerTrEncoder
class Chomp1d(nn.Module):
    """用于因果卷积裁掉右侧填充，防止泄漏未来信息"""
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, p=0.1, causal=False):
        super().__init__()
        self.causal = causal

        if causal:
            pad = (kernel_size - 1) * dilation  # 因果：全向右填充，然后chomp掉未来
            self.conv1 = weight_norm(nn.Conv1d(in_ch,  out_ch, kernel_size,
                                               padding=pad, dilation=dilation))
            self.chomp1 = Chomp1d(pad)
            self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                               padding=pad, dilation=dilation))
            self.chomp2 = Chomp1d(pad)
        else:
            # 非因果：用same padding，长度严格保持不变
            self.conv1 = weight_norm(nn.Conv1d(in_ch,  out_ch, kernel_size,
                                               padding='same', dilation=dilation))
            self.chomp1 = nn.Identity()
            self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                               padding='same', dilation=dilation))
            self.chomp2 = nn.Identity()

        self.act1  = nn.GELU()
        self.drop1 = nn.Dropout(p)
        self.act2  = nn.GELU()
        self.drop2 = nn.Dropout(p)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.init_weights()

    def init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='linear')

    def forward(self, x):   # x: (B,C,S)
        y = self.conv1(x); y = self.chomp1(y); y = self.act1(y); y = self.drop1(y)
        y = self.conv2(y); y = self.chomp2(y); y = self.act2(y); y = self.drop2(y)
        res = self.downsample(x)
        # 这时 y.shape[-1] == res.shape[-1]
        return F.relu(y + res)

class TemporalBlock_nodilation(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, p=0.1, causal=False):
        super().__init__()
        self.causal = causal

        if causal:
            pad = (kernel_size - 1) * dilation  # 因果：全向右填充，然后chomp掉未来
            self.conv1 = weight_norm(nn.Conv1d(in_ch,  out_ch, kernel_size,
                                               padding=pad))
            self.chomp1 = Chomp1d(pad)
            self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                               padding=pad))
            self.chomp2 = Chomp1d(pad)
        else:
            # 非因果：用same padding，长度严格保持不变
            self.conv1 = weight_norm(nn.Conv1d(in_ch,  out_ch, kernel_size,
                                               padding='same'))
            self.chomp1 = nn.Identity()
            self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                               padding='same'))
            self.chomp2 = nn.Identity()

        self.act1  = nn.GELU()
        self.drop1 = nn.Dropout(p)
        self.act2  = nn.GELU()
        self.drop2 = nn.Dropout(p)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.init_weights()

    def init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='linear')

    def forward(self, x):   # x: (B,C,S)
        y = self.conv1(x); y = self.chomp1(y); y = self.act1(y); y = self.drop1(y)
        y = self.conv2(y); y = self.chomp2(y); y = self.act2(y); y = self.drop2(y)
        res = self.downsample(x)
        # 这时 y.shape[-1] == res.shape[-1]
        return F.relu(y + res)

class TCN(nn.Module):
    def __init__(self, in_ch, ch_list, kernel_size=3, dilations=(1,2), p=0.1, causal=False):
        super().__init__()
        layers = []
        c_prev = in_ch
        for c, d in zip(ch_list, dilations):
            layers += [TemporalBlock(c_prev, c, kernel_size, dilation=d, p=p, causal=causal)]
            c_prev = c
        self.net = nn.Sequential(*layers)

    def forward(self, x):     # x: (B,C,S)
        return self.net(x)
class StructuredKmerEmbedding(nn.Module):
    def __init__(self, d_model=64,len=6):
        super().__init__()
        # per-token 20维 -> d_model（等价于 1x1 Conv）
        self.len=len
        self.proj = nn.Linear(20, d_model)
        #self.pos  = nn.Embedding(20, d_model)  # 位置编码（强烈建议加）

    def forward(self, x384):
        """
        x384: (B, 384)  = [ 1-mer(20*4=80) | 2-mer(19*16=304) ]
        return: (B, 20, d_model)
        """
        B = x384.size(0)
        first_6_nt_feature_indices = list(range(80-4*self.len, 80)) + list(range(384-16*(self.len-1), 384))
        x384=x384[:,first_6_nt_feature_indices]
        # 拆 1-mer
        x1 = x384[:, :4*self.len].reshape(B, -1, 4)                 # (B,20,4)
        # 拆 2-mer（起点0..18），最后一位补零
        x2_core = x384[:, 4*self.len:].reshape(B, -1, 16)           # (B,19,16)
        pad = torch.zeros(B, 1, 16, device=x384.device, dtype=x384.dtype)
        x2 = torch.cat([x2_core, pad], dim=1)               # (B,20,16)

        feat = torch.cat([x1, x2], dim=-1)                  # (B,20,20)
        #tok = self.proj(feat)                               # (B,20,d_model)

        pos_idx = torch.arange(self.len, device=x384.device).unsqueeze(0).expand(B, self.len)
        #tok = tok + self.pos(pos_idx)                       # 加位置
        return feat

class TCNClassifier(nn.Module):
    def __init__(self, d_model=32, num_classes=21, causal=False, p=0.1):
        super().__init__()
        self.len=6
        self.emb = StructuredKmerEmbedding(d_model=20, len=self.len)
        self.in_proj = nn.Linear(20, d_model)
        self.tcn = TCN(in_ch=d_model, ch_list=[d_model, d_model],
                       kernel_size=3, dilations=(1,2), p=p, causal=causal)
        self.tcn2 = TCN(in_ch=self.len, ch_list=[self.len, self.len],
                       kernel_size=3, dilations=(1, 2), p=p, causal=causal)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Flatten(),                 # (B, S*D)
            nn.Linear(d_model*6, 128),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(128, num_classes)
        )

    def forward(self, x0,x):     # x: (B,6,20)
        x=self.emb(x)
        x = self.in_proj(x)   # (B,6,32)
        x=self.tcn2(x)
        '''x = x.transpose(1,2)  # (B,32,6)  -> 给 Conv1d
        x = self.tcn(x)       # (B,32,6)
        x = x.transpose(1,2)  # (B,6,32)'''
        logits = self.out(x)  # (B,21)
        return torch.softmax(logits,dim=-1)
class BiTCN(nn.Module):
    def __init__(self, in_ch, ch_list=(32,32), kernel_size=3, dilations=(1,2), p=0.1, share_weights=False):
        super().__init__()
        self.forward_tcn = TCN(in_ch, ch_list, kernel_size, dilations, p)
        if share_weights:
            self.backward_tcn = self.forward_tcn  # 共享参数
        else:
            self.backward_tcn = TCN(in_ch, ch_list, kernel_size, dilations, p)

    def forward(self, x):  # x: (B,C,S)
        y_f = self.forward_tcn(x)            # (B,Cf,S)
        x_rev = torch.flip(x, dims=[-1])     # 反向
        y_b = self.backward_tcn(x_rev)
        y_b = torch.flip(y_b, dims=[-1])     # 翻回原顺序
        y = torch.cat([y_f, y_b], dim=1)     # (B, Cf+Cb, S)
        return y
class BiTCNClassifier(nn.Module):
    def __init__(self, d_model=32, num_classes=21, p=0.1,
                 ch_list=(32,32), dilations=(1,2), share_weights=False):
        super().__init__()
        self.len = 6
        self.emb = StructuredKmerEmbedding(d_model=20, len=self.len)
        self.in_proj = nn.Linear(20, d_model)          # 20 -> d_model
        self.bitcn = BiTCN(in_ch=d_model, ch_list=ch_list,
                           kernel_size=3, dilations=dilations,
                           p=p, share_weights=share_weights)
        c_last = ch_list[-1] * (2 if not share_weights else 2)  # 双向拼接后通道数
        self.head = nn.Sequential(
            nn.LayerNorm(c_last),
            nn.Flatten(),                               # (B, S * C_last)
            nn.Linear(c_last*6, 128),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(128, num_classes)
        )

    def forward(self,x0, x):             # x: (B,6,20)
        x = self.emb(x)
        x = self.in_proj(x)           # (B,6,d_model)
        x = x.transpose(1, 2)         # (B,d_model,6) 适配 Conv1d 的 (B,C,S)
        y = self.bitcn(x)             # (B, 2*C, 6)
        y = y.transpose(1, 2)         # (B,6, 2*C)
        logits = self.head(y)         # (B,21)
        return torch.softmax(logits,dim=-1)


class MiniSeqTransformer(nn.Module):
    def __init__(self, d_model, head_dim=32, ffn_mult=2, drop=0.1, res_scale=0.5):
        super().__init__()
        self.res_scale = res_scale
        # 单头注意力：在 S=6 下没必要多头，稳定且参数少
        self.q = nn.Linear(d_model, head_dim, bias=False)
        self.k = nn.Linear(d_model, head_dim, bias=False)
        self.v = nn.Linear(d_model, d_model,  bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        hidden = max(d_model, int(d_model*ffn_mult))
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, d_model),
        )
        self.drop = nn.Dropout(drop)
        # 简单可学习位置编码（6步极短时用绝对PE足够）
        self.pos = nn.Parameter(torch.zeros(1, 6, d_model))

    def forward(self, x):              # x: (B,S=6,D)
        B,S,D = x.shape
        h=x + self.pos

        q = self.q(h)                  # (B,S,dh)
        k = self.k(h)                  # (B,S,dh)
        v = self.v(h)                  # (B,S,D)

        attn = torch.matmul(q, k.transpose(1,2)) / (q.shape[-1]**0.5)  # (B,S,S)
        a = torch.softmax(attn, dim=-1)
        y = torch.matmul(a, v)         # (B,S,D)
        y = self.out(y)
        y = x + self.res_scale * self.drop(y)      # 残差缩放

        z = self.ln2(y)
        z = y + self.res_scale * self.drop(self.ffn(z))
        return z                        # (B,S,D)

class SeqRefine(nn.Module):
    def __init__(self, d_model, drop=0.01, res_scale=0.5):
        super().__init__()
        self.pre_ln = nn.LayerNorm(d_model)
        self.fc1x1  = self.conv2 = weight_norm(nn.Conv1d(d_model, d_model, 1,
                                               padding='same', dilation=1))
        self.drop   = nn.Dropout(drop)
        self.res_scale = res_scale
        self.act1 = nn.GELU()
    def forward(self, x):              # x: (B,S,D)
        #y = self.pre_ln(x)
        y = self.fc1x1(x.transpose(1,2)).transpose(1,2)
        y=self.act1(y)
        y = self.drop(y)
        return F.relu(y + x)#x + self.res_scale * y

import torch.nn as nn
import torch.nn.functional as F

class SeqDWConv3_Identity(nn.Module):
    def __init__(self, d_model, p=0.0, res_scale=0.2):
        super().__init__()
        self.pre_ln = nn.LayerNorm(d_model)
        self.dw = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model, bias=False)
        self.pw = nn.Conv1d(d_model, d_model, 1, bias=True)
        self.drop = nn.Dropout(p)
        self.res_scale = res_scale
        # 让 DWConv3 近似恒等：中心权重=1，其余≈0
        with torch.no_grad():
            self.dw.weight.zero_()
            center = 1
            for c in range(d_model):
                self.dw.weight[c, 0, center] = 1.0
            nn.init.zeros_(self.pw.bias)

    def forward(self, x):  # x: (B,S,D)
        #y = self.pre_ln(x).transpose(1,2)   # (B,D,S)
        y = x.transpose(1, 2)
        y = self.dw(y)
        y = F.gelu(self.pw(y))
        y = self.drop(y).transpose(1,2)
        return y#x + self.res_scale * y


class AxisTCN(nn.Module):
    def __init__(self, d_model=32, num_classes=21, p=0.1,len=6):
        super().__init__()
        self.len = len
        self.emb = StructuredKmerEmbedding(d_model=20, len=self.len)
        self.in_proj = nn.Linear(20, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        # 序列轴 TCN：x -> (B, d_model, 6) 再沿 S=6 滑
        #self.tcn_seq = SeqRefine(d_model)
        self.tcn_seq = TCN(in_ch=d_model, ch_list=[d_model],kernel_size=1, dilations=(1,), p=0*p, causal=False)
        self.seq=SeqDWConv3_Identity(d_model)
        # 特征轴 TCN：x -> (B, 6, d_model) 不转置，沿 S=d_model 滑
        self.tcn_feat = TCN(in_ch=self.len, ch_list=[self.len, self.len],
                            kernel_size=3, dilations=(1,2), p=p, causal=False)

        self.tcn_feat2 = TCN(in_ch=self.len, ch_list=[self.len, self.len],
                            kernel_size=3, dilations=(1, 2), p=p, causal=False)
        '''self.fuse_gate = nn.Sequential(
            nn.Conv1d(d_model + self.len, 64, 1), nn.GELU(),
            nn.Conv1d(64, 1, 1)
        )'''
        self.fuse_gate = nn.Sequential(
            nn.Conv1d(d_model*2, d_model*2, 1), nn.GELU(),
            nn.Conv1d(d_model*2, 1, 1)
        )
        self.trans_en=MiniSeqTransformer(d_model)#NerTrEncoder(d_model)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),  # 作用在最后一维
            nn.Flatten(),
            nn.Linear(d_model*self.len, 128), nn.GELU(), nn.Dropout(p),
            nn.Linear(128, num_classes)
        )

    def forward(self, x0, x):
        x = self.emb(x)              # (B,6,20)
        x = self.in_proj(x)          # (B,6,32)

        #y_feat = self.tcn_feat(x)
        #y_feat = self.ln2(y_feat)
        #y_feat = self.trans_en(y_feat)
        x = x.transpose(1,2)     # (B,32,6) 给序列轴 TCN
        y_seq = self.tcn_seq(x)  # (B,32,6)
        y_seq = y_seq.transpose(1,2) # (B,6,32)


        y_feat2 = self.tcn_feat(y_seq)    # (B,6,32?) 取决于实现，保证回到 (B,6,32)

        # 融合（沿“通道维”拼接，再用 1x1 卷积产生门）
        #z = torch.cat([y_seq, y_feat], dim=-1)  # (B, 32+6, 6)
        #g = torch.sigmoid(self.fuse_gate(z))                                  # (B,1,6)
        #y = (1-g) * y_seq.transpose(1,2) + g * y_feat.transpose(1,2)         # (B,32,6)
        #y = y.transpose(1,2)                                                  # (B,6,32)

        logits = self.out(y_feat2)
        return torch.softmax(logits, dim=-1)
class AxisTCN(nn.Module):
    def __init__(self, d_model=32, num_classes=21, p=0.1,len=6):
        super().__init__()
        self.len = len
        self.emb = StructuredKmerEmbedding(d_model=20, len=self.len)
        self.in_proj = nn.Linear(20, d_model)

        # 序列轴 TCN：x -> (B, d_model, 6) 再沿 S=6 滑
        self.tcn_seq = TCN(in_ch=d_model, ch_list=[d_model],kernel_size=1, dilations=(1,), p=0*p, causal=False)

        # 特征轴 TCN：x -> (B, 6, d_model) 不转置，沿 S=d_model 滑
        self.tcn_feat = TCN(in_ch=self.len, ch_list=[self.len, self.len],
                            kernel_size=3, dilations=(1,2), p=p, causal=False)



        self.out = nn.Sequential(
            nn.LayerNorm(d_model),  # 作用在最后一维
            nn.Flatten(),
            nn.Linear(d_model*self.len, 128), nn.GELU(), nn.Dropout(p),
            nn.Linear(128, num_classes)
        )

    def forward(self, x0, x):
        x = self.emb(x)              # (B,6,20)
        x = self.in_proj(x)          # (B,6,32)


        x = x.transpose(1,2)     # (B,32,6) 给序列轴 TCN
        y_seq = self.tcn_seq(x)  # (B,32,6)
        y_seq = y_seq.transpose(1,2) # (B,6,32)

        y_feat2 = self.tcn_feat(y_seq)    # (B,6,32?) 取决于实现，保证回到 (B,6,32)
        logits = self.out(y_feat2)
        return torch.softmax(logits, dim=-1)
