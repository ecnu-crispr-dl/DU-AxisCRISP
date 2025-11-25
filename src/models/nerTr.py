
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.encoder import DataEmbedding
'''torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()'''
class DecoderLabelQueryHead(nn.Module):
    def __init__(self, num_classes, feature_dim, use_cosine_sim=False):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_cosine_sim = use_cosine_sim

        # 可学习的 label queries
        self.label_queries = nn.Parameter(torch.randn(num_classes, feature_dim))

        # 分类头（可以是简单线性层，或 non-linear）
        self.fc = nn.Linear(feature_dim, 1)
    def forward(self, indel_feats):
        """
        indel_feats: Tensor of shape (B, N, D)
        return: logits: shape (B, C)
        """
        B, N, D = indel_feats.shape
        C = self.num_classes

        # Expand label queries to batch
        queries = self.label_queries.unsqueeze(0).expand(B, -1, -1)  # (B, C, D)

        # Attention: [B, C, N] = Q·K^T / sqrt(d)
        attn_scores = torch.matmul(queries, indel_feats.transpose(1, 2)) / (D ** 0.5)

        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, C, N)

        # Attention weighted sum: (B, C, D)
        context = torch.matmul(attn_weights, indel_feats)

        # Feed into classification layer
        logits = self.fc(context).squeeze(-1)  # (B, C)
        return logits  # 可配合 BCEWithLogitsLoss 用于多标签
class LiteNerTrDecoder(nn.Module):
    def __init__(self, num_ner, sim_dim, device):
        super(LiteNerTrDecoder, self).__init__()
        self.num_ner = num_ner
        self.device = device
        self.query_embed = nn.Embedding(num_ner, sim_dim)
        self.attn = nn.MultiheadAttention(embed_dim=sim_dim, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(sim_dim, sim_dim),
            nn.ReLU(),
            nn.Linear(sim_dim, sim_dim)
        )
        self.norm = nn.LayerNorm(sim_dim)

    def forward_query_ner(self, semantic):
        # semantic: (B, T, D)
        B = semantic.size(0)
        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, num_ner, D)
        attn_output, _ = self.attn(query, semantic, semantic)          # (B, num_ner, D)
        output = self.ffn(attn_output)
        return self.norm(output)

    def get_cos_sim(self, query_result, text_embedding):
        return torch.cosine_similarity(query_result.unsqueeze(1),
                                       text_embedding.unsqueeze(-2), dim=-1)

    def forward(self, semantic):
        query_result = self.forward_query_ner(semantic)
        cos_sim = self.get_cos_sim(query_result, semantic)
        return query_result, cos_sim


class NerTrDecoder(torch.nn.Module):
    def __init__(self, num_ner, sim_dim, device):
        super(NerTrDecoder, self).__init__()
        self.num_ner = num_ner
        self.device = device
        self.obj_query_embedding = torch.nn.Embedding(num_ner, sim_dim)
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=sim_dim,
                                                              nhead=2,
                                                              batch_first=True)
        self.decoder = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        self.normalize = torch.nn.LayerNorm(normalized_shape=sim_dim)

    def get_obj_query_embedding(self, batch_size):
        obj_query_index = torch.tensor([x for x in range(self.num_ner)])
        obj_query_embedding = self.obj_query_embedding(obj_query_index.to(self.device))
        obj_query_embedding_batched = obj_query_embedding.repeat(batch_size, 1, 1)
        return obj_query_embedding_batched.to(self.device)

    def forward_query_ner(self, semantic):
        obj_query_embedding_batched = self.get_obj_query_embedding(semantic.shape[0])
        query_result = self.decoder(memory=semantic,
                                    tgt=obj_query_embedding_batched)
        return query_result

    def get_cos_sim(self, query_result, text_embedding):
        cos_sim = torch.cosine_similarity(query_result.unsqueeze(1),
                                          text_embedding.unsqueeze(-2),
                                          dim=-1)
        return cos_sim

    def forward(self, semantic):
        query_result = self.forward_query_ner(semantic)
        cos_sim = self.get_cos_sim(query_result, semantic)
        return query_result, cos_sim

class NerTrEncoder(torch.nn.Module):
    def __init__(self, sim_dim):
        super(NerTrEncoder, self).__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=sim_dim,dim_feedforward=256,
                                                          nhead=2,
                                                          batch_first=True,dropout=0.1)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer,
                                                   num_layers=1)
        self.normalize = torch.nn.LayerNorm(normalized_shape=sim_dim)
        #self.norm2 = torch.nn.LayerNorm(normalized_shape=sim_dim)

    def forward(self, bert_future):
        semantic = self.normalize(bert_future)
        semantic = self.encoder(semantic)

        return semantic

class StructuredKmerEmbedding(nn.Module):
    def __init__(self, d_model=64,len=5):
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


class RelPosDecaySelfAttn(nn.Module):
    """
    自注意力（batch_first=True）+ 相对位置衰减 + 切点邻域偏置
    x: (B, N, D)
    key_padding_mask: (B, N)  1=padding，应屏蔽；0=有效
    cut_idx: (B,) 或 int，切点位置（可选）
    """
    def __init__(self, cut_idx,d_model=64, nhead=4, dropout=0.1, alpha=0.0, beta=0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.cut_idx =cut_idx
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # 衰减强度（可学习，也可固定）

        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=False)
        self.beta  = nn.Parameter(torch.tensor(beta,  dtype=torch.float32), requires_grad=False)

    @staticmethod
    def _distance_bias(N, device, dtype):
        # |i - j| 距离矩阵 -> (1, 1, N, N) 便于广播到 (B, H, N, N)
        idx = torch.arange(N, device=device)
        dist = (idx[None, :] - idx[:, None]).abs().to(dtype)
        return dist[None, None, :, :]  # (1,1,N,N)

    @staticmethod
    def _cut_site_bias(N, cut_idx, device, dtype):
        # 对 key 位置 j 施加与切点距离的衰减：-|j - c|
        # 支持 cut_idx 为 int 或 (B,) 张量
        j = torch.arange(N, device=device)
        if isinstance(cut_idx, int):
            cj = (j - cut_idx).abs().to(dtype)             # (N,)
            bias = cj[None, None, None, :]                 # (1,1,1,N)
        else:
            # cut_idx: (B,)
            B = cut_idx.shape[0]
            cj = (j[None, :] - cut_idx[:, None]).abs().to(dtype)  # (B,N)
            bias = cj[:, None, None, :]                              # (B,1,1,N)
        return bias

    def forward(self, x, key_padding_mask=None):
        B, N, D = x.shape
        H, Dh = self.nhead, self.d_head

        q = self.q_proj(x).view(B, N, H, Dh).transpose(1, 2)  # (B,H,N,Dh)
        k = self.k_proj(x).view(B, N, H, Dh).transpose(1, 2)  # (B,H,N,Dh)
        v = self.v_proj(x).view(B, N, H, Dh).transpose(1, 2)  # (B,H,N,Dh)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # (B,H,N,N)

        # 相对距离衰减：-alpha * |i-j|
        dist_bias = self._distance_bias(N, x.device, x.dtype)               # (1,1,N,N)
        attn_logits = attn_logits - self.alpha.clamp_min(0.0) * dist_bias

        # 切点邻域偏置：-beta * |j - cut|
        if self.cut_idx is not None and self.beta.item() != 0.0:
            cut_bias = self._cut_site_bias(N, self.cut_idx, x.device, x.dtype)   # (B,1,1,N) 或 (1,1,1,N)
            attn_logits = attn_logits - self.beta.clamp_min(0.0) * cut_bias


        # padding mask: True=pad -> -inf
        if key_padding_mask is not None:
            # (B,N) -> (B,1,1,N)
            mask = key_padding_mask[:, None, None, :].to(torch.bool)
            attn_logits = attn_logits.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn_logits, dim=-1)  # (B,H,N,N)
        #attn = apply_multiplicative_prior(attn, self.alpha, self.beta, self.cut_idx)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)            # (B,H,N,Dh)
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # (B,N,D)
        out = self.o_proj(out)
        return out, attn  # 返回注意力权重便于可视化
def apply_multiplicative_prior(attn_probs, alpha, beta, cut_idx=None):
    """
    attn_probs: softmax 后的 (B,H,N,N)
    返回: 乘以 prior 再归一化的后验
    prior_{ij} = exp(-alpha*|i-j| - beta*|j-cut|)
    """
    B, H, N, _ = attn_probs.shape
    device, dtype = attn_probs.device, attn_probs.dtype

    # |i-j|
    idx = torch.arange(N, device=device)
    dist = (idx[None, :] - idx[:, None]).abs().to(dtype)   # (N,N)
    prior = torch.exp(-alpha.clamp_min(0.0) * dist)        # (N,N)

    # |j-cut|
    if cut_idx is not None and beta.item() != 0.0:
        j = torch.arange(N, device=device)
        if isinstance(cut_idx, int):
            cj = (j - cut_idx).abs().to(dtype)             # (N,)
            key_prior = torch.exp(-beta.clamp_min(0.0) * cj)[None, None, None, :]  # (1,1,1,N)
        else:
            cj = (j[None, :] - cut_idx[:, None]).abs().to(dtype)  # (B,N)
            key_prior = torch.exp(-beta.clamp_min(0.0) * cj)[:, None, None, :]     # (B,1,1,N)
        prior = prior[None, None, :, :] * key_prior         # (B,1,N,N)
    else:
        prior = prior[None, None, :, :]                     # (1,1,N,N)

    post = attn_probs * prior                               # 乘先验
    post = post / (post.sum(dim=-1, keepdim=True) + 1e-9)   # 再归一化
    return post

class EncoderBlock(nn.Module):
    def __init__(self, cut_idx,d_model=64, nhead=4, ffn_ratio=64, p=0.1, alpha=0.1, beta=0.2):
        super().__init__()
        self.attn = RelPosDecaySelfAttn(cut_idx,d_model, nhead, p, alpha, beta)
        self.ln1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(p)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*ffn_ratio),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(d_model*ffn_ratio, d_model),
        )
        self.drop2 = nn.Dropout(p)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        h, _ = self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = self.ln2(x + self.drop1(h))
        x = self.ln3(x + self.drop2(self.ffn(x)))
        return x
class DSResBlock(nn.Module):
    def __init__(self, d_model,len, dw_kernel=3, dilation=1, p=0.05):
        super().__init__()
        self.pw1 = nn.Conv1d(d_model, d_model, 1)
        self.dw  = nn.Conv1d(d_model, d_model, dw_kernel,
                             padding=dilation*(dw_kernel//2), dilation=dilation, groups=d_model)
        self.ln  = nn.LayerNorm(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, 1)
        self.drop= nn.Dropout(p)
        self.act = nn.GELU()

    def forward(self, x):        # x: (B,S,D)
        x1 = x.transpose(1,2)    # (B,D,S)
        y  = self.pw1(x1)
        y  = self.dw(y).transpose(1,2)   # (B,S,D)
        y  = self.act(self.ln(y)).transpose(1,2)  # LN 在 (B,S,D)
        y  = self.pw2(y)#.transpose(1,2)
        y  = self.drop(y)
        return (y + x1).transpose(1,2)   # 残差

class NerTr(torch.nn.Module):
    def __init__(self,  sim_dim, num_ner,seq_len,  device):
        '''
        :param bert_model: the huggingface bert model
        :param sim_dim: the dimention of the bert model like 768
        :param num_ner: the num of kinds of ner
        :param ann_type: croase or fine
        '''
        super(NerTr, self).__init__()
        self.device = device
        #self.ann_type = ann_type
        #self.bert_model = bert_model
        self.sim_dim = sim_dim
        outdim=20
        self.len=6
        #self.alignment = alignment
        #self.encoder = NerTrEncoder(sim_dim=self.len)
        #self.encoder_p=EncoderBlock(cut_idx=self.len,d_model=outdim,nhead=2,alpha=0,beta=0.5)
        self.encoder_p=NerTrEncoder(sim_dim=outdim)
        self.decoder = NerTrDecoder(num_ner=num_ner, sim_dim=outdim, device=self.device)
        self.linear = torch.nn.Linear(in_features= outdim*seq_len//4, out_features=num_ner)
        self.linear01= torch.nn.Linear(in_features=outdim, out_features=1)
        #self.crf = CRF(num_ner, batch_first=True)
        self.normalize = torch.nn.LayerNorm(normalized_shape=outdim)
        self.linear_stack = nn.Sequential(
            nn.Linear(sim_dim, outdim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(outdim, outdim),
        )
        self.DSRes=DSResBlock(outdim,len=self.len)
        self.out_linear  = nn.Sequential(
            nn.Linear(outdim*self.len, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_ner),
        )
        self.softmax = nn.Softmax(dim=1)
        self.emb=StructuredKmerEmbedding(d_model=outdim,len=self.len)
        self.gate=nn.Sequential(
            nn.Linear(outdim*self.len*2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

    def encode(self, data0):
        """返回分类头之前的序列级表征 seq_repr：(B, outdim*self.len)"""
        B = data0.shape[0]
        x = self.emb(data0)  # (B, len, outdim)
        out = self.encoder_p(x)  # 你现在用的编码器输出 (B, len, outdim)
        seq_repr = out.reshape(B, -1)  # 展平作为序列级表征
        return seq_repr

    def head(self, seq_repr):
        """从序列级表征得到 logits：(B, num_ner)"""
        return self.out_linear(seq_repr)

    # 原 forward 保持不变（或内部直接用上面两步）
    '''def forward(self, x, data0, seq=None, mask=None):
        seq_repr = self.encode(data0)
        logits = self.head(seq_repr)
        return self.softmax(logits)'''
    def forward(self, x,data0, seq=None, mask=None):
        B = data0.shape[0]
        #data = data.reshape(B, -1, 4)
        data=self.emb(data0)

        dsr=self.DSRes(data)
        out = self.encoder_p(dsr)
        '''w=self.gate( (torch.cat([dsr, out], dim=-1)).reshape(B,-1) )
        w1 = w[:, 0].unsqueeze(-1).unsqueeze(-1)
        w2 = w[:, 1].unsqueeze(-1).unsqueeze(-1)
        out = w1 * dsr + w2 * out'''

        '''#x = self.linear_stack(data)  # (B, N, D)
        data1=data.permute(0,2,1)
        semantic = self.encoder(data1).permute(0,2,1)  # (B, N, D)
        #semantic = 0.3*semantic+sem_p
        data=self.normalize(data)
        semantic=torch.cat([
            #data,
            semantic,
            sem_p,

        ], dim=-1)'''
        # 池化成序列级向量 g
        '''if mask is not None:
            m = mask.unsqueeze(-1).float()
            g = (semantic * m).sum(1) / m.sum(1).clamp_min(1e-6)  # (B, D)
        else:
            g = semantic.mean(1)  # (B, D)'''

        # 解码器生成每类查询向量
        #q, _ = self.decoder(semantic)  # q: (B, 21, D)

        '''# 相似度作为 logits（可学习温度）
        g_norm = torch.nn.functional.normalize(g, dim=-1)  # (B, D)
        q_norm = torch.nn.functional.normalize(q, dim=-1)  # (B, 21, D)
        logits = torch.einsum('bd,bkd->bk', g_norm, q_norm)  # (B, 21)'''
        logits = self.out_linear(out.reshape(B,-1))

        # 可加 tau: logits = logits / self.tau
        return self.softmax(logits)

    '''def forward(self, x,data,mask=None):
        #可以尝试indel和feature维度交换提取特征
        B = data.shape[0]
        data = data.reshape(B,-1,4)
        #data=data.permute(0, 2, 1).contiguous()
        #print(0)

        #output_encoder: [b_s, max_num_token, sim_dim]
        data= self.linear_stack(data)#.permute(0, 2, 1).contiguous()
        output_encoder = self.encoder(data)
        output_encoder = self.normalize(output_encoder)
        #decoder_embedding:[b_s, num_kind_of_ner, sim_dim], the embedding of quert
        #cos_sim: [b_s, max_num_token, num_kind_of_ner],
        #the cos_sim between output_encoder and decoder_embedding, e.g. the similarity of
        #tokens and query embedding
        #print(0)
        decoder_embedding, cos_sim = self.decoder(output_encoder)
        #print(1)
        #transform cos_sim into prob
        cos_sim_prob = torch.softmax(cos_sim, dim=-1)
        prob_query = self.prob_times_query(cos_sim_prob, decoder_embedding)
        embedding_with_prob_query = data + prob_query
        embedding_with_prob_query = self.normalize(embedding_with_prob_query)
        #print(embedding_with_prob_query.shape)
        ner_prob = self.softmax(self.linear(embedding_with_prob_query.float().reshape(B,-1)))
        #print(2)
        return ner_prob
'''
    def prob_times_query_diag(self, cos_sim_prob, decoder_embedding):
        """
        每个 indel 只与 decoder_embedding 中同 index 的 label emb 相乘
        :param cos_sim_prob: [B, N, N]
        :param decoder_embedding: [B, N, sim_dim]
        :return: [B, N, sim_dim]
        """
        B, N, _ = cos_sim_prob.shape
        _, _, D = decoder_embedding.shape

        # 提取对角线的相似度得分：[B, N]
        #或者加一个linear(n,1)
        diag_probs = cos_sim_prob.diagonal(dim1=1, dim2=2)  # [B, N]

        # 需要扩展维度以匹配 embedding 相乘：[B, N, 1]
        diag_probs = diag_probs.unsqueeze(-1)

        # 相乘：[B, N, sim_dim]
        result = diag_probs * decoder_embedding

        return result

    def prob_times_query(self, cos_sim_prob, decoder_embedding):

        '''b_s, prob_row, prob_clum = cos_sim_prob.shape
        _, query_row, query_clum = decoder_embedding.shape
        result = torch.zeros((b_s, prob_row, query_clum)).to(self.device)
        for b_s_index in range(b_s):
            for prob_row_index in range(prob_row):
                result_per_row = torch.zeros((prob_clum, query_clum)).to(self.device)
                for y in range(prob_clum):
                    result_per_row[y] = \
                        cos_sim_prob[b_s_index, prob_row_index, y] \
                        * decoder_embedding[b_s_index, y, :]
                result[b_s_index, prob_row_index, :] = \
                    torch.sum(result_per_row, dim=0, keepdim=True)

        return result'''
        return torch.bmm(cos_sim_prob, decoder_embedding)

    def get_bert_feature_bulk(self, data):
        output_bert = self.bert_model(input_ids=data['input_ids'],
                                      attention_mask=data['attention_mask_bert'])
        last_hidden_state = output_bert['last_hidden_state']
        b_s, _, sim_dim = last_hidden_state.shape
        bert_feature_bulk = []
        for b_s_index in range(b_s):
            word_ids_one_batch = data['words_ids'][b_s_index]
            index_not_none = torch.where(word_ids_one_batch!=-100)
            last_hidden_state_one_batch = last_hidden_state[b_s_index]
            last_hidden_state_real_word = last_hidden_state_one_batch[index_not_none]
            word_ids_one_batch_not_none = word_ids_one_batch[index_not_none]
            _, indices = torch.unique(word_ids_one_batch_not_none, return_inverse=True)
            grouped_bert_embedding = []
            for i in range(torch.max(indices) + 1):
                grouped_bert_embedding.append(last_hidden_state_real_word
                                                 [indices == i])
            grouped_bert_embedding_avg = [torch.mean(v, dim=0)
                                          for v in grouped_bert_embedding]
            bert_feature_bulk.append(torch.stack(grouped_bert_embedding_avg))

        return torch.stack(bert_feature_bulk)

    def post_process_flow(self, ner_prob, data):
        b_s, num_tokens, num_ner = ner_prob.shape
        prob = []
        for b_s_index in range(b_s):
            word_ids_one_batch = data['words_ids'][b_s_index]
            index_not_none = torch.where(word_ids_one_batch != -100)
            ner_prob_one_batch = ner_prob[b_s_index]
            word_ids_one_batch_not_none = word_ids_one_batch[index_not_none]
            ner_prob_real_word = ner_prob_one_batch[index_not_none]
            _, indices = torch.unique(word_ids_one_batch_not_none, return_inverse=True)
            grouped_ner_prob = []
            for i in range(torch.max(indices) + 1):
                grouped_ner_prob.append(ner_prob_real_word
                                                 [indices == i])
            grouped_ner_prob_sum = [torch.sum(v, dim=0)
                                          for v in grouped_ner_prob]
            prob.append(torch.stack(grouped_ner_prob_sum))

        return torch.stack(prob)






    # def get_bert_feature(self, data):
    #     '''
    #     :param data: output of dataloader 'input_ids'[bs,1000], 'input_ids_length'[bs,618]
    #                 'attention_mask'bs.618, 'label_croase' 'label_fine'
    #     :return:
    #     '''
    #     output_bert = self.bert_model(input_ids=data['input_ids'],
    #                                   attention_mask=data['attention_mask_bert'])
    #     last_hidden_state = output_bert['last_hidden_state']
    #     b_s, true_length, _ = last_hidden_state.shape
    #     bert_avg = []
    #     for b_s_index in range(b_s):
    #         last_hidden_state_one_batch = last_hidden_state[b_s_index]
    #         last_hidden_state_real = last_hidden_state_one_batch[
    #                                  :len(data['input_ids_length'][b_s_index])]
    #         temp = []
    #         i = 0
    #         for length in data['input_ids_length'][b_s_index][
    #             data['input_ids_length'][b_s_index].nonzero().squeeze()]:
    #             temp.append(torch.mean(last_hidden_state_real[i:i + length], dim=0))
    #             i += length
    #         stack = torch.stack(temp, dim=0)
    #         padding = (0, 0, 0, self.max_len_words - len(stack))
    #         bert_avg.append(torch.nn.functional.pad(stack, padding))
    #
    #     return torch.stack(bert_avg, dim=0z