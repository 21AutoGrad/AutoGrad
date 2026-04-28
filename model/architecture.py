"""
Model architecture — direct port of the classes defined in pseudoscore-x.ipynb.

Keep this file byte-compatible with the notebook, otherwise the saved
`best_model_v5.pt` checkpoint will fail to load (state_dict key mismatches).
"""
import math
import torch
import torch.nn as nn


# ────────────────────────────────────────────────────────────────────────────
# 1. Attention pooling
# ────────────────────────────────────────────────────────────────────────────
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, hidden, mask=None):
        scores = self.attn(hidden).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * hidden).sum(dim=1)


# ────────────────────────────────────────────────────────────────────────────
# 2. Post-attention feed-forward block
# ────────────────────────────────────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.net(x))


# ────────────────────────────────────────────────────────────────────────────
# 3. Cross-attention + FFN block
# ────────────────────────────────────────────────────────────────────────────
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, Q_tokens, KV_tokens, KV_mask=None):
        B, m, d = Q_tokens.shape
        _, n, _ = KV_tokens.shape

        q = self.query(Q_tokens).view(B, m, self.n_heads, self.d_head).transpose(1, 2)
        k = self.key(KV_tokens).view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        v = self.value(KV_tokens).view(B, n, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if KV_mask is not None:
            mask_expanded = (KV_mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, m, d)
        out = self.dropout(self.out(out))

        x = self.norm(Q_tokens + out)
        x = self.ffn(x)
        return x, attn.detach()


# ────────────────────────────────────────────────────────────────────────────
# 4. Stacked cross-attention
# ────────────────────────────────────────────────────────────────────────────
class StackedCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, Q_tokens, KV_tokens, KV_mask=None):
        out = Q_tokens
        all_attn = []
        for layer in self.layers:
            out, attn = layer(out, KV_tokens, KV_mask)
            all_attn.append(attn)
        return out, all_attn


# ────────────────────────────────────────────────────────────────────────────
# 5. MLP scoring head
# ────────────────────────────────────────────────────────────────────────────
class MLP_Head(nn.Module):
    def __init__(self, input_dim, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ────────────────────────────────────────────────────────────────────────────
# 6. Full model
# ────────────────────────────────────────────────────────────────────────────
class CriterionWiseScoringSystem(nn.Module):
    """
    Three information paths:
      1. Q → C → A : question conditioned on criterion attends to answer
      2. A → C     : answer directly attends to criterion
      3. A_direct  : raw answer pooling

    Scoring head input: [qa_vec | ac_vec | a_vec] (3 * embedding_dim).
    Output: normalised score in [0, 1]; multiply by max_score to get marks.
    """

    def __init__(self, embedding_dim=1024, n_heads=8,
                 n_cross_layers=2, dropout=0.1):
        super().__init__()

        # QC stream is 1 layer (matches notebook)
        self.cross_attn_qc = StackedCrossAttention(
            embedding_dim, n_heads, n_layers=1, dropout=dropout)
        self.cross_attn_qa = StackedCrossAttention(
            embedding_dim, n_heads, n_layers=n_cross_layers, dropout=dropout)
        self.cross_attn_ac = StackedCrossAttention(
            embedding_dim, n_heads, n_layers=n_cross_layers, dropout=dropout)

        self.pool_qa = AttentionPooling(embedding_dim)
        self.pool_ac = AttentionPooling(embedding_dim)
        self.pool_a = AttentionPooling(embedding_dim)

        self.scoring_head = MLP_Head(
            input_dim=embedding_dim * 3, dropout=0.15)

    def extract_features(self, Q_tokens, A_tokens, c_tokens,
                         A_mask=None, c_mask=None, return_attn=False):
        # Path 1: Q -> C -> A
        Q_cond, qc_attn = self.cross_attn_qc(Q_tokens, c_tokens, c_mask)
        attended_qa, qa_attn = self.cross_attn_qa(Q_cond, A_tokens, A_mask)
        qa_vec = self.pool_qa(attended_qa, A_mask)

        # Path 2: A -> C
        attended_ac, ac_attn = self.cross_attn_ac(A_tokens, c_tokens, c_mask)
        ac_vec = self.pool_ac(attended_ac, A_mask)

        # Path 3: raw answer
        a_vec = self.pool_a(A_tokens, A_mask)

        features = torch.cat([qa_vec, ac_vec, a_vec], dim=-1)

        if return_attn:
            attn_dict = {
                "qc": qc_attn[-1],   # (B, heads, q_len, c_len)
                "qa": qa_attn[-1],   # (B, heads, q_len, a_len)
                "ac": ac_attn[-1],   # (B, heads, a_len, c_len)
            }
            return features, attn_dict

        return features

    def forward(self, Q_tokens, A_tokens, c_tokens,
                A_mask=None, c_mask=None, return_attn=False):
        if return_attn:
            features, attn_dict = self.extract_features(
                Q_tokens, A_tokens, c_tokens,
                A_mask=A_mask, c_mask=c_mask, return_attn=True,
            )
            score = self.scoring_head(features)
            return score, attn_dict

        features = self.extract_features(
            Q_tokens, A_tokens, c_tokens,
            A_mask=A_mask, c_mask=c_mask, return_attn=False,
        )
        score = self.scoring_head(features)
        return score
