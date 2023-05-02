import torch
import torch.nn as nn
from torch.nn import LayerNorm
from einops import rearrange

class KernelAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, kernel="gaussian", sigma=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.sigma = sigma
        self.kernel = kernel

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.dropout_module = torch.nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        bsz, tgt_len, embed_dim = q.size()

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q *= self.scaling

        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)

        if self.kernel == "gaussian":
            kernel_attn = torch.exp(-((q.unsqueeze(3) - k.unsqueeze(2)) ** 2).sum(-1) / (2 * self.sigma ** 2))
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")

        kernel_attn = kernel_attn / kernel_attn.sum(dim=-1, keepdim=True)

        if attn_mask is not None:
            kernel_attn = kernel_attn * attn_mask.unsqueeze(1)

        attn_probs = self.dropout_module(kernel_attn)
        attn = torch.einsum('b h t s, b h s d -> b h t d', attn_probs, v)

        attn = rearrange(attn, 'b h t d -> b t (h d)', h=self.num_heads)
        attn = self.out_proj(attn)

        return attn, kernel_attn

class OptimizedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()

        self.kernel_attn = KernelAttention(embed_dim, num_heads, dropout)
        self.layer_norm = LayerNorm(embed_dim)

    def forward(self, q, k, v, attn_mask=None):
        attn, attn_weights = self.kernel_attn(q, k, v, attn_mask)
        attn = self.layer_norm(attn)
        attn_weights = attn_weights.to(torch.float16)

        return attn, attn_weights
