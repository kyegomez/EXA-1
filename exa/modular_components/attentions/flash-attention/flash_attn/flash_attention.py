import math
import torch
import torch.nn as nn

from einops import rearrange

from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                        device=qkv.device)
                output = flash_attn_unpadded_qkvpacked_func(
                    qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                output_unpad = flash_attn_unpadded_qkvpacked_func(
                    x_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                            indices, batch_size, seqlen),
                                'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            output = flash_attn_unpadded_qkvpacked_func(
                qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )

        return output, None


class FlashMHA(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, key_padding_mask=None, need_weights=False):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        context, attn_weights = self.inner_attn(qkv, key_padding_mask=key_padding_mask,
                                                need_weights=need_weights, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights


"""
The FlashAttention class provided in the code implements the Flash Attention mechanism, which is a faster and more memory-efficient alternative to the standard Scaled Dot-Product Attention. Here's a step-by-step breakdown of the forward function:

The input tensor qkv has shape (B, S, 3, H, D), where B is the batch size, S is the sequence length, H is the number of attention heads, and D is the head dimension. qkv contains the query, key, and value tensors.

If cu_seqlens is None, the implementation computes the cumulative sequence lengths cu_seqlens and reshapes the input tensor as required for the Flash Attention function.

If key_padding_mask is not None, the input tensor is first unpadded using the unpad_input function, then reshaped accordingly. After applying the Flash Attention function, the output tensor is padded back using the pad_input function.

The Flash Attention function (flash_attn_unpadded_qkvpacked_func) is called with the processed input tensor, cumulative sequence lengths, dropout rate (if in training mode), softmax scaling factor, and a flag indicating whether causal attention is required.

The output tensor of the Flash Attention function is reshaped as needed, and the attention weights are returned as None since the need_weights flag is set to False.

The FlashMHA class integrates the FlashAttention mechanism into a multi-head attention (MHA) module. The forward function in FlashMHA consists of the following steps:

The input tensor x is passed through the Wqkv linear layer, which computes the query, key, and value tensors. The output tensor is reshaped using the rearrange function to match the expected shape for the Flash Attention module.

The reshaped tensor is passed through the inner_attn layer, which is an instance of the FlashAttention class. The need_weights flag is set to False, indicating that the attention weights are not needed in the output.

The output tensor from the inner_attn layer is reshaped again using the rearrange function to match the expected shape for the out_proj linear layer.

The reshaped tensor is passed through the out_proj linear layer, which returns the final output tensor. The attention weights are returned as None since the need_weights flag is set to False.

"""