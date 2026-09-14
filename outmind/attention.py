"""4-Tier Cascaded Attention Engine (FlashAttention-3 -> FlashAttention-2 -> SDPA -> Eager Math)."""

import math
from typing import Optional
import torch
import torch.nn.functional as F

_ATTN_BACKEND: Optional[str] = None


def detect_attention_backend() -> str:
    global _ATTN_BACKEND
    if _ATTN_BACKEND is not None:
        return _ATTN_BACKEND

    if torch.cuda.is_available():
        try:
            import flash_attn_interface  # noqa: F401
            _ATTN_BACKEND = "fa3"
            return _ATTN_BACKEND
        except ImportError:
            pass

        try:
            import flash_attn  # noqa: F401
            _ATTN_BACKEND = "fa2"
            return _ATTN_BACKEND
        except ImportError:
            pass

    if hasattr(F, "scaled_dot_product_attention"):
        _ATTN_BACKEND = "sdpa"
    else:
        _ATTN_BACKEND = "math"

    return _ATTN_BACKEND


def scaled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    is_causal: bool = True,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Computes attention with automated backend selection and GQA head alignment.

    Expected input shapes:
        q: (batch_size, n_heads, seq_len_q, head_dim)
        k: (batch_size, n_kv_heads, seq_len_k, head_dim)
        v: (batch_size, n_kv_heads, seq_len_k, head_dim)
    """
    batch_size, n_heads, seq_len_q, head_dim = q.shape
    _, n_kv_heads, seq_len_k, _ = k.shape

    # Expand key and value heads if Grouped-Query Attention (GQA) is active
    if n_heads != n_kv_heads:
        repeat_factor = n_heads // n_kv_heads
        k = torch.repeat_interleave(k, repeats=repeat_factor, dim=1)
        v = torch.repeat_interleave(v, repeats=repeat_factor, dim=1)

    backend = detect_attention_backend()

    # FlashAttention-2 / FlashAttention-3 path (requires CUDA and (B, T, H, D) layout)
    if backend in ("fa2", "fa3") and q.is_cuda and mask is None and seq_len_q == seq_len_k:
        try:
            from flash_attn import flash_attn_func
            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
            out = flash_attn_func(q_fa, k_fa, v_fa, dropout_p=dropout_p, causal=is_causal)
            return out.transpose(1, 2)
        except Exception:
            # Fall back to PyTorch native SDPA if FlashAttention kernel fails
            pass

    # Native PyTorch SDPA (C++ FlashAttention / xFormers / CPU vectorized kernel)
    if hasattr(F, "scaled_dot_product_attention"):
        causal_flag = is_causal if (mask is None and seq_len_q == seq_len_k) else False
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=causal_flag
        )

    # Pure Eager Math Fallback
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    if is_causal and seq_len_q == seq_len_k:
        causal_mask = torch.tril(torch.ones((seq_len_q, seq_len_k), device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    elif mask is not None:
        scores = scores + mask

    attn_probs = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn_probs = F.dropout(attn_probs, p=dropout_p)

    return torch.matmul(attn_probs, v)
