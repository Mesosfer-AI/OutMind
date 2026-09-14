"""Modern Causal Decoder Transformer (RMSNorm, RoPE, SwiGLU, GQA, MoE, KV-Cache)."""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from outmind.attention import scaled_attention
from outmind.config import OutMindConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding supporting high-theta base frequencies and YaRN context extension."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        theta: float = 500000.0,
        scaling_type: str = "default",
        scaling_factor: float = 1.0,
        original_max_seq_len: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        self.original_max_seq_len = original_max_seq_len

        base_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        if scaling_type == "linear" and scaling_factor > 1.0:
            inv_freq = base_freq / scaling_factor
        elif scaling_type == "yarn" and scaling_factor > 1.0:
            inv_freq = self._compute_yarn_freqs(base_freq, dim, scaling_factor, original_max_seq_len)
        else:
            inv_freq = base_freq

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    @staticmethod
    def _compute_yarn_freqs(
        base_freq: torch.Tensor,
        dim: int,
        factor: float,
        orig_max: int,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ) -> torch.Tensor:
        # ponytail: standard YaRN ramp interpolation (Peng et al., 2023)
        wavelength = 2.0 * 3.141592653589793 / base_freq
        low = orig_max / beta_fast
        high = orig_max / beta_slow
        gamma = (wavelength - low) / (high - low)
        gamma = torch.clamp(gamma, 0.0, 1.0)
        return (1.0 - gamma) * base_freq + gamma * (base_freq / factor)

    def _build_cache(self, max_seq_len: int):
        t = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        end_pos = start_pos + seq_len
        # Dynamic cache expansion: double capacity if input sequence exceeds current buffer
        if end_pos > self.cos_cached.shape[0]:
            self._build_cache(max(end_pos, self.cos_cached.shape[0] * 2))

        cos = self.cos_cached[start_pos:end_pos, :].to(q.dtype).unsqueeze(0).unsqueeze(1)
        sin = self.sin_cached[start_pos:end_pos, :].to(q.dtype).unsqueeze(0).unsqueeze(1)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: OutMindConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout_p

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = rope(q, k, start_pos=start_pos)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v)
        out = scaled_attention(q, k, v, mask=mask, is_causal=True, dropout_p=self.dropout_p)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out), present_kv


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class SparseMoE(nn.Module):
    def __init__(self, config: OutMindConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.use_shared_expert = config.use_shared_expert

        self.router = nn.Linear(config.d_model, config.num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(config.d_model, config.d_ffn) for _ in range(config.num_experts)])

        if self.use_shared_expert:
            shared_dim = config.shared_expert_ffn_dim or config.d_ffn
            self.shared_expert = SwiGLU(config.d_model, shared_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)

        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                selected_x = x_flat[mask]
                expert_out = expert(selected_x)
                # Compute effective routing weight per selected token
                weights_for_expert = (topk_weights * (topk_indices == i)).sum(dim=-1, keepdim=True)[mask]
                out[mask] += expert_out * weights_for_expert

        if self.use_shared_expert:
            out = out + self.shared_expert(x_flat)

        return out.view(batch_size, seq_len, d_model)


class TransformerBlock(nn.Module):
    def __init__(self, config: OutMindConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)

        if config.use_moe:
            self.ffn = SparseMoE(config)
        else:
            self.ffn = SwiGLU(config.d_model, config.d_ffn)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, present_kv = self.attn(self.attn_norm(x), rope, past_kv=past_kv, start_pos=start_pos, mask=mask)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, present_kv


class OutMindForCausalLM(nn.Module):
    def __init__(self, config: OutMindConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
            scaling_type=config.rope_type,
            scaling_factor=config.rope_scaling_factor,
            original_max_seq_len=config.original_max_seq_len,
        )
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        _, seq_len = input_ids.shape
        x = self.tok_embeddings(input_ids)

        present_kvs = []
        for i, layer in enumerate(self.layers):
            layer_past_kv = past_kvs[i] if past_kvs is not None else None
            x, present_kv = layer(x, self.rope, past_kv=layer_past_kv, start_pos=start_pos)
            present_kvs.append(present_kv)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Autoregressive causal language modeling loss: predict token t+1 from position t
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss, present_kvs
