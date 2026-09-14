"""Model configuration dataclasses and presets for OutMind."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OutMindConfig:
    vocab_size: int = 64000
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    d_ffn: int = 1024
    max_seq_len: int = 2048
    rope_theta: float = 500000.0
    rope_type: str = "default"  # "default", "linear", or "yarn"
    rope_scaling_factor: float = 1.0
    original_max_seq_len: int = 2048
    rms_norm_eps: float = 1e-6
    dropout_p: float = 0.0
    tie_word_embeddings: bool = True
    
    # Sparse Mixture-of-Experts (MoE) settings
    use_moe: bool = False
    num_experts: int = 4
    num_experts_per_tok: int = 2
    use_shared_expert: bool = True
    shared_expert_ffn_dim: Optional[int] = None

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        if self.shared_expert_ffn_dim is None:
            self.shared_expert_ffn_dim = self.d_ffn

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def nano(cls, **kwargs) -> "OutMindConfig":
        """Ultra-lightweight preset (~34M parameters) ideal for laptops and CPUs."""
        cfg = cls(
            vocab_size=64000,
            d_model=384,
            n_layers=6,
            n_heads=6,
            n_kv_heads=2,
            d_ffn=1024,
            max_seq_len=2048,
            use_moe=False,
            tie_word_embeddings=True,
        )
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def small(cls, **kwargs) -> "OutMindConfig":
        """Standard SLM preset (~125M parameters) for single consumer GPUs."""
        cfg = cls(
            vocab_size=64000,
            d_model=768,
            n_layers=12,
            n_heads=12,
            n_kv_heads=4,
            d_ffn=2048,
            max_seq_len=2048,
            use_moe=False,
            tie_word_embeddings=True,
        )
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def moe(cls, **kwargs) -> "OutMindConfig":
        """Sparse MoE preset (~122M total / ~49M active parameters)."""
        cfg = cls(
            vocab_size=64000,
            d_model=512,
            n_layers=8,
            n_heads=8,
            n_kv_heads=2,
            d_ffn=1365,
            max_seq_len=2048,
            use_moe=True,
            num_experts=4,
            num_experts_per_tok=2,
            use_shared_expert=True,
            tie_word_embeddings=True,
        )
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg
