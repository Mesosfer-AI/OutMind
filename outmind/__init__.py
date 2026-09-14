"""OutMind: Minimalist Open-Source SLM Framework."""

from outmind.attention import detect_attention_backend, scaled_attention
from outmind.config import OutMindConfig
from outmind.dataset import PretrainDataset, SFTDataset
from outmind.generate import generate_stream
from outmind.lora import LinearWithLoRA, apply_lora, merge_lora
from outmind.model import OutMindForCausalLM
from outmind.tokenizer import OutMindTokenizer
from outmind.trainer import Trainer, TrainerConfig

__version__ = "0.1.0"

__all__ = [
    "OutMindConfig",
    "OutMindForCausalLM",
    "OutMindTokenizer",
    "Trainer",
    "TrainerConfig",
    "PretrainDataset",
    "SFTDataset",
    "LinearWithLoRA",
    "apply_lora",
    "merge_lora",
    "generate_stream",
    "detect_attention_backend",
    "scaled_attention",
]
