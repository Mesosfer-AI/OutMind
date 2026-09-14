"""Native PyTorch LoRA (Low-Rank Adaptation) and zero-latency weight merging."""

import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearWithLoRA(nn.Module):
    """Wraps an nn.Linear module with trainable low-rank decomposition matrices A and B."""

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.merged = False

        # Freeze base parameters
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        in_dim = base_layer.in_features
        out_dim = base_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))

        # Initialize A uniformly, B to zero so initial delta is strictly zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)
        base_out = self.base_layer(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

    def merge_weights(self):
        """Merges adapter delta into the base layer weight for zero-latency inference."""
        if self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base_layer.weight.data += delta
        self.merged = True


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj"),
) -> List[LinearWithLoRA]:
    """Replaces targeted linear layers with LinearWithLoRA adapters."""
    adapters = []
    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            if any(target in child_name for target in target_modules) and isinstance(child_module, nn.Linear):
                adapter = LinearWithLoRA(child_module, rank=rank, alpha=alpha)
                setattr(module, child_name, adapter)
                adapters.append(adapter)
    return adapters


def merge_lora(model: nn.Module):
    """In-place merges all LinearWithLoRA weights back into the base model."""
    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            module.merge_weights()
