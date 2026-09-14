"""Unified AdamW trainer with gradient accumulation, AMP, and cosine scheduler."""

from dataclasses import dataclass
import math
import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainerConfig:
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 100
    max_steps: int = 1000
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    save_every: int = 500
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"
    use_amp: bool = True


class Trainer:
    def __init__(self, model: nn.Module, config: TrainerConfig):
        self.model = model
        self.config = config

        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = config.device

        self.model.to(self.device)
        self.optimizer = self._create_optimizer()
        self.scaler = torch.amp.GradScaler("cuda") if (self.device == "cuda" and config.use_amp) else None
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def _create_optimizer(self) -> torch.optim.AdamW:
        """Filters weight decay: applied only to 2D transformation weights, not 1D gains or biases."""
        decay_params = []
        nodecay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            optim_groups,
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
        )

    def _get_lr(self, step: int) -> float:
        """Cosine annealing with linear warmup."""
        if step < self.config.warmup_steps:
            return self.config.lr * (step + 1) / (self.config.warmup_steps + 1)
        if step > self.config.max_steps:
            return self.config.min_lr

        decay_ratio = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.lr - self.config.min_lr)

    def save_checkpoint(self, step: int, path: Optional[str] = None):
        target = path or os.path.join(self.config.checkpoint_dir, f"checkpoint_step_{step}.pt")
        state = {
            "step": step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(state, target)

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        return state.get("step", 0)

    def train(self, dataloader: DataLoader):
        self.model.train()
        step = 0
        running_loss = 0.0
        pbar = tqdm(total=self.config.max_steps, desc="Training")

        data_iter = iter(dataloader)
        self.optimizer.zero_grad(set_to_none=True)

        use_amp = self.config.use_amp and (self.device == "cuda")
        amp_dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

        while step < self.config.max_steps:
            accum_loss = 0.0

            for _ in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        _, loss, _ = self.model(input_ids, labels=labels)
                        loss = loss / self.config.gradient_accumulation_steps
                else:
                    _, loss, _ = self.model(input_ids, labels=labels)
                    loss = loss / self.config.gradient_accumulation_steps

                accum_loss += loss.item()

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            # Gradient clipping and optimizer step
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)

            # Update learning rate
            step += 1
            lr = self._get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            running_loss = 0.9 * running_loss + 0.1 * accum_loss if running_loss > 0 else accum_loss
            pbar.set_postfix({"loss": f"{running_loss:.4f}", "lr": f"{lr:.2e}"})
            pbar.update(1)

            if step % self.config.save_every == 0 or step == self.config.max_steps:
                self.save_checkpoint(step)

        pbar.close()
