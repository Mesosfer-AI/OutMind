"""Pretraining and SFT datasets with prompt loss masking."""

from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset

from outmind.tokenizer import OutMindTokenizer


class PretrainDataset(Dataset):
    """Chunks tokenized text sequences into fixed context lengths for autoregressive pretraining."""

    def __init__(self, token_ids: List[int], max_seq_len: int = 2048):
        self.max_seq_len = max_seq_len
        # Slice into chunks of length max_seq_len + 1 (for input_ids and shifted labels)
        total_len = len(token_ids)
        chunk_size = max_seq_len + 1
        num_chunks = total_len // chunk_size

        self.samples = [
            token_ids[i * chunk_size : (i + 1) * chunk_size]
            for i in range(num_chunks)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.samples[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


class SFTDataset(Dataset):
    """Supervised Fine-Tuning dataset with dynamic loss masking (prompt tokens ignored with -100)."""

    def __init__(
        self,
        dialogues: List[List[Dict[str, str]]],
        tokenizer: OutMindTokenizer,
        max_seq_len: int = 2048,
    ):
        self.samples = []
        for conversation in dialogues:
            sample = self._format_and_mask(conversation, tokenizer, max_seq_len)
            if sample is not None:
                self.samples.append(sample)

    def _format_and_mask(
        self,
        conversation: List[Dict[str, str]],
        tokenizer: OutMindTokenizer,
        max_seq_len: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        input_ids: List[int] = [tokenizer.bos_id]
        labels: List[int] = [-100]

        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            header = f"{tokenizer.start_header_token}{role}{tokenizer.end_header_token}\n"
            header_ids = tokenizer.encode(header)
            content_ids = tokenizer.encode(f"{content}{tokenizer.eot_token}\n")

            input_ids.extend(header_ids)
            labels.extend([-100] * len(header_ids))

            input_ids.extend(content_ids)
            if role == "assistant":
                # Compute loss exclusively on assistant response tokens
                labels.extend(content_ids)
            else:
                labels.extend([-100] * len(content_ids))

        if len(input_ids) < 2:
            return None

        # Truncate to maximum sequence length
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

        # Pad to max_seq_len for batch consistency
        pad_len = max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids.extend([tokenizer.pad_id] * pad_len)
            labels.extend([-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
