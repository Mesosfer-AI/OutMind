"""Streaming autoregressive token generator with KV-cache."""

from typing import Iterator, List, Optional
import torch
import torch.nn.functional as F

from outmind.model import OutMindForCausalLM
from outmind.tokenizer import OutMindTokenizer


def sample_top_p_top_k(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> int:
    """Samples next token ID with temperature scaling, top-k truncation, and nucleus (top-p) filtering."""
    if temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / max(temperature, 1e-5)

    # Top-K filtering
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., [-1]]] = float("-inf")

    # Top-P (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask to keep at least the top-1 candidate
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


@torch.no_grad()
def generate_stream(
    model: OutMindForCausalLM,
    tokenizer: OutMindTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    stop_tokens: Optional[List[int]] = None,
) -> Iterator[str]:
    """Generates tokens autoregressively with KV-cache, yielding decoded text chunks in real-time."""
    model.eval()
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt)
    curr_input = torch.tensor([input_ids], dtype=torch.long, device=device)

    if stop_tokens is None:
        stop_tokens = [tokenizer.eos_id, tokenizer.eot_id]

    past_kvs = None
    start_pos = 0

    for _ in range(max_new_tokens):
        logits, _, past_kvs = model(curr_input, past_kvs=past_kvs, start_pos=start_pos)
        next_token_logits = logits[:, -1, :]

        next_token = sample_top_p_top_k(
            next_token_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        if next_token in stop_tokens:
            break

        yield tokenizer.decode([next_token])

        # Step KV-cache forward with single next-token input
        start_pos += curr_input.shape[1]
        curr_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
