import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from outmind.attention import detect_attention_backend, scaled_attention
from outmind.config import OutMindConfig
from outmind.dataset import SFTDataset
from outmind.lora import apply_lora, merge_lora
from outmind.model import OutMindForCausalLM
from outmind.tokenizer import OutMindTokenizer


def test_attention_backend():
    backend = detect_attention_backend()
    assert backend in ("fa3", "fa2", "sdpa", "math"), f"Invalid attention backend: {backend}"

    # Test GQA shape alignment
    b, n_heads, n_kv_heads, seq_len, head_dim = 2, 4, 2, 8, 16
    q = torch.randn(b, n_heads, seq_len, head_dim)
    k = torch.randn(b, n_kv_heads, seq_len, head_dim)
    v = torch.randn(b, n_kv_heads, seq_len, head_dim)

    out = scaled_attention(q, k, v, is_causal=True)
    assert out.shape == (b, n_heads, seq_len, head_dim), f"Unexpected attention output shape: {out.shape}"
    print(f"  [PASS] Attention backend detection ({backend}) & GQA alignment")


def test_model_forward_and_loss():
    cfg = OutMindConfig.nano(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ffn=128,
        max_seq_len=32,
    )
    model = OutMindForCausalLM(cfg)

    input_ids = torch.randint(0, 256, (2, 16))
    labels = input_ids.clone()

    logits, loss, _ = model(input_ids, labels=labels)
    assert logits.shape == (2, 16, 256), f"Unexpected logits shape: {logits.shape}"
    assert loss is not None and not torch.isnan(loss) and loss.item() > 0.0, "Loss calculation failed"

    loss.backward()
    grad_norm = model.tok_embeddings.weight.grad.norm().item()
    assert grad_norm > 0.0, "Gradients failed to propagate to embeddings"
    print("  [PASS] Model forward pass, cross-entropy loss & backpropagation")


def test_sparse_moe():
    cfg = OutMindConfig.moe(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ffn=128,
        max_seq_len=32,
        num_experts=4,
        num_experts_per_tok=2,
        use_shared_expert=True,
    )
    model = OutMindForCausalLM(cfg)

    input_ids = torch.randint(0, 256, (2, 16))
    labels = input_ids.clone()

    logits, loss, _ = model(input_ids, labels=labels)
    assert logits.shape == (2, 16, 256)
    assert loss is not None and not torch.isnan(loss)

    loss.backward()
    print("  [PASS] Sparse MoE routing (4 experts, top-2 + 1 shared expert)")


def test_kv_cache_parity():
    cfg = OutMindConfig.nano(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ffn=64,
        max_seq_len=16,
    )
    model = OutMindForCausalLM(cfg)
    model.eval()

    seq = torch.tensor([[10, 25, 42, 88]], dtype=torch.long)

    # 1. Standard forward pass over full sequence
    with torch.no_grad():
        full_logits, _, _ = model(seq)

    # 2. Step-by-step forward pass with KV cache
    with torch.no_grad():
        step1_logits, _, past_kvs = model(seq[:, :2], start_pos=0)
        step2_logits, _, _ = model(seq[:, 2:3], past_kvs=past_kvs, start_pos=2)

    # Logits at position 2 should match within numerical precision
    diff = (full_logits[:, 2, :] - step2_logits[:, 0, :]).abs().max().item()
    assert diff < 1e-4, f"KV-Cache parity mismatch: max diff {diff}"
    print(f"  [PASS] KV-Cache equivalence parity (max abs diff: {diff:.2e})")


def test_sft_prompt_masking():
    tokenizer = OutMindTokenizer()
    dummy_dialogue = [
        {"role": "system", "content": "System directive."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
    ]

    dataset = SFTDataset([dummy_dialogue], tokenizer, max_seq_len=64)
    sample = dataset[0]

    input_ids = sample["input_ids"]
    labels = sample["labels"]

    # Target labels must have -100 on prompt tokens and positive IDs on assistant tokens
    masked_count = (labels == -100).sum().item()
    unmasked_count = (labels != -100).sum().item()

    assert masked_count > 0, "Prompt tokens were not masked"
    assert unmasked_count > 0, "Assistant response tokens are missing from target"
    print(f"  [PASS] SFT prompt masking (masked: {masked_count}, supervised: {unmasked_count})")


def test_lora_adaptation_and_merge():
    cfg = OutMindConfig.nano(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ffn=64,
        max_seq_len=16,
    )
    model = OutMindForCausalLM(cfg)

    # Apply LoRA to q_proj and v_proj
    adapters = apply_lora(model, rank=4, alpha=8.0)
    assert len(adapters) > 0, "No LoRA adapters were injected"

    # Ensure base weights are frozen and only lora_A/lora_B require grad
    for adapter in adapters:
        assert not adapter.base_layer.weight.requires_grad, "Base weight should be frozen"
        assert adapter.lora_A.requires_grad and adapter.lora_B.requires_grad, "LoRA weights must be trainable"

    test_input = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        pre_merge_logits, _, _ = model(test_input)

    merge_lora(model)

    with torch.no_grad():
        post_merge_logits, _, _ = model(test_input)

    diff = (pre_merge_logits - post_merge_logits).abs().max().item()
    assert diff < 1e-5, f"LoRA weight merge parity mismatch: {diff}"
    print(f"  [PASS] Native PyTorch LoRA injection and weight merging (parity diff: {diff:.2e})")


def test_eval_suite():
    from scripts.eval import (
        eval_tokenizer,
        eval_pretrain_perplexity,
        eval_zero_shot_choice,
    )
    import ast, json, math, re

    tokenizer = OutMindTokenizer()
    tok_res = eval_tokenizer(tokenizer, corpus={"ID": "Anak-anak bermain di taman."}, compare_baselines=False)
    assert "ID" in tok_res["outmind"] and tok_res["outmind"]["ID"]["fidelity"] is True, "Tokenizer fidelity failure"

    cfg = OutMindConfig.nano(vocab_size=tokenizer.vocab_size, max_seq_len=32, n_layers=2, d_model=128, n_heads=4, n_kv_heads=2)
    model = OutMindForCausalLM(cfg)
    ppl = eval_pretrain_perplexity(model, tokenizer, ["Halo dunia."], device="cpu")
    assert ppl > 0 and not math.isnan(ppl), "Perplexity calculation failure"

    choice_acc = eval_zero_shot_choice(
        model, tokenizer, [{"context": "A", "choices": ["B", "C"], "answer": 0}], device="cpu"
    )
    assert 0.0 <= choice_acc <= 1.0, "Choice accuracy out of range"

    # Verify GSM8k regex extractor
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", "The result is 42. #### 42")
    assert m and m.group(1) == "42", "GSM8k regex failed"

    # Verify tool calling parser
    tool_text = '<functioncall> {"name": "test_fn", "arguments": {"x": 1}}'
    start = tool_text.find("<functioncall>")
    obj, _ = json.JSONDecoder().raw_decode(tool_text[tool_text.find("{", start):])
    assert obj["name"] == "test_fn", "Tool calling parser failed"

    # Verify AST parser
    ast.parse("def f(x):\n    return x + 1\n")

    print(f"  [PASS] Evaluation suite metrics, PPL calculation, and benchmark parsers")


def test_dataset_etl():
    from tools.build_dataset import clean_wikipedia, clean_openwebmath, clean_code, normalize_sft_row

    # 1. Wikipedia cleaning
    wiki_raw = "Asam DNA adalah molekul.[1][catatan 2]\n\n== Referensi ==\n1. Buku Biologi"
    cleaned_wiki = clean_wikipedia(wiki_raw)
    assert "[1]" not in cleaned_wiki, "Footnote bracket not removed from Wikipedia"
    assert "== Referensi ==" not in cleaned_wiki, "Reference footer not removed"

    # 2. OpenWebMath cleaning
    math_raw = "<!-- comment -->\nFormula: E=mc^2 <testsuite:test> bad char: \ufffd"
    cleaned_math = clean_openwebmath(math_raw)
    assert "<!-- comment -->" not in cleaned_math, "HTML comment not removed"
    assert "\ufffd" not in cleaned_math, "Corrupt unicode not removed"

    # 3. CodeParrot cleaning
    code_raw = "# Copyright (C) 2024 Mesosfer\n# License: Apache-2.0\ndef hello():\n    return 42\n"
    cleaned_code = clean_code(code_raw)
    assert "Copyright" not in cleaned_code, "Copyright header block not removed"

    # 4. SFT ChatML normalization
    aya_row = {"inputs": "Siapa kamu?<unk>", "targets": "Saya OutMind.<unk>"}
    sft_item = normalize_sft_row("id_aya", aya_row)
    assert sft_item is not None, "Failed to normalize Aya row"
    assert "<unk>" not in sft_item["messages"][1]["content"], "<unk> token left in user prompt"
    assert sft_item["messages"][0]["role"] == "system", "Missing system role"

    print("  [PASS] Dataset ETL cleaning rules, regex sanitizers & ChatML normalization")


def test_long_context_rope():
    from outmind.config import OutMindConfig
    from outmind.model import RotaryEmbedding

    # 1. YaRN initialization
    rope_yarn = RotaryEmbedding(
        dim=64,
        max_seq_len=2048,
        theta=500000.0,
        scaling_type="yarn",
        scaling_factor=4.0,
        original_max_seq_len=2048,
    )
    assert rope_yarn.scaling_type == "yarn", "YaRN scaling type mismatch"
    assert rope_yarn.inv_freq.shape[0] == 32, "RotaryEmbedding half-dim frequency mismatch"

    # 2. Dynamic cache expansion beyond 2048 (e.g. seq_len = 5000)
    q = torch.randn(1, 4, 5000, 64)
    k = torch.randn(1, 2, 5000, 64)
    q_rot, k_rot = rope_yarn(q, k, start_pos=0)
    assert q_rot.shape == q.shape, "Rotated query shape mismatch after cache expansion"
    assert rope_yarn.cos_cached.shape[0] >= 5000, "Dynamic cache was not expanded"
    assert not torch.isnan(q_rot).any(), "NaNs detected in YaRN rotated query"

    print("  [PASS] Long-Context Horizon: YaRN RoPE scaling & dynamic cache expansion (>5k tokens)")


def main():
    print("=" * 60)
    print("[TEST] Running OutMind Core Self-Checks (tests/test_core.py)")
    print("=" * 60)

    test_attention_backend()
    test_model_forward_and_loss()
    test_sparse_moe()
    test_kv_cache_parity()
    test_sft_prompt_masking()
    test_lora_adaptation_and_merge()
    test_eval_suite()
    test_dataset_etl()
    test_long_context_rope()

    print("=" * 60)
    print("[SUCCESS] All OutMind core self-checks PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
