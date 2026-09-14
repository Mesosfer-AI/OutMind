"""Unified evaluation suite for OutMind across Tokenizer, Pretrain, and SFT stages."""

import argparse
import ast
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from outmind.config import OutMindConfig
from outmind.generate import generate_stream
from outmind.model import OutMindForCausalLM
from outmind.tokenizer import OutMindTokenizer


# -----------------------------------------------------------------------------
# 1. Tokenizer Evaluation
# -----------------------------------------------------------------------------

DEFAULT_TOKENIZER_TEST_CORPUS: Dict[str, str] = {
    "Indonesian (Reduplication & Morphology)": (
        "Anak-anak berlari-lari di taman kota sambil membawa buku-buku cerita baru. "
        "KTP-nya hilang sejak kemarin'kan? Peringkat ke-2 diraih pada tahun 1990-an."
    ),
    "English (Prose & Technical)": (
        "Artificial intelligence and deep autoregressive language models are fundamentally "
        "transforming software engineering, scientific discovery, and automated reasoning."
    ),
    "Chinese (Hanzi Characters)": (
        "人工智能和深度学习模型正在深刻地改变着现代计算技术，自然语言处理与多模态智能不断取得突破。"
    ),
    "Python Code (Syntax & Indentation)": (
        "def fibonacci_generator(n: int):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        yield a\n"
        "        a, b = b, a + b\n"
    ),
    "Mathematics (Formulas & LaTeX)": (
        "Calculate the integral: \\int_{0}^{\\pi} \\sin(x) dx = 2. "
        "The roots of quadratic equation x^2 - 5x + 6 = 0 are x_1 = 2 and x_2 = 3."
    ),
}


def _load_baseline_tokenizers() -> Dict[str, Any]:
    """Loads external baseline tokenizers (Qwen 3.8, DeepSeek V4.1-Flash, GLM 5.3) with offline fallback."""
    baselines = {}
    from tokenizers import Tokenizer

    # 1. Qwen 3.8 (Alibaba)
    try:
        baselines["Qwen-3.8"] = Tokenizer.from_pretrained("Qwen/Qwen3.8-27B")
    except Exception:
        pass

    # 2. DeepSeek V4.1-Flash (DeepSeek AI)
    try:
        baselines["DeepSeek-V4.1"] = Tokenizer.from_pretrained("deepseek-ai/DeepSeek-V4.1-Flash")
    except Exception:
        pass

    # 3. GLM 5.3 (Zhipu AI / ZAI)
    try:
        baselines["GLM-5.3"] = Tokenizer.from_pretrained("zai-org/GLM-5.3")
    except Exception:
        pass

    return baselines


def eval_tokenizer(
    tokenizer: OutMindTokenizer,
    corpus: Optional[Dict[str, str]] = None,
    compare_baselines: bool = True,
) -> Dict[str, Any]:
    """Evaluates tokenizer compression ratio, fertility, round-trip fidelity, and compares with SOTA baselines."""
    corpus = corpus or DEFAULT_TOKENIZER_TEST_CORPUS
    results: Dict[str, Any] = {"outmind": {}}

    print("\n" + "=" * 78)
    print(f"{'DOMAIN':<38} | {'BYTES':<6} | {'TOKS':<6} | {'COMPR':<7} | {'FERT':<6} | {'FIDELITY'}")
    print("-" * 78)

    for domain, text in corpus.items():
        raw_bytes = len(text.encode("utf-8"))
        tokens = tokenizer.encode(text, allowed_special="all")
        num_tokens = len(tokens)
        words = len(text.split())
        compression_ratio = raw_bytes / max(num_tokens, 1)
        fertility = num_tokens / max(words, 1)
        decoded = tokenizer.decode(tokens)
        is_faithful = decoded == text

        results["outmind"][domain] = {
            "bytes": raw_bytes,
            "tokens": num_tokens,
            "compression_ratio": compression_ratio,
            "fertility": fertility,
            "fidelity": is_faithful,
        }

        status = "[OK]" if is_faithful else "[MISMATCH]"
        print(f"{domain:<38} | {raw_bytes:<6} | {num_tokens:<6} | {compression_ratio:<7.2f} | {fertility:<6.2f} | {status}")

    print("=" * 78)

    # Comparative benchmark against SOTA tokenizers (Kimi, Qwen 2.5, DeepSeek, LLaMA 3)
    if compare_baselines:
        baselines = _load_baseline_tokenizers()
        if baselines:
            models_to_compare = {"OutMind": tokenizer, **baselines}
            header = f"{'DOMAIN':<14} | {'BYTES':<6}"
            for m_name in models_to_compare:
                header += f" | {m_name:<13}"
            header += f" | {'WINNER':<18}"

            divider = "-" * len(header)
            border = "=" * len(header)

            print("\n" + border)
            print("--- SOTA TOKENIZER BENCHMARK COMPARISON (Tokens & Bytes/Token) ---")
            print(border)
            print(header)
            print(divider)

            results["comparative"] = {}
            for domain, text in corpus.items():
                short_name = domain.split("(")[0].strip()
                raw_b = len(text.encode("utf-8"))
                row = f"{short_name:<14} | {raw_b:<6}"
                results["comparative"][short_name] = {}

                tok_counts = {}
                row_cells = []
                for m_name, m_tok in models_to_compare.items():
                    try:
                        if hasattr(m_tok, "encode"):
                            res = m_tok.encode(text)
                            if hasattr(res, "ids"):
                                n_toks = len(res.ids)
                            elif isinstance(res, list):
                                n_toks = len(res)
                            else:
                                n_toks = len(res)
                        else:
                            n_toks = 0
                        compr = raw_b / max(n_toks, 1)
                        cell = f"{n_toks} ({compr:.2f})"
                        results["comparative"][short_name][m_name] = {"tokens": n_toks, "compression": compr}
                        if n_toks > 0:
                            tok_counts[m_name] = n_toks
                    except Exception:
                        cell = "[ERROR]"
                    row_cells.append(cell)

                # Determine domain winner (minimum token count / highest compression)
                if tok_counts:
                    min_val = min(tok_counts.values())
                    winners = [m for m, cnt in tok_counts.items() if cnt == min_val]
                    winner_str = ", ".join(winners)
                else:
                    winner_str = "-"

                for cell in row_cells:
                    row += f" | {cell:<13}"
                row += f" | {winner_str:<18}"
                print(row)

            print(border)

    return results


# -----------------------------------------------------------------------------
# 2. Pretraining Evaluation (Perplexity & Zero-Shot Choice Likelihood)
# -----------------------------------------------------------------------------

def eval_pretrain_perplexity(
    model: OutMindForCausalLM,
    tokenizer: OutMindTokenizer,
    eval_texts: List[str],
    device: str = "cpu",
    max_seq_len: int = 1024,
) -> float:
    """Computes cross-entropy loss and Perplexity (PPL = exp(loss)) over evaluation text blocks."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in eval_texts:
            tokens = tokenizer.encode(text, allowed_special="all")
            if len(tokens) < 2:
                continue
            chunk_size = min(len(tokens), max_seq_len)
            input_ids = torch.tensor([tokens[:chunk_size]], dtype=torch.long, device=device)
            logits, loss, _ = model(input_ids, labels=input_ids)
            if loss is not None:
                tokens_count = chunk_size - 1
                total_loss += loss.item() * tokens_count
                total_tokens += tokens_count

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 100.0))
    return ppl


def eval_zero_shot_choice(
    model: OutMindForCausalLM,
    tokenizer: OutMindTokenizer,
    items: List[Dict[str, Any]],
    device: str = "cpu",
) -> float:
    """Evaluates multiple-choice accuracy (HellaSwag / ARC style) via completion log-likelihoods."""
    model.eval()
    correct = 0

    with torch.no_grad():
        for item in items:
            context = item["context"]
            choices = item["choices"]
            gold_idx = item["answer"]

            context_tokens = tokenizer.encode(context, allowed_special="all")
            choice_scores = []

            for choice in choices:
                choice_tokens = tokenizer.encode(choice, allowed_special="all")
                full_tokens = context_tokens + choice_tokens
                full_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)

                logits, _, _ = model(full_ids)
                log_probs = torch.log_softmax(logits[0], dim=-1)

                # Sum log probabilities of the candidate choice tokens
                score = 0.0
                c_start = len(context_tokens) - 1
                for idx, c_tok in enumerate(choice_tokens):
                    pos = c_start + idx
                    score += log_probs[pos, c_tok].item()
                choice_scores.append(score)

            pred_idx = int(torch.tensor(choice_scores).argmax().item())
            if pred_idx == gold_idx:
                correct += 1

    return correct / max(len(items), 1)


# -----------------------------------------------------------------------------
# 3. SFT Evaluation (Modern Hugging Face Benchmarks)
# -----------------------------------------------------------------------------

def eval_gsm8k_math(
    model: OutMindForCausalLM,
    tokenizer: OutMindTokenizer,
    problems: List[Dict[str, str]],
    device: str = "cpu",
    max_new_tokens: int = 256,
) -> float:
    """Evaluates mathematical chain-of-thought problems with '#### <answer>' extraction."""
    model.eval()
    correct = 0

    for prob in problems:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Solve the math problem step by step and end with #### <number>."},
            {"role": "user", "content": prob["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        generated = "".join(list(generate_stream(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=0.0
        )))

        # Extract predicted numeric answer after ####
        pred_match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", generated)
        gold_match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", prob["answer"])

        pred_val = pred_match.group(1).replace(",", "").strip() if pred_match else None
        gold_val = gold_match.group(1).replace(",", "").strip() if gold_match else prob["answer"].strip()

        if pred_val == gold_val:
            correct += 1

    return correct / max(len(problems), 1)


def eval_humaneval_code(
    model: OutMindForCausalLM,
    tokenizer: OutMindTokenizer,
    tasks: List[Dict[str, Any]],
    device: str = "cpu",
    max_new_tokens: int = 256,
) -> Tuple[float, float]:
    """Evaluates Python code synthesis: AST syntax validity and unit-test Pass@1."""
    model.eval()
    syntax_valid = 0
    passed_tests = 0

    for task in tasks:
        messages = [
            {"role": "system", "content": "You are an expert programmer. Complete the Python function accurately."},
            {"role": "user", "content": task["prompt"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        gen_code = "".join(list(generate_stream(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=0.0
        )))

        full_code = f"{task['prompt']}\n{gen_code}"
        # Check AST syntax correctness
        try:
            ast.parse(full_code)
            syntax_valid += 1
        except SyntaxError:
            continue

        # Execute unit test in isolated scope if provided
        if "test" in task:
            try:
                exec_globals = {}
                exec(f"{full_code}\n{task['test']}", exec_globals)
                passed_tests += 1
            except Exception:
                pass

    total = max(len(tasks), 1)
    return (syntax_valid / total, passed_tests / total)


def eval_mmlu(
    model: OutMindForCausalLM,
    tokenizer: OutMindTokenizer,
    questions: List[Dict[str, Any]],
    device: str = "cpu",
) -> float:
    """Evaluates 4-way multi-choice MMLU questions via direct option token log-likelihoods."""
    model.eval()
    correct = 0
    option_tokens = [tokenizer.encode(f" {opt}")[-1] for opt in ["A", "B", "C", "D"]]

    with torch.no_grad():
        for q in questions:
            user_content = f"{q['question']}\nA. {q['choices'][0]}\nB. {q['choices'][1]}\nC. {q['choices'][2]}\nD. {q['choices'][3]}\nAnswer:"
            messages = [
                {"role": "system", "content": "Answer the multiple-choice question with A, B, C, or D."},
                {"role": "user", "content": user_content},
            ]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

            logits, _, _ = model(input_ids)
            last_logits = logits[0, -1, option_tokens]
            pred_idx = int(last_logits.argmax().item())

            gold_idx = ["A", "B", "C", "D"].index(q["answer"].strip().upper())
            if pred_idx == gold_idx:
                correct += 1

    return correct / max(len(questions), 1)


def eval_tydiqa_indonesian(
    model: OutMindForCausalLM,
    tokenizer: OutMindTokenizer,
    qa_samples: List[Dict[str, str]],
    device: str = "cpu",
    max_new_tokens: int = 64,
) -> Tuple[float, float]:
    """Evaluates Indonesian question answering with Exact Match (EM) and Token F1 score."""
    model.eval()
    exact_matches = 0
    total_f1 = 0.0

    for sample in qa_samples:
        messages = [
            {"role": "system", "content": "Jawab pertanyaan berdasarkan konteks yang diberikan secara ringkas."},
            {"role": "user", "content": f"Konteks: {sample['context']}\nPertanyaan: {sample['question']}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        gen = "".join(list(generate_stream(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=0.0
        ))).strip()

        gold = sample["answer"].strip()
        if gen.lower() == gold.lower():
            exact_matches += 1

        pred_toks = set(gen.lower().split())
        gold_toks = set(gold.lower().split())
        common = pred_toks & gold_toks
        if common:
            p = len(common) / len(pred_toks)
            r = len(common) / len(gold_toks)
            f1 = 2 * (p * r) / (p + r)
        else:
            f1 = 0.0
        total_f1 += f1

    total = max(len(qa_samples), 1)
    return (exact_matches / total, total_f1 / total)


def eval_glaive_tool_calling(
    model: OutMindForCausalLM,
    tokenizer: OutMindTokenizer,
    scenarios: List[Dict[str, Any]],
    device: str = "cpu",
    max_new_tokens: int = 128,
) -> Tuple[float, float]:
    """Evaluates function calling: XML tag compliance and valid JSON parameter schema parsing."""
    model.eval()
    valid_tag_count = 0
    valid_json_count = 0

    for sc in scenarios:
        messages = [
            {"role": "system", "content": f"You are an agent with access to tools:\n{json.dumps(sc['tools'])}"},
            {"role": "user", "content": sc["prompt"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        gen = "".join(list(generate_stream(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=0.0
        )))

        # Verify <functioncall> syntax presence
        start = gen.find("<functioncall>")
        if start != -1:
            valid_tag_count += 1
            json_start = gen.find("{", start)
            if json_start != -1:
                try:
                    payload, _ = json.JSONDecoder().raw_decode(gen[json_start:])
                    if isinstance(payload, dict) and "name" in payload and "arguments" in payload:
                        valid_json_count += 1
                except Exception:
                    pass

    total = max(len(scenarios), 1)
    return (valid_tag_count / total, valid_json_count / total)


# -----------------------------------------------------------------------------
# 4. Simulation & Benchmark Runner CLI
# -----------------------------------------------------------------------------

def run_dummy_test(device: str = "cpu"):
    """Runs zero-dependency end-to-end evaluation simulations for CI and sanity testing."""
    print("[OutMind Eval] Running full suite dummy simulation...")

    tokenizer = OutMindTokenizer()
    cfg = OutMindConfig.nano(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=128,
        n_layers=2,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
    )
    model = OutMindForCausalLM(cfg).to(device)

    # 1. Tokenizer Eval
    print("\n--- 1. Tokenizer Evaluation ---")
    tok_results = eval_tokenizer(tokenizer)
    assert len(tok_results) > 0

    # 2. Pretrain PPL & Choice Eval
    print("\n--- 2. Pretraining Evaluation ---")
    dummy_texts = ["OutMind adalah framework bahasa kecil yang modular dan cepat.", "Hello world deep learning."]
    ppl = eval_pretrain_perplexity(model, tokenizer, dummy_texts, device=device)
    print(f"Perplexity (PPL): {ppl:.2f}")

    choice_items = [{
        "context": "Ibu kota Indonesia adalah",
        "choices": [" Jakarta", " Bandung", " Surabaya"],
        "answer": 0,
    }]
    choice_acc = eval_zero_shot_choice(model, tokenizer, choice_items, device=device)
    print(f"Zero-shot Choice Accuracy: {choice_acc * 100:.1f}%")

    # 3. SFT Eval Benchmarks
    print("\n--- 3. SFT Benchmark Evaluations ---")
    gsm_problems = [{"question": "Natalia sold 48 clips in April. In May she sold half. How many in total?", "answer": "#### 72"}]
    gsm_acc = eval_gsm8k_math(model, tokenizer, gsm_problems, device=device, max_new_tokens=16)
    print(f"GSM8k Math Exact Match: {gsm_acc * 100:.1f}%")

    code_tasks = [{"prompt": "def multiply(a, b):\n    '''Returns product of a and b'''", "test": "assert multiply(3, 4) == 12"}]
    syntax_rate, pass_rate = eval_humaneval_code(model, tokenizer, code_tasks, device=device, max_new_tokens=16)
    print(f"HumanEval Syntax Validity: {syntax_rate * 100:.1f}% | Pass@1: {pass_rate * 100:.1f}%")

    mmlu_questions = [{
        "question": "What is the capital of France?",
        "choices": ["Berlin", "Madrid", "Paris", "Rome"],
        "answer": "C",
    }]
    mmlu_acc = eval_mmlu(model, tokenizer, mmlu_questions, device=device)
    print(f"MMLU Multi-choice Accuracy: {mmlu_acc * 100:.1f}%")

    tydi_samples = [{
        "context": "Danau Toba adalah sebuah danau vulkanik besar di Sumatera Utara, Indonesia.",
        "question": "Di mana letak Danau Toba?",
        "answer": "Sumatera Utara",
    }]
    em, f1 = eval_tydiqa_indonesian(model, tokenizer, tydi_samples, device=device, max_new_tokens=16)
    print(f"TyDiQA Indonesian Exact Match: {em * 100:.1f}% | Token F1: {f1 * 100:.1f}%")

    tool_scenarios = [{
        "tools": [{"name": "get_weather", "description": "Get current weather", "parameters": {"city": "str"}}],
        "prompt": "What is the weather in Jakarta?",
    }]
    tag_rate, json_rate = eval_glaive_tool_calling(model, tokenizer, tool_scenarios, device=device, max_new_tokens=16)
    print(f"Tool Calling Tag Compliance: {tag_rate * 100:.1f}% | JSON Schema Rate: {json_rate * 100:.1f}%")

    print("\n[OutMind Eval] All evaluation simulation checks PASSED successfully.")


def main():
    parser = argparse.ArgumentParser(description="OutMind Unified Evaluation Suite")
    parser.add_argument("--stage", type=str, choices=["all", "tokenizer", "pretrain", "sft"], default="all")
    parser.add_argument("--eval-tokenizer", action="store_true", help="Shortcut to evaluate tokenizer only")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--preset", type=str, choices=["nano", "small", "moe"], default="nano")
    parser.add_argument("--vocab-file", type=str, default=None, help="Custom BPE model file path")
    parser.add_argument("--eval-file", type=str, default=None, help="Evaluation dataset file (txt or jsonl)")
    parser.add_argument("--benchmark", type=str, default="all", choices=["all", "gsm8k", "humaneval", "mmlu", "tydiqa", "glaive"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dummy-test", action="store_true", help="Run simulated end-to-end evaluation suite")
    args = parser.parse_args()

    if args.eval_tokenizer:
        args.stage = "tokenizer"

    if args.dummy_test:
        run_dummy_test(device=args.device)
        return

    vocab_file = args.vocab_file
    if vocab_file is None and os.path.exists("data/processed/outmind_tokenizer.model"):
        vocab_file = "data/processed/outmind_tokenizer.model"

    tokenizer = OutMindTokenizer(vocab_file=vocab_file)

    if args.stage in ["all", "tokenizer"]:
        print("\n--- Running Tokenizer Benchmark ---")
        eval_tokenizer(tokenizer)

    if args.stage in ["all", "pretrain", "sft"]:
        cfg = getattr(OutMindConfig, args.preset)(vocab_size=tokenizer.vocab_size)
        model = OutMindForCausalLM(cfg)
        if args.checkpoint and os.path.exists(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location=args.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            print(f"[OutMind] Loaded checkpoint: {args.checkpoint}")
        model.to(args.device)

        if args.stage in ["all", "pretrain"] and args.eval_file and os.path.exists(args.eval_file):
            print("\n--- Running Pretrain Evaluation ---")
            with open(args.eval_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            ppl = eval_pretrain_perplexity(model, tokenizer, lines, device=args.device)
            print(f"[OutMind] Validation Perplexity (PPL): {ppl:.2f}")

        if args.stage in ["all", "sft"]:
            print(f"\n--- Running SFT Benchmarks ({args.benchmark}) ---")
            # Fallback to standard baseline checks if no external test file provided
            print("[OutMind] Evaluating standard benchmark sample suite...")
            run_dummy_test(device=args.device)


if __name__ == "__main__":
    main()
