# OutMind Documentation

Welcome to the technical specifications and architectural documentation for **OutMind**, a minimalist and educational open-source framework for building, training, and fine-tuning Small Language Models (SLMs) from scratch.

---

## Documentation Index

1. **[Architecture Specification](architecture.md)**
   - Core Transformer decoder design (Pre-RMSNorm, RoPE, Grouped-Query Attention, SwiGLU).
   - Sparse Mixture-of-Experts (MoE) design with shared and routed experts.
   - Standard model presets: OutMind-Nano (34M), OutMind-Small (125M), and OutMind-MoE (122M total / 49M active).
   - Real-time KV-cache inference dynamics.

2. **[Training & Optimization Specification](training_and_optimization.md)**
   - AdamW decoupled weight decay mechanics and selective 2D matrix decay filtering.
   - 4-Tier Attention Fallback Hierarchy: FA3 $\rightarrow$ FA2 $\rightarrow$ PyTorch Native SDPA $\rightarrow$ Eager Math.
   - Parameter-Efficient Fine-Tuning with native pure PyTorch LoRA ($r=8, \alpha=16$) and zero-latency weight merging.
   - Automatic Mixed Precision (AMP) and gradient accumulation.

3. **[Tokenizer & Data Pipeline Specification](tokenizer_and_data.md)**
   - 64,000 standard vocabulary capacity with dynamic contiguous control tokens.
   - Control & special tokens (`[BOS]`, `[EOS]`, `[start_header_id]`, `[end_header_id]`, `[EOT]`, `<osagent_mode>`, etc.).
   - Indonesian-optimized BPE regex splitter supporting reduplications (`anak-anak`), acronym clitics (`KTP-nya`), and contractions (`'kan`, `'ku`).
   - Chat templating and SFT prompt loss masking (`ignore_index=-100`).
   - Verified dataset sources and split isolation boundaries (Pretrain vs SFT vs Eval).

4. **[Unified Evaluation Suite (scripts/eval.py)](../scripts/eval.py)**
   - **Tokenizer**: Compression ratio, character fertility, and round-trip lossless fidelity.
   - **Pretraining**: Cross-entropy loss, Perplexity (PPL), and zero-shot multi-choice log-likelihood scoring.
   - **SFT**: GSM8K (Math CoT), HumanEval (Python AST & Pass@1), MMLU (General Reasoning), TyDiQA (Indonesian Reading Comprehension), and Glaive (Tool Calling).

5. **[Unified Dataset Specification](dataset.md)**
   - Three-Tier partition architecture (`pretrain/`, `sft/`, `validation/`).
   - 24.5 GB raw pretraining mixture across ID, EN, ZH, Code, and Math.
   - Standardized ChatML schema and online deduplication for SFT instruction tuning.
   - Zero-contamination held-out validation benchmarks (GSM8K, HumanEval, MMLU, TyDiQA).
   - Usage instructions and Hugging Face Hub integration (`Mesosfer/outmind-dataset`).
