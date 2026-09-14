# Tokenizer & Data Pipeline Specification: OutMind

## 1. Overview & Vocabulary Architecture

OutMind adopts the **Byte-Level Byte-Pair Encoding (BPE)** scheme with token controls compatible with **Moonshot AI's Kimi-K3** structure. It utilizes a standard vocabulary capacity of **64,000** tokens (optimized for Bahasa Indonesia, English, Chinese, Mathematics LaTeX, and Source Code), while supporting arbitrary scaling up to 163,840.

All special control tokens are mapped **contiguously** at the top of the vocabulary index space (`N_merges .. N_merges + len(specials) - 1`), eliminating unallocated embedding rows and token gap panics.

---

## 2. Special Tokens Specification

The model incorporates structured control tokens to demarcate message headers, role boundaries, agent modes, and media placeholders:

| Token String | Relative Index | Type | Architectural Purpose |
| :--- | :--- | :--- | :--- |
| `[BOS]` | `Top - 16` | Special | **Beginning of Sequence**: Injected at the start of input sequences. |
| `[EOS]` | `Top - 15` | Special | **End of Sequence**: Emitted by the model to signify text completion. |
| `<|end_of_msg|>` | `Top - 14` | Special | Explicit message delimiter. |
| `<|open|>` | `Top - 13` | Added | XTML structural open tag. |
| `<|close|>` | `Top - 12` | Added | XTML structural close tag. |
| `<|sep|>` | `Top - 11` | Added | XTML attribute separator. |
| `[start_header_id]` | `Top - 10` | Special | Role header open delimiter (e.g., `[start_header_id]user[end_header_id]`). |
| `[end_header_id]` | `Top - 9` | Special | Role header close delimiter. |
| `[EOT]` | `Top - 8` | Special | **End of Turn**: Marks the conclusion of a speaker's turn. |
| `<|media_begin|>` | `Top - 7` | Special | Beginning of visual/audio/media token blocks. |
| `<|media_content|>` | `Top - 6` | Special | Media feature placeholder. |
| `<|media_end|>` | `Top - 5` | Special | End of media token blocks. |
| `<|media_pad|>` | `Top - 4` | Special | Media tensor padding. |
| `<osagent_mode>` | `Top - 3` | Special | Autonomous OS Agent execution mode indicator. |
| `[UNK]` | `Top - 2` | Special | Fallback unknown token representation. |
| `[PAD]` | `Top - 1` | Special | Sequence padding token for batched training. |

---

## 3. Indonesian-Optimized BPE Regex Splitter (`pat_str`)

### 3.1 Linguistic Rationale for Bahasa Indonesia
Standard BPE pre-splitters (such as GPT-4 or standard Kimi-K3) treat hyphens (`-`) as general punctuation symbols. In Indonesian, this causes severe fragmentation:
1. **Reduplication (Kata Ulang)**: Words such as `anak-anak`, `berlari-lari`, and `buku-buku` are fractured into separate tokens (`[" anak", "-", "anak"]`), damaging morphology and wasting context window capacity.
2. **Affixation to Acronyms**: Hyphenated acronym clitics such as `KTP-nya` and `SIM-ku` are broken across punctuation boundaries.
3. **Ordinal Numerals**: Formats like `ke-2`, `ke-10`, and `1990-an` lose structural cohesion.
4. **Colloquial Contractions**: Suffixes with apostrophes like `'kan`, `'ku`, `'mu`, and `'nya` need native grouping alongside English contractions (`'s`, `'re`, `'ve`).

### 3.2 Production Regex Implementation
```python
import regex as re

pat_str = "|".join([
    # 1. Han / CJK script (retained for multilingual/code compatibility)
    r"""[\p{Han}]+""",
    
    # 2. Latin / Indonesian words with support for hyphenated reduplication
    #    and contractions (Indonesian: 'kan, 'ku, 'mu, 'nya; English: 's, 't, 're, 've, etc.)
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?:-[\p{L}\p{N}]+)*(?i:'s|'t|'re|'ve|'m|'ll|'d|'kan|'ku|'mu|'nya)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?:-[\p{L}\p{N}]+)*(?i:'s|'t|'re|'ve|'m|'ll|'d|'kan|'ku|'mu|'nya)?""",
    
    # 3. Indonesian ordinal and temporal affix patterns (e.g., ke-1, ke-24, 1990-an)
    r"""(?:ke-\p{N}+|\p{N}+-an)""",
    
    # 4. Numeric digits grouped 1 to 3 numbers (handles decimals, currency, dates)
    r"""\p{N}{1,3}""",
    
    # 5. Punctuation, symbols, and operators
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    
    # 6. Whitespace and newline groupings
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])
```

### 3.3 Tokenization Comparison

| Input Text | Standard Splitter | Indonesian-Optimized Splitter | Benefit |
| :--- | :--- | :--- | :--- |
| `anak-anak bermain` | `[' anak', '-', 'anak', ' bermain']` | `[' anak-anak', ' bermain']` | Preserves single-word semantics |
| `KTP-nya hilang` | `[' KTP', '-', 'nya', ' hilang']` | `[' KTP-nya', ' hilang']` | Prevents punctuation token waste |
| `peringkat ke-2` | `[' peringkat', ' ke', '-', '2']` | `[' peringkat', ' ke-2']` | Clean numerical-morphological unit |
| `bukan begitu'kan?` | `[' bukan', ' begitu', "'", 'kan', '?']` | `[' bukan', " begitu'kan", '?']` | Preserves colloquial dialogue flow |

---

## 4. Training Datasets & Chat Template

### 4.1 Chat Template Format
OutMind structures conversation turns using the Kimi-K3 / LLaMA-3 header scheme:
```
[BOS][start_header_id]system[end_header_id]
Kamu adalah OutMind, asisten AI cerdas.[EOT]
[start_header_id]user[end_header_id]
Halo, apa kabar?[EOT]
[start_header_id]assistant[end_header_id]
Halo! Saya baik. Bagaimana saya bisa membantu Anda hari ini?[EOT]
```

### 4.2 Supervised Fine-Tuning (SFT) Prompt Masking
During SFT, the model must **only** learn to generate the assistant's responses, not memorize user prompts or system headers:
* **Target Tensor**: All tokens corresponding to `system`, `user`, and header boundaries are masked with label `-100`:
  $$\text{Target}_i = \begin{cases} -100 & \text{if token } i \text{ belongs to prompt or header} \\ \text{InputToken}_{i+1} & \text{if token } i \text{ is assistant output} \end{cases}$$
* PyTorch's `nn.CrossEntropyLoss(ignore_index=-100)` automatically skips gradient calculation for masked positions.

---

## 5. Verified Dataset Sources & Split Verification

To prevent benchmark contamination and ensure rigorous data engineering, all datasets are strictly partitioned into three isolated functional tiers:

### 5.1 Pretraining Datasets (Continuous Next-Token Prediction)
*Objective: Unsupervised language representation, grammar, and multi-domain world knowledge.*

| Domain | Dataset Source (Hugging Face) | Split / Config | License | Role |
| :--- | :--- | :--- | :--- | :--- |
| **Indonesian (ID)** | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) | `20231101.id` | CC-BY-SA 3.0 | Clean encyclopedic Indonesian prose |
| **Indonesian Web (ID)** | [`uonlp/CulturaX`](https://huggingface.co/datasets/uonlp/CulturaX) | `id` | ODC-By | Filtered, deduplicated Indonesian web corpus |
| **English (EN)** | [`HuggingFaceTB/smollm-corpus`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | `cosmopedia-v2` / `fineweb-edu-dedup` | ODC-By | High-quality educational textbook knowledge |
| **Chinese (ZH)** | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) | `20231101.zh` | CC-BY-SA 3.0 | Standard Simplified & Traditional Chinese knowledge |
| **Coding (Code)** | [`HuggingFaceTB/smollm-corpus`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | `python-edu` | ODC-By | Syntactically sound code and docstrings |
| **Math (Math)** | [`open-web-math/open-web-math`](https://huggingface.co/datasets/open-web-math/open-web-math) | `train` | ODC-By | Mathematical equations, LaTeX, and technical prose |

### 5.2 Supervised Fine-Tuning (SFT) Datasets (Prompt-Masked Instruction Tuning)
*Objective: Instruction adherence, multi-turn conversation, reasoning CoT, function calling, and identity.*

| Domain / Task | Dataset Source (Hugging Face) | Filter / Split | License | Target Capability |
| :--- | :--- | :--- | :--- | :--- |
| **Indonesian Instruction** | [`CohereForAI/aya_dataset`](https://huggingface.co/datasets/CohereForAI/aya_dataset) | `language == "ind"` | Apache-2.0 | Native Indonesian conversational & task instructions |
| **English Dialogue** | [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | `train_sft` | MIT | Multi-turn coherent dialogical interaction |
| **Chinese Instruction** | [`m-a-p/COIG-CQIA`](https://huggingface.co/datasets/m-a-p/COIG-CQIA) | `train` | Apache-2.0 / Open | High-quality Chinese Q&A and reasoning |
| **Math Reasoning (CoT)** | [`meta-math/MetaMathQA`](https://huggingface.co/datasets/meta-math/MetaMathQA) | `train` | MIT | Step-by-step mathematical problem solving |
| **Coding Instructions** | [`ise-uiuc/Magicoder-Evol-Instruct-110K`](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K) | `train` | Apache-2.0 | Complex programming tasks, debugging, and refactoring |
| **Tool / Function Calling** | [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | `train` | Apache-2.0 | Structured JSON tool definitions, invocations, and returns |
| **Identity & Guardrails** | Synthetic Generator (`data/raw/identity.jsonl`) | Handcrafted | Apache-2.0 | Mesosfer-AI identity grounding in ID, EN, and ZH |

### 5.3 Evaluation Benchmarks (Strictly Held-Out & Isolated)
*Objective: Objective metric evaluation. ZERO overlap or exposure during Pretraining or SFT.*

| Target Capability | Benchmark Dataset (Hugging Face) | Split | Metric | Decontamination Rule |
| :--- | :--- | :--- | :--- | :--- |
| **Indonesian Comprehension** | [`google-research-datasets/tydiqa`](https://huggingface.co/datasets/google-research-datasets/tydiqa) | `validation` (`indonesian`) | F1 / Exact Match | Isolated from pretrain/SFT corpora |
| **Indonesian Translation/Fluency** | [`facebook/flores`](https://huggingface.co/datasets/facebook/flores) | `devtest` (`ind_Latn`) | BLEU / chrF++ | Excluded from SFT pairs |
| **Math Reasoning** | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | `test` (1,319 problems) | Exact Match Accuracy | Strictly held-out; SFT uses disjoint train augmentations |
| **Code Generation** | [`bigcode/humanevalpack`](https://huggingface.co/datasets/bigcode/humanevalpack) | `test` (Python) | Pass@1 | Decontaminated against Magicoder training subsets |
| **General Knowledge** | [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) | `test` (57 subjects) | Multi-choice Accuracy | Never ingested during any training phase |

