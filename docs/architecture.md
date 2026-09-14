# Architecture Specification: OutMind

## 1. Overview & Vision
OutMind is a lightweight, educational, and hackable open-source framework designed for building and training custom Small Language Models (SLMs) from scratch. Inspired by **nanoGPT**, **MiniMind**, and modern frontier architectures (**LLaMA-3**, **DeepSeek-V3**, **Qwen3.8-Max**), OutMind focuses on clean, transparent PyTorch implementations without superfluous abstractions.

---

## 2. Core Transformer Architecture

OutMind implements a modern **Causal Decoder-Only Transformer** featuring pre-normalization, rotary position embeddings, grouped-query attention, gated activations, and an optional sparse mixture of experts.

```
                           Input Token IDs
                                 │
                                 ▼
                     Token Embedding (V, d_model)
                                 │
     ┌───────────────────────────┴───────────────────────────┐
     │ Transformer Layer (repeated L times)                  │
     │                                                       │
     │   x ──► RMSNorm ──► Grouped-Query Attention (GQA) ──(+)
     │   │                 (with RoPE & KV-Cache)        │   │
     │   │                                               │   │
     │   └───────────────────────────────────────────────┘   │
     │                                                       │
     │   x ──► RMSNorm ──► SwiGLU FFN (or Sparse MoE) ──(+)─►│
     │   │                                               │   │
     │   └───────────────────────────────────────────────┘   │
     └───────────────────────────┬───────────────────────────┘
                                 │
                           Final RMSNorm
                                 │
                     LM Head (d_model, V)
                                 │
                                 ▼
                           Output Logits
```

---

## 3. Mathematical Primitives

### 3.1 Pre-RMSNorm (Root Mean Square Normalization)
Replaces standard LayerNorm by removing mean-centering ($E[x]$), reducing compute overhead by ~15% while providing identical gradient stability:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
* Where $\gamma \in \mathbb{R}^d$ is a learnable scale parameter and $\epsilon = 10^{-6}$.

### 3.2 Rotary Position Embedding (RoPE)
Applies rotation to Query and Key representations in 2D complex subspaces rather than adding learned absolute positional vectors:
$$\mathbf{R}_{\Theta, m}^d = \text{diag}\left(R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m}\right)$$
$$R_{\theta_i, m} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$
* Provides natural relative distance awareness and enables context extrapolation.

### 3.3 Grouped-Query Attention (GQA) & KV-Cache
Bridges standard Multi-Head Attention (MHA) and Multi-Query Attention (MQA). $N_q$ query heads share $N_{kv}$ key/value heads (e.g., ratio of 3:1 or 4:1):
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M_{\text{causal}}\right) V$$
* **KV-Cache**: In autoregressive generation, past $K$ and $V$ tensors are stored in cache, turning step-by-step token generation complexity from $O(N^2)$ to $O(1)$ per step.

### 3.4 SwiGLU Feed-Forward Network
Outperforms standard GELU and ReLU non-linearities:
$$\text{SwiGLU}(x) = \left(x W_{\text{gate}} \odot \text{SiLU}(x W_{\text{up}})\right) W_{\text{down}}$$
* Where $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d_{model} \times d_{ffn}}$ and $W_{\text{down}} \in \mathbb{R}^{d_{ffn} \times d_{model}}$.

---

## 4. Sparse Mixture of Experts (MoE)

Following the architectural insights of **Qwen3.8-Max** and **DeepSeek-V3**, OutMind provides an educational **Sparse MoE** module as a configurable toggle in `model.py`:

* **Router**: A linear projection computing routing logits:
  $$s(x) = \text{Softmax}(\text{TopK}(x W_{\text{gate}}, k=2))$$
* **Shared + Routed Experts**:
  - $N$ routed experts (e.g., 4 or 8), where only top-$k$ ($k=2$) are computed per token.
  - 1 optional permanently active **Shared Expert** to retain core base language knowledge.
* **Output**:
  $$\text{MoE}(x) = \text{SharedFFN}(x) + \sum_{i \in \text{TopK}} s_i(x) \cdot \text{Expert}_i(x)$$

---

## 5. Model Configuration Presets

OutMind defines three standard model tiers:

| Parameter | OutMind-Nano (CPU / Laptop) | OutMind-Small (Single GPU) | OutMind-MoE (Sparse) |
| :--- | :--- | :--- | :--- |
| **Total Parameters** | **~34.0 Million** | **~124.7 Million** | **~121.9 Million (Total) / ~48.6M (Active)** |
| **Hidden Dim ($d_{model}$)** | 384 | 768 | 512 |
| **Layers ($L$)** | 6 | 12 | 8 |
| **Query Heads ($N_q$)** | 6 | 12 | 8 |
| **KV Heads ($N_{kv}$)** | 2 (GQA 3:1) | 4 (GQA 3:1) | 2 (GQA 4:1) |
| **Intermediate Dim ($d_{ffn}$)** | 1,024 | 2,048 | 1,365 per expert |
| **Experts / Top-K** | Dense | Dense | 4 Routed (Top-2) + 1 Shared |
| **Max Sequence Length** | 2,048 tokens | 2,048 tokens | 2,048 tokens |
| **Vocabulary Size** | 64,000 (Trilingual + Code/Math) | 64,000 (Trilingual + Code/Math) | 64,000 (Trilingual + Code/Math) |
| **Hardware Target** | Any modern laptop CPU | RTX 3060 / 4060 / T4 | Single GPU / 8GB VRAM |
