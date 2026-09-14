# Training & Optimization Specification: OutMind

## 1. Optimizer: AdamW

OutMind utilizes **AdamW** (Loshchilov & Hutter, 2017) as the primary training optimizer for both Pretraining (PT) and Supervised Fine-Tuning (SFT).

### 1.1 Decoupled Weight Decay Mechanics
Standard Adam with L2 regularization couples weight decay with gradient scaling, causing frequently updated weights to receive insufficient decay. AdamW explicitly decouples weight decay directly into parameter updates:
$$\theta_t = \theta_{t-1} - \eta_t \lambda \theta_{t-1} - \eta_t \frac{m_t}{\sqrt{v_t} + \epsilon}$$

### 1.2 Hyperparameter Configuration
* **Base Learning Rate ($\eta$)**: $3 \times 10^{-4}$ (with linear warmup over the first $5\%$ of steps, followed by cosine decay down to $10\%$ of peak).
* **Betas**: $\beta_1 = 0.9, \beta_2 = 0.95$.
* **Epsilon**: $\epsilon = 10^{-8}$.
* **Weight Decay ($\lambda$)**: $0.1$.
* **Selective Decay Filtering**: Weight decay is applied **exclusively** to 2D transformation matrices (`Linear` weights). It is disabled for 1D tensors (RMSNorm gains $\gamma$ and embedding biases) to prevent destabilizing normalization dynamics.

---

## 2. Attention Acceleration: 4-Tier Fallback Hierarchy

To ensure maximum performance on server GPUs while guaranteeing crash-free execution on consumer laptops and non-CUDA environments, OutMind implements an automated 4-tier attention hierarchy:

```
                    ┌───────────────────────────────┐
                    │      Input: Q, K, V Tensors   │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  Tier 1: FlashAttention-3     │──► Available? ──► Execute FA3 (Hopper H100+)
                    └───────────────┬───────────────┘
                              Unavailable
                                    │
                    ┌───────────────▼───────────────┐
                    │  Tier 2: FlashAttention-2     │──► Available? ──► Execute FA2 (Ampere/Ada RTX 30/40)
                    └───────────────┬───────────────┘
                              Unavailable
                                    │
                    ┌───────────────▼───────────────┐
                    │  Tier 3: PyTorch Native SDPA  │──► Available? ──► Execute F.scaled_dot_product_attention
                    └───────────────┬───────────────┘
                              Unavailable
                                    │
                    ┌───────────────▼───────────────┐
                    │  Tier 4: Pure Eager Math      │──► Execute: softmax((Q @ K.T)/sqrt(d) + M) @ V
                    └───────────────────────────────┘
```

### 2.1 Tier Specifications

| Tier | Engine | Hardware Target | Characteristics |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **FlashAttention-3 (FA3)** | NVIDIA Hopper (H100/H200) | Uses Tensor Memory Accelerator (TMA) & warp specialization; up to 1.5–2× faster than FA2. |
| **Tier 2** | **FlashAttention-2 (FA2)** | NVIDIA Ampere & Ada (A100, RTX 30xx, RTX 40xx) | Parallelizes across sequence length and head dimension with minimal memory footprint. |
| **Tier 3** | **PyTorch Native SDPA** | Universal (CUDA, MPS, CPU) | `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.0+). Dispatches to internal C++ kernels without external package dependencies. |
| **Tier 4** | **Pure Eager Math** | Any PyTorch environment | Educational reference implementation: $\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$. |

### 2.2 Silent Runtime Probing Pattern
```python

def detect_attention_backend() -> str:
    try:
        import flash_attn_interface
        return "fa3"
    except ImportError:
        pass

    try:
        import flash_attn
        return "fa2"
    except ImportError:
        pass

    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return "sdpa"
    return "math"
```

---

## 3. Parameter-Efficient Fine-Tuning: Native LoRA

### 3.1 LoRA vs. QLoRA Trade-off Evaluation
* **QLoRA (4-bit NF4)**: Intended for fitting 7B–70B parameter models onto consumer cards. In models under 500M parameters, the base model in FP16 takes only ~50MB (25M) to ~200MB (100M). Introducing `bitsandbytes` adds fragile C++/CUDA dependencies on Windows with negligible memory benefits.
* **Native LoRA (Recommended)**: Pure PyTorch implementation (~40 LOC). Universal compatibility (CPU, Apple Silicon, Windows, Linux) with zero third-party dependencies.

### 3.2 Mathematical Formulation
Base weight $W_0 \in \mathbb{R}^{d \times k}$ is frozen. Low-rank decomposition matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ are trained with rank $r \ll \min(d, k)$:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
* **Default Hyperparameters**: Rank $r = 8$, Scaling factor $\alpha = 16$.
* **Weight Merging**: For zero-latency inference after fine-tuning, LoRA adapters are collapsed into the base weights:
  $$W_{\text{merged}} = W_0 + \frac{\alpha}{r} (B \cdot A)$$

---

## 4. Mixed Precision & Gradient Accumulation

1. **Automatic Mixed Precision (AMP)**:
   - Uses `torch.autocast(device_type, dtype=torch.bfloat16)` where supported (Ampere+, modern CPUs), or `torch.float16` with `GradScaler`.
2. **Gradient Accumulation**:
   - Enables arbitrary effective batch sizes on limited VRAM by accumulating gradients across $N$ micro-batches before executing `optimizer.step()` and `optimizer.zero_grad()`.
