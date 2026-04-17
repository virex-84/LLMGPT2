# LLMGPT2: Ultra-Compact GPT-2 Implementation in C# with OpenCL & GGUF Export

A C# implementation for creating and training extremely small GPT-2 style models from scratch, accelerated via OpenCL through ILGPU. The primary goal is to produce highly efficient models that can be exported to the GGUF format for inference on `llama.cpp`, LM Studio, and other modern LLM runners.

The final exported GGUF model is only **~440 KB**, making it suitable for highly constrained environments.

## Features

- **100% C#**: Model definition, training loop, and GGUF exporter are all written in C# using ILGPU.
- **OpenCL Acceleration**: GPU-accelerated training via ILGPU's OpenCL backend.
- **GGUF Export**: Native export to GGUF v3 format for maximum compatibility.
- **Ultra-Lightweight**: Final model ~440 KB, ~104K parameters.
- **Stateless Single-Turn Chat**: Truncated chat template prevents context accumulation, making it a predictable dialogue agent.

## Model Architecture

Standard decoder-only GPT-2 transformer with minimal parameter count.

### Model Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `n_params` | ~104K | Total trainable parameters |
| `n_vocab` | 512 | BPE vocabulary size |
| `n_ctx` | 63 | Maximum context length (auto-adapted from data) |
| `n_embd` | 64 | Embedding dimension |
| `n_head` | 2 | Number of attention heads |
| `n_layer` | 1 | Number of transformer blocks |
| `head_dim` | 32 | Dimension per head (`n_embd / n_head`) |
| `ffn_dim` | 128 | Feed-forward hidden dimension |

**Parameter breakdown:**
- Token embeddings: 32,768  (`512 × 64`)
- Position embeddings: 4,032  (`63 × 64`)
- Transformer block (1×): 33,472
  - Attention (Q/K/V + Wo): 16,640
  - Two LayerNorms: 256
  - FFN (GELU, two linear layers): 16,576
- Final layer norm: 128
- LM head projection: 33,280  (`64 × 512 + 512`)
- **Total: 103,680 parameters**

### Layer Structure (Sequential)

```
Input IDs
  ↓
Token Embedding (wte)           [vocab=512, dim=64]
  ↓
+ Positional Embedding (wpe)    [seq_len=63, dim=64]  ← learnable + sinusoidal
  ↓
×1 × TransformerBlockPreNorm
  ├─ LayerNorm 1 (attn_norm)    [dim=64]
  ├─ MultiHeadAttention         [qkv fused, 2 heads, causal mask]
  ├─ Residual Add
  ├─ LayerNorm 2 (ffn_norm)     [dim=64]
   ├─ FeedForward (GELU)         [64 → 128 → 64]
  └─ Residual Add
  ↓
Final LayerNorm (ln_f)          [dim=64]
  ↓
Linear LM Head (lm_head)        [64 → 512]
  ↓
Logits → Softmax → Token
```

### Transformer Block Composition (Pre-Norm)

The model contains **one** transformer block following the **Pre-LayerNorm** architecture:

```
x ──┐
    │
    ├─→ LayerNorm(x) ─→ MultiHeadAttention(Q,K,V) ──┐
    │                                              │
    └──────────────────────+───────────────────────┘
                           │
                           v
                     x + attention_output
                           │
                           ├─→ LayerNorm ─→ GELU(Linear_up) ─→ Linear_down
                           │
                           └──────────────────────+───────────────┘
                                                  │
                                                  v
                                            x + ffn_output
```

**Attention details:**
- **Causal self-attention** with look-ahead mask (triangular)
- Q/K/V projections: fused into single `[seq_len, 3*emb_dim]` kernel
- Head dimension: `emb_dim / num_heads = 32`
- Scale factor: `1/√32 ≈ 0.1768`
- Output projection: separate linear layer

**Feed-Forward details:**
- Two-layer MLP: `Linear(emb→hidden) → GELU → Linear(hidden→emb)`
- Hidden dimension: 128 (2× embedding)
- Biases enabled on all linear layers
- GELU activation: `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715*x³)))`

## Training Data

Two-stage training pipeline on custom Russian-language datasets.

### Pre-training Corpus

- **Size:** ~2 KB (24 sentences)
- **Content:** Basic factual statements about natural phenomena
- **Purpose:** Teach fundamental language structure and word relationships
- **Example:** "солнце восходит на востоке и садится на западе"

### Fine-tuning Corpus (Dialogue)

- **Size:** ~8 KB (54 single-turn Q&A pairs)
- **Content:** Greetings, introductions, and simple conversational exchanges
- **Purpose:** Adapt model for single-turn dialogue with specified format

**Total training data:** ~10 KB of raw text.

## Chat Template & Stateless Behavior

A defining feature is the intentionally **stateless** chat template. Unlike typical chat models that accumulate conversation history, this model only sees the **last** user message.

**Template stored in GGUF metadata:**
```
{% if messages %}
{% set message = messages[-1] %}
{% if message['role'] == 'user' %}
<user> {{ message['content'] }}<assistant>
{% endif %}
{% endif %}
```

**Runtime format:**
```
<user> {user_input}<assistant>
```

No past turns are included in the context window. This ensures:
- No memory of previous interactions
- Pure single-turn instruction following
- Predictable behavior regardless of chat history
- Prevents context pollution over long conversations

Special tokens used:
| Token | ID | Role |
| :--- | :--- | :--- |
| `<pad>` | 0 | Padding |
| `<unk>` | 1 | Unknown |
| `<s>` | 2 | BOS (added by tokenizer) |
| `</s>` | 3 | EOS (appended in training data) |
| `<user>` | 4 | User turn marker |
| `<assistant>` | 5 | Assistant response start |
| `<sep>` | 6 | Separator (unused) |
| `<sep>` | 6 | Separator (unused) |

## Tokenization

- **Algorithm:** Byte-Pair Encoding (BPE)
- **Pre-tokenization:** GPT-2 regex with special token awareness
- **Byte mapping:** GPT-2 `bytes_to_unicode()` for full UTF-8 coverage
- **Initial vocab:** Special tokens (7) + 256 byte tokens = 263 base tokens
- **Merges:** Learned incrementally from training corpus
- **Final vocab size:** 512 tokens (custom BPE vocab for dialogue)

The BPE trainer forces special tokens as atomic units, preventing them from being split during encoding.

## GGUF Export

The `GgufExporter` produces a fully compliant **GGUF v3** file:

- **Architecture:** `gpt2`
- **Context length:** 63 tokens (auto-adapted from corpus 95th percentile + rounding)
- **Tensor format:** FP32 (all tensors stored as 32-bit floats)
- **Vocab:** Standard GPT-2 byte-level BPE with custom special tokens
- **Metadata:** Includes chat template, token IDs, merges, and scores
- **Tensor remapping:** Vocab is reordered to match GPT-2 standard layout (special tokens at 0–6, then byte tokens in canonical order, then merges)
- **Positional embeddings:** Combined sinusoidal + learnable (summed, then stored as `position_embd.weight`)
- **No quantization:** Full precision (model is already tiny)

**Tensors exported:**
```
token_embd.weight          [64, 512]
position_embd.weight       [64, 63]

blk.0.attn_norm.weight     [64]
blk.0.attn_norm.bias       [64]
blk.0.attn_qkv.weight      [64, 192]   (Q/K/V fused, transposed)
blk.0.attn_qkv.bias        [192]
blk.0.attn_output.weight   [64, 64]    (Wo)
blk.0.attn_output.bias     [64]
blk.0.ffn_norm.weight      [64]
blk.0.ffn_norm.bias        [64]
blk.0.ffn_up.weight        [64, 128]   (W1)
blk.0.ffn_up.bias          [128]
blk.0.ffn_down.weight      [128, 64]   (W2)
blk.0.ffn_down.bias        [64]

output_norm.weight         [64]
output_norm.bias           [64]
output.weight              [64, 512]   (lm_head, transposed)
output.bias                [512]
```

**Final file size:** ~440 KB (103,680 parameters × 4 bytes × overhead ~6%).

## Technical Stack

| Component | Technology |
| :--- | :--- |
| Language | C# 12 / .NET 8 |
| GPU Backend | ILGPU 1.5.3 (OpenCL) |
| Compute | ILGPU.Algorithms |
| Tokenizer | Custom BPE (GPT-2 compatible) |
| Export Format | GGUF v3 |
| Target Runtimes | llama.cpp, LM Studio, Ollama (via conversion) |

## Project Structure

```
LLMGPT2/
├── src/
│   ├── GPT2/
│   │   ├── LLMGPT2.cs           # Main model class (train + inference)
│   │   └── GgufExporter.cs      # GGUF v3 exporter
│   ├── Layers/
│   │   ├── EmbeddingLayer.cs
│   │   ├── PositionalEmbeddingLayer.cs  (learnable + sinusoidal)
│   │   ├── TransformerBlockPreNorm.cs   # Pre-Norm block
│   │   ├── MultiHeadAttentionLayer.cs   # Causal attention
│   │   ├── LayerNormLayer.cs
│   │   ├── GELUFeedForwardLayer.cs      # MLP with GELU
│   │   ├── LinearLayer.cs
│   │   └── LayerWeights.cs
│   ├── Tokenizers/
│   │   ├── BPETokenizer.cs      # GPT-2 style BPE
│   │   └── ITokenizer.cs
│   ├── ModelProfile.cs          # Auto-config + corpus analyzer
│   ├── Vocab.cs                 # Token ↔ ID mapping
│   ├── Context.cs               # ILGPU context wrapper
│   ├── DatasetLoader.cs         # JSON/CSV loader
│   ├── Adam.cs                  # Adam optimizer
│   ├── LossManager.cs           # Cross-entropy + softmax
│   └── MatrixOps.cs             # GPU kernels (softmax, clip, etc.)
├── Program.cs                   # Interactive CLI
├── data/
│   ├── pretraining_data.json   # 24 sentences
│   └── chat_training_data.json # 54 Q&A pairs
├── LLMGPT2.csproj              # ILGPU packages
└── README.md                   # This file
```

## Quick Start

### Prerequisites

- **.NET 8 SDK** or later
- **OpenCL 1.2+ compatible GPU** (AMD, Intel, NVIDIA)
- Windows 10/11, Linux, or macOS

### Build & Run

```bash
# Clone and enter project
cd LLMGPT2

# Restore packages
dotnet restore

# Build
dotnet build --configuration Release

# Run interactive CLI
dotnet run --project LLMGPT2.csproj
```

### Training Workflow

1. **Create new model** → BPE tokenizer trains on your dataset
2. **Select training phases** → choose pretrain, finetune, or both
3. **Training** → model learns from corpus (hyperparameters auto-configured)
4. **Export to GGUF** → `model.gguf` ready for `llama.cpp`

## Inference with GGUF

### Using `llama.cpp` (CLI)

```bash
llama-cli -m model.gguf -c 2048
```

### Using LM Studio

1. Drag `model.gguf` into LM Studio model path
2. Select the model
3. Start chatting — it will be single-turn only (no memory)


## How It Works

### Forward Pass (GPU)

1. **Token IDs** → `EmbeddingLayer` (lookup) → `[seq_len, emb_dim]`
2. Add **sinusoidal + learnable** positional encodings
3. For each Transformer block:
   - **Attn:** Q,K,V = Linear(x) → reshape heads → causal attention (Q·K/√d → softmax → attn·V) → Wo
   - **Add residual** (x + attn_out)
   - **FFN:** LayerNorm → Linear → GELU → Linear
   - **Add residual** (x + ffn_out)
4. **Final LayerNorm**
5. **Linear** → logits over vocabulary

### Backward Pass (GPU)

1. Compute cross-entropy loss (neg log-likelihood)
2. Backprop through LM head (softmax + linear gradient)
3. Reverse through each block:
   - Accumulate gradients from residual branch
   - Backward through FFN (W2 → GELU → W1)
   - Backward through attention (Wo → softmax → QKV)
4. Update weights via **Adam optimizer**
5. Gradient clipping (max norm = 2.0–5.0 depending on dataset size)

### GGUF Export Flow

1. **Infer config** from layer list (count blocks, read dimensions)
2. **Rebuild GPT-2 standard vocab** with remapping:
   - IDs 0–6: special tokens (`<pad>`, `<unk>`, `<s>`, `</s>`, `<user>`, `<assistant>`, `<sep>`)
   - IDs 7–262: 256 byte tokens in GPT-2 `bytes_to_unicode()` order
   - IDs 263+: BPE merges (preserving original merge rank as score)
3. **Collect all tensors** from GPU buffers:
   - Transpose weight matrices from `[in, out]` to `[out, in]` for GGML column-major layout
   - Reorder vocab rows in `token_embd.weight` and `output.weight` by remapping old→new IDs
   - Fuse sinusoidal into learnable positional embeddings
4. **Write file** in single pass:
   - Header (24 bytes)
   - KV metadata (including chat template)
   - Tensor info array
   - Padding to 32-byte alignment
   - Tensor data arrays (each aligned to 32 bytes)

No temporary file buffering — streamed directly to disk.

## Performance Notes

- **Memory:** ~3× parameters during training (weights + gradients + Adam moments)
  - 104K params × 4 bytes × 3 ≈ **1.2 MB** GPU memory (excluding buffers)
- **Training speed:** ~50–200 tokens/sec on integrated GPUs (depends on sequence length)
- **Inference speed:** ~15–50 tokens/sec on CPU (via llama.cpp)
- **Batch size:** 1 (no batching implemented — training is online)
- **Throughput:** Limited by data copying between CPU/GPU each step

## References

- [GPT-2 Original Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [ILGPU Documentation](https://ilgpu.net/docs/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp/)

## License

MIT