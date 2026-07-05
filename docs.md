# EdgeLocate Model Architecture

## Table of Contents
1. [Model Overview](#1-model-overview)
2. [Design Philosophy](#2-design-philosophy)
3. [Backbone Architecture](#3-backbone-architecture)
4. [Building Blocks](#4-building-blocks)
5. [Vision Encoder (VE)](#5-vision-encoder)
6. [Projector (Connector)](#6-projector)
7. [Token System](#7-token-system)
8. [Language Model (LLM)](#8-language-model)
9. [Parallel Box Decoding (PBD)](#9-parallel-box-decoding)
10. [Training Pipeline](#10-training-pipeline)
11. [Inference & Evaluation](#11-inference--evaluation)
12. [Configuration & Variants](#12-configuration--variants)

---

## 1. Model Overview

EdgeLocate is a **<1 billion parameter vision-language model** for open-vocabulary object detection. It is a reimplementation of NVIDIA's LocateAnything (EagleVL), optimized for resource-constrained environments (4–8 GB VRAM).

At its core, EdgeLocate treats object detection as a **language generation problem**: it takes an image and a text prompt (e.g., "Find all the cats"), and generates bounding box coordinates as discrete vocabulary tokens — similar to how an LLM generates words in a sentence.

### The Pipeline (High-Level)

```
Input (Image + Text Prompt)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Vision Encoder (frozen)                                  │
│     Extracts visual features from the image                  │
│     Output: (B, N_patches, D_ve)                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Projector (trainable)                                    │
│     Maps visual features → LLM embedding space              │
│     Output: (B, N_patches, 896)                              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  3. LLM + LoRA (LoRA trainable, base frozen)                │
│     Replaces <|image|> token with projected visual features  │
│     Generates box tokens autoregressively or in parallel     │
│     Output: logits over 152,673-token vocabulary             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Token Decoding                                          │
│     Parses generated tokens into <ref>label</ref><box><d1>.. │
│     Denormalizes coordinate tokens → pixel-space boxes      │
│     Output: List of [x1, y1, x2, y2] bounding boxes         │
└─────────────────────────────────────────────────────────────┘
```

### Quick Stats

| Component | Parameters | Trainable? |
|---|---|---|
| Vision Encoder | 93–408M | No (frozen) |
| Projector | ~2–4M | Yes |
| Qwen2.5-0.5B base | 494M | No (frozen) |
| LoRA adapters | ~35M | Yes |
| LM Head | ~137M | Yes |
| **Total** | **~589–904M** | **~37–44M trainable** |

---

## 2. Design Philosophy

Five key principles drive the architecture:

### 2.1 Detection as Language Generation

Instead of using specialized detection heads (like DETR's object queries or Faster R-CNN's region proposal network), EdgeLocate formulates detection as **next-token prediction**. Bounding box coordinates are 1001 discrete bins (`<0>`–`<1000>`), each represented as a unique vocabulary token. The model generates a sequence like:

```
<ref>cat</ref><box><d1><d2><d3><d4></box><ref>dog</ref><box><d5><d6><d7><d8></box>
```

### 2.2 Cross-Entropy Over Regression

Why discrete tokens rather than an MSE regression head?

- **Gradient flow**: Cross-entropy back-propagates through every layer (LM head → LLM → projector → VE), providing rich supervision to all components
- **No task-specific heads**: The same LM head predicts both text tokens and coordinate tokens — no need for separate regression, classification, or object query modules
- **Natural multi-box support**: The autoregressive generation loop naturally produces variable-length box sequences without predefined maximums

### 2.3 Frozen + Adapter Paradigm

The base components (VE and LLM) are frozen to save memory. Only lightweight adapters are trained:

- **LoRA** (Low-Rank Adaptation) on LLM attention projections — rank 128, ~35M trainable params
- **Projector** — a small MLP mapping VE → LLM dimensions
- **LM Head** — the final classification layer (untied from embeddings)

This keeps memory at 4–8 GB, making training feasible on consumer GPUs.

### 2.4 Parallel Decoding for Speed

Autoregressive generation (1 token at a time) is slow for detection where patterns are highly structured. **Parallel Box Decoding (PBD)** predicts a complete 6-token box in one forward pass using a non-causal attention mask within each block.

### 2.5 Resolution Agnosticism

MoonViT uses **2D Rotary Position Embeddings** that generalize to arbitrary input resolutions without interpolation. This allows native-resolution processing — unlike SigLIP which requires fixed 224×224 input.

---

## 3. Backbone Architecture

The model follows a standard vision-language architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LocateAnythingForDetection                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐   ┌────────────┐   ┌───────────────────────────┐   │
│  │ Vision      │ → │ Projector  │ → │ Qwen2.5 + LoRA           │   │
│  │ Encoder     │   │ (MLP)      │   │ • 12 transformer layers   │   │
│  │             │   │            │   │ • 896 hidden dim          │   │
│  │ SigLIP /    │   │ 768→896    │   │ • LoRA r=128 on q/k/v/o  │   │
│  │ SigLIP2 /   │   │ or         │   │ • GQA (14 heads, 2 KV)   │   │
│  │ MoonViT     │   │ 4608→896   │   │ • RoPE, 32K context      │   │
│  └─────────────┘   └────────────┘   └───────────┬───────────────┘   │
│                                                  │                   │
│                                                  ▼                   │
│                                         ┌──────────────────┐        │
│                                         │ LM Head (untied) │        │
│                                         │ Linear(896,152673)│        │
│                                         └──────────────────┘        │
│                                                  │                   │
│                                                  ▼                   │
│                                         ┌──────────────────┐        │
│                                         │ Token Decoder    │        │
│                                         │ (PBD or AR)      │        │
│                                         └──────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: Image `(3, H, W)` + text prompt
2. **Text tokenization**: Prompt → token IDs, with `<|image|>` token inserted
3. **Image preprocessing**: Resize to model input size + normalize `(mean=0.5, std=0.5)`
4. **Vision encoding**: Patches → transformer encoder → patch features `(N, D_ve)`
5. **Projection**: Linear/gelu/linear → LLM-space features `(N, 896)`
6. **Feature merge**: Replace `<|image|>` embedding with projected visual features
7. **LLM forward**: Qwen2.5 processes the merged sequence with causal attention
8. **LM Head**: Projects final hidden states to vocabulary logits
9. **Decoding**: Cross-entropy loss (training) or token sampling (inference)

---

## 4. Building Blocks

### 4.1 The Merge Operation

The most critical architectural step: the `merge_visual_features` operation.

The text prompt contains a special `<|image|>` token (ID 151665). During merge:

1. The text prompt is embedded via the LLM's `embed_tokens` layer → `(B, text_len, 896)`
2. Visual features are extracted via VE + Projector → `(B, N_patches, 896)`
3. The single `<|image|>` embedding is **removed** and replaced with the full sequence of visual features
4. The remaining text embeddings follow after

The final sequence the LLM sees:

```
[tok_1, tok_2, ..., vis_1, vis_2, ..., vis_N, tok_M, tok_M+1, ...]
```

This is done per-sample, so each sequence may have a different length after merge.

### 4.2 LoRA (Low-Rank Adaptation)

LoRA decomposes weight updates into low-rank matrices:

```
W' = W + BA    where B ∈ R^(d×r), A ∈ R^(r×k), r=128
```

- Applied to all 6 linear projection types in each transformer layer: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Scale factor: alpha/r = 256/128 = 2×
- Only BA matrices are trained; original W stays frozen
- Saves ~459M parameters from being trained

### 4.3 Attention Types

The LLM uses **Grouped Query Attention (GQA)** with 14 query heads and 2 key/value heads — reducing KV cache size during inference. The vision encoder uses standard multi-head self-attention.

### 4.4 Positional Encoding

- **LLM**: 1D Rotary Position Embeddings (RoPE) with θ=1,000,000 for 32K context
- **SigLIP/SigLIP2**: Learned 1D absolute position embeddings (196 or 256 positions)
- **MoonViT**: 2D Rotary Position Embeddings applied in grid space — supports variable resolution natively

---

## 5. Vision Encoder

The vision encoder converts raw pixels into patch-level feature vectors. Three encoder paths exist, auto-detected at runtime based on model name or HuggingFace config.

### 5.1 Architecture Comparison

| Feature | SigLIP | SigLIP2 | MoonViT |
|---|---|---|---|
| **Source** | HuggingFace | HuggingFace | Custom (modeling_vit.py) |
| **Parameters** | ~93M | ~98M | ~408M |
| **Input resolution** | 224×224 fixed | 224×224 (or naflex) | Native (variable) |
| **Patch size** | 16×16 | 16×16 | 14×14 |
| **Patches (224 input)** | 196 | 196 | 256 |
| **Output dim** | 768 | 768 | 1152 (4608 after merge) |
| **Layers** | 24 | 24 | 27 |
| **Position encoding** | Learned 1D | Learned 1D (interpolated) | 2D RoPE |
| **PBD support** | No | No | **Yes** |

### 5.2 SigLIP Path

Standard HuggingFace ViT. Image resized to 224×224, normalized to `[0,1]`, patched into 16×16 tokens, passed through 24 transformer layers. Output is `(196, 768)` patch features. Simple, reliable, works out of the box.

### 5.3 SigLIP2 Path

SigLIP2 changes the patch embedding from Conv2d to Linear. The model expects each patch to be pre-flattened — so the raw image `(3, 224, 224)` must be manually unrolled into `(196, 768)` before the Linear layer. The checkpoint stores Conv2d weights, requiring a reshape from `(768, 3, 16, 16)` → `(768, 768)` to load correctly.

### 5.4 MoonViT Path

A custom 27-layer ViT designed for high-resolution, multi-scale input:

- **2D RoPE**: Rotary position embeddings are computed on a 2D grid `(H/14, W/14)` rather than a 1D sequence. This lets the model attend to spatial relationships at any resolution without position embedding interpolation.
- **Patch Merge**: After the 27 encoder layers, a 2×2 convolution merges adjacent patches. Output has 1/4 the spatial size but 4× the channel width (4608 = 1152 × 4).
- **Native Resolution**: Can process images at their original resolution (or tiles thereof), avoiding the information loss from aggressive downsampling.

MoonViT is the only encoder that supports PBD, because the patch merge step produces tokens that naturally group into spatial regions — aligning with the block decoding pattern.

---

## 6. Projector

The projector (also called the connector or MLP bridge) maps vision features from the VE's dimension to the LLM's 896-dim embedding space.

### 6.1 Architecture

**For SigLIP/SigLIP2 (768 → 896):**
```
Linear(768, 896) → GELU → Linear(896, 896)
```
- 2 layers, no bias, no normalization
- ~1.4M parameters

**For MoonViT (4608 → 896):**
```
LayerNorm(4608) → Linear(4608, 896) → GELU → Linear(896, 896)
```
- 2 layers with LayerNorm before first linear
- ~4.2M parameters

### 6.2 Purpose

The projector serves as a **learned adaptation layer**. Since the VE was pre-trained on image-text contrastive tasks (SigLIP) or unsupervised objectives (MoonViT), its feature space is not aligned with the LLM's text embedding space. The projector learns to transform visual semantics into the LLM's "language" — mapping spatial-visual concepts to the representation space where the LLM can reason about them.

Without the projector, the LLM would receive raw vision features in an alien embedding space, making it impossible to associate visual patterns with coordinate token predictions.

---

## 7. Token System

The most distinctive architectural choice: **bounding boxes are expressed as vocabulary tokens**.

### 7.1 Token Vocabulary

| Token Category | Token(s) | ID(s) | Count |
|---|---|---|---|
| Image anchor | `<\|image\|>` | 151665 | 1 |
| Box delimiters | `<box>`, `</box>` | 151666, 151667 | 2 |
| Label delimiters | `<ref>`, `</ref>` | 151668, 151669 | 2 |
| Coordinate bins | `<0>` – `<1000>` | 151670 – 152670 | 1001 |
| Null detection | `<null>` | 152671 | 1 |
| MTP mask | `<text_mask>` | 152672 | 1 |
| **Total added** | | | **1008** |
| **Qwen2.5 base** | | 0 – 151664 | ~151,665 |
| **Grand total** | | **0 – 152672** | **152,673** |

### 7.2 Coordinate Encoding (Data Preparation)

Each box is encoded as 4 coordinate tokens, normalized to `[0, 1000]`:

```
x1_token = round(x1_pixel * 1000 / img_width)
y1_token = round(y1_pixel * 1000 / img_height)
x2_token = round(x2_pixel * 1000 / img_width)
y2_token = round(y2_pixel * 1000 / img_height)
```

This produces integer tokens in `[0, 1000]`, which are then mapped to vocabulary IDs `151670 + token_value`.

Example serialization:
```
<ref>cat</ref><box><432><219><687><544></box>
```

### 7.3 Coordinate Decoding (Inference)

The reverse operation during inference:

```
x1_pixel = x1_token * img_width / 1000
y1_pixel = y1_token * img_height / 1000
```

This is done per-image using each image's actual dimensions, so the model works correctly regardless of input resolution.

### 7.4 Why 1000 Bins?

- At 224×224 input, one bin ≈ 0.224 pixels — sub-pixel precision is sufficient for detection
- At higher resolutions (e.g., 448×448 via MoonViT), precision scales to ~0.45 pixels per bin
- 1000 is large enough for adequate spatial precision but small enough to keep the vocabulary manageable
- Compare: 100 bins would give 2.24 px precision at 224px (too coarse for accurate boxes); 10,000 bins would add 9,000 tokens to the vocabulary (unnecessary)

---

## 8. Language Model

### 8.1 Backbone: Qwen2.5-0.5B-Instruct

A compact yet capable LLM from the Qwen family:

- **12 transformer layers** with 896 hidden dimension
- **Grouped Query Attention**: 14 query heads, 2 key/value heads (reduced KV cache)
- **32,768 token context window** with RoPE (θ=1,000,000)
- **SwiGLU activation** in the feed-forward network (intermediate size = 4864)
- **~494M parameters**

The Instruct variant is used because it's trained to follow instructions and produce structured outputs — important for generating formatted box sequences.

### 8.2 Visual Feature Injection

The `<|image|>` token acts as a **placeholder** in the text sequence. During forward pass:

1. Text is tokenized normally (including `<|image|>`)
2. Text tokens are embedded → `(B, L_text, 896)`
3. The `<|image|>` embedding at position `p` is **removed**
4. Visual features `(B, N_patches, 896)` are **inserted** at position `p`
5. The remaining text embeddings are appended after

The LLM sees the full sequence as one continuous stream — it doesn't distinguish between visual and text tokens at the attention level. The attention pattern is purely causal (each token attends to all previous tokens).

### 8.3 LM Head

The language modeling head is an **untied linear layer**: `Linear(896, 152673)`. "Untied" means it has its own weight matrix, independent from the input embedding matrix. This is important because:

- Adding 1001 coordinate tokens would require the embedding matrix to learn token representations for them
- The LM head needs to produce logits for these new tokens at output positions during generation
- Keeping them separate allows the LM head to specialize in classification while the embedding matrix focuses on representation

### 8.4 LoRA Adapters

LoRA adds trainable low-rank matrices to 6 linear projections per layer × 12 layers = 72 matrices total:

```
W' = W + BA,  B ∈ R^(896×128), A ∈ R^(128×dim)
```

- **Rank**: 128 (a good balance between expressiveness and parameter count)
- **Alpha**: 256 (scaling factor; effective LR is scaled by alpha/r = 2×)
- **Target modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Total trainable**: ~35M parameters

Without LoRA, training the full 494M-parameter LLM would require ~16 GB VRAM. With LoRA, it runs in 4–6 GB.

---

## 9. Parallel Box Decoding

PBD is the inference acceleration mechanism — not used during training. It exploits the fact that box tokens follow a rigid pattern (`<box>` + 4 coords + `</box>` = 6 tokens) that can be predicted in parallel.

### 9.1 The Problem with Autoregressive Decoding

Standard generation produces one token at a time:

```
Step 1: <ref>
Step 2: cat
Step 3: </ref>
Step 4: <box>
Step 5: <432>   ← each coordinate depends on previous tokens
Step 6: <219>   ← ...
Step 7: <687>   ← ...
Step 8: <544>   ← ...
Step 9: </box>
```

For a 32-box image, this requires 32 × 6 = 192 sequential decoding steps. Each step requires a full forward pass through the LLM (cached KV, but still slow).

### 9.2 PBD Solution

PBD predicts all 6 tokens of a box in a single forward pass:

```
Single step:
  Input: [context tokens (cached) + 6 mask tokens]
  Output: 6 logit vectors → 6 tokens simultaneously
```

This is possible because of two techniques:

**1. MTP Attention Mask**

The attention mask for the 6 prediction tokens is **non-causal within the block**:

```
            ctx_1  ctx_2  ...  ctx_N  pred_1  pred_2  ...  pred_6
ctx_1        ✓     .            .      ✓       ✓            ✓
ctx_2        ✓     ✓            .      ✓       ✓            ✓
...          .     .            .      .       .            .
ctx_N        ✓     ✓            ✓      ✓       ✓            ✓
pred_1       ✓     ✓            ✓      ✓       ✓            ✓
pred_2       ✓     ✓            ✓      ✓       ✓            ✓
...          .     .            .      .       .            .
pred_6       ✓     ✓            ✓      ✓       ✓            ✓
```

- Context tokens use standard causal masking
- Prediction tokens see ALL context tokens (full visibility)
- Prediction tokens see ALL other prediction tokens (non-causal)
- Each prediction token can condition on the others, enabling coordinated box prediction

**2. Weighted Coordinate Averaging**

Instead of taking the argmax for each coordinate position, `decode_bbox_avg()` uses a **weighted average** of the top-k logits:

```
For each of the 4 coordinate positions:
  1. Take top-4 logits within [coord_start, coord_end]
  2. Compute softmax over those 4 logits
  3. avg_coord = sum(softmax_score_i * coord_value_i) / sum(scores)
  4. Round to nearest integer in [0, 1000]
```

This produces smoother, more accurate coordinates than naive argmax — compensating for the fact that the model has less information per coordinate (since it hasn't seen the previous coordinate's actual value).

### 9.3 Generation Modes

```
┌──────────────────────────────────────────────────────────────────┐
│                        Generation Loop                            │
│                                                                    │
│  ┌─────────┐     valid box     ┌──────────┐     im_end           │
│  │  MTP    │ ────────────────→ │ Continue │ ──────────→ STOP     │
│  │  Block  │                   │  MTP     │                      │
│  │  (6 tx) │                   └──────────┘                      │
│  └────┬────┘                                                      │
│       │ error_box / ref_object                                    │
│       ▼                                                           │
│  ┌─────────┐     </box> ref     ┌──────────┐                     │
│  │  AR     │ ────────────────→ │ Resume   │                     │
│  │  (1 tx) │                   │  MTP     │                     │
│  └─────────┘                   └──────────┘                     │
└──────────────────────────────────────────────────────────────────┘
```

| Mode | Behavior | Speed | Quality |
|---|---|---|---|
| **fast** | Always MTP, never falls back | Fastest | Risk of malformed boxes |
| **hybrid** (default) | MTP for most boxes, AR fallback when MTP produces malformed output | Fast | Best balance |
| **slow** | Pure autoregressive (standard `model.generate()`) | Baseline | Matches training distribution |

### 9.4 Pattern Classification

After each MTP block, the 6 decoded tokens are classified:

| Pattern | Condition | Action |
|---|---|---|
| `coord_box` | `[<box>, 4×coord, </box>]` | Continue MTP |
| `error_box` | `<box>` present but malformed coords | Hybrid: fallback to AR |
| `empty_box` | Contains `<null>` | Skip, continue MTP |
| `im_end` | Contains end token | Stop generation |
| `ref_object` | Contains `<ref>` (text label) | Switch to AR for label |

---

## 10. Training Pipeline

### 10.1 Loss Function

Standard **cross-entropy** on the LM head's output logits. Key detail: loss is **masked** — only computed on the assistant (GPT) response portion of the sequence:

```
Text:   <|im_start|>user\n<|image|>\nFind the cat.<|im_end|>\n<|im_start|>assistant\n<ref>cat</ref><box><432><219><687><544></box><|im_end|>
Labels: -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 <ref> cat </ref> <box> <432> <219> <687> <544> </box> <im_end>
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        These tokens are ignored in loss (masked with -100)
                                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                                    These tokens are trained (coordinate tokens included)
```

The model learns to predict the full box sequence including the 4 coordinate values, just as it learns to predict any other vocabulary token.

### 10.2 Training Dynamics

- **Learning rate**: 2e-5 (cosine schedule with 3% warmup)
- **Batch size**: 4 per GPU (effective: 4 with gradient accumulation steps)
- **Precision**: BF16 mixed precision
- **Gradient checkpointing**: Enabled to save VRAM at the cost of 15% slower training
- **Weight decay**: 0.1

### 10.3 Data Format

Training data uses the ShareGPT format:

```json
{
  "image": "path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nFind all objects."},
    {"from": "gpt", "value": "<ref>cat</ref><box><432><219><687><544></box><ref>dog</ref><box><100><200><300><400></box>"}
  ]
}
```

### 10.4 Augmentation

A random long-edge resize strategy: with 50% probability, the image's longer edge is resized to a random value in `[640, 2560]` (preserving aspect ratio), then finally resized to the model's input size (224×224). This provides scale diversity without distorting aspect ratios.

### 10.5 What Freezes vs What Trains

```
FROZEN (saves memory):
├── SigLIP/MoonViT VE      93–408M params
├── Qwen2.5 base weights   459M params
└── (optionally VE LoRA if enabled)

TRAINABLE (~37M params total):
├── LoRA adapters (72 matrices)   ~35M
├── MLP Projector (2 layers)      ~2M
└── LM Head (untied)             ~137M (only small % learns; rest already trained)
```

---

## 11. Inference & Evaluation

### 11.1 Inference Flow

1. **Image preprocessing**: Resize to model input size + normalize
2. **Prompt formatting**: Prepend `<|image|>\n` to text, apply chat template
3. **Generation dispatch**:
   - If MoonViT + mode ≠ slow → `generate_pbd()` (parallel)
   - Otherwise → `model.generate()` (autoregressive)
4. **Text decoding**: Token IDs → text
5. **Box parsing**: Regex extract patterns:
   - Primary: `<ref>.*?</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>`
   - Fallbacks for malformed output
6. **Coordinate denormalization**: `coord × img_dim / 1000`
7. **Output**: `List[[x1, y1, x2, y2]]` in pixel coordinates

### 11.2 Evaluation Metrics

Standard COCO detection metrics:

| Metric | What it measures |
|---|---|
| **AP** (mean) | Average Precision across IoU thresholds 0.50:0.05:0.95 |
| **AP@0.50** | PASCAL VOC standard (50% IoU) |
| **AP@0.75** | Strict localization (75% IoU) |
| **mIoU** | Mean Intersection-over-Union |
| **Precision / Recall / F1** | At each IoU threshold |

### 11.3 Visualization

The `visualize_prediction()` function draws GT boxes (green) and predicted boxes (red) on the image with IoU scores. Supports `--visualize N` flag to save N sample outputs.

---

## 12. Configuration & Variants

### 12.1 Supported Configurations

| Variant | VE | LLM | Trainable | VRAM | PBD |
|---|---|---|---|---|---|
| SigLIP Base | `siglip-base-patch16-224` (93M) | Qwen2.5-0.5B | ~37M | ~4 GB | No |
| SigLIP2 Base | `siglip2-base-patch16-224` (98M) | Qwen2.5-0.5B | ~37M | ~4 GB | No |
| MoonViT | MoonViT (408M) | Qwen2.5-0.5B | ~44M | ~8 GB | Yes |
| + VE LoRA | Any + LoRA r=16 | Qwen2.5-0.5B | ~40–47M | +1 GB | Depends |
| + Packing | Any | Qwen2.5-0.5B | ~37M | ~4–8 GB | Depends |

### 12.2 Trade-offs

- **SigLIP vs MoonViT**: SigLIP is lighter and faster for inference but doesn't support PBD. MoonViT supports native resolution and PBD but is 4× larger in the vision encoder.
- **LoRA rank**: Default 128 is good for general detection. Higher rank (e.g., 256) captures more task-specific features at the cost of more parameters. Lower rank (64) for limited VRAM.
- **With vs without packing**: Packing increases training throughput by 1.5–2× but disables visual feature merge (uses raw `input_ids` instead of merged embeddings). Currently experimental.
- **Generation mode**: `hybrid` is recommended for production — nearly as fast as `fast` but handles edge cases robustly. `slow` for maximum compatibility. `fast` for maximum speed on clean data.

### 12.3 Typical Training Run

- Dataset: COCO 2017 (118k training images)
- Epochs: 3 (recommended minimum)
- Steps: ~11,000 (at batch size 4, grad accum 1)
- Time: ~4–6 hours on RTX 3090, ~8–12 hours on RTX 3060
- Expected AP@50 after 3 epochs: ~15–25 (limited by <1B model size + frozen VE)

---

## Summary

EdgeLocate reimagines object detection as a **language modeling task** inside a standard vision-language architecture:

1. A **Vision Encoder** converts images to patch features
2. A **Projector** bridges visual and language embedding spaces
3. A **Qwen2.5 LLM with LoRA** performs visual reasoning and generates box tokens
4. **Discrete coordinate tokens** encode bounding boxes as vocabulary items
5. **Parallel Box Decoding** accelerates inference via non-causal block attention
6. **LoRA + frozen base** keeps training feasible on constrained hardware

The result is a compact (<1B params) yet functional detection model that can run and train on consumer GPUs while maintaining the flexibility of a language-based approach — supporting arbitrary text prompts, multiple objects, and structured output.
