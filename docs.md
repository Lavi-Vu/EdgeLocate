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

**Loss formulation**. At training time, the model minimizes the negative log-likelihood of the ground-truth token sequence. For a sequence of length $T$ with labels $y_t$ (where $y_t = -100$ for masked positions), the loss is:

$$ \mathcal{L} = -\frac{1}{\sum_t \mathbb{1}[y_t \neq -100]} \sum_{t=1}^{T} \mathbb{1}[y_t \neq -100] \cdot \log p(y_t \mid x_{<t}) $$

where $p(y_t \mid x_{<t}) = \text{softmax}(\mathbf{W}_{\text{lm}} \mathbf{h}_t)$ is the probability assigned to the correct token $y_t$ by the LM head given the LLM's hidden state $\mathbf{h}_t$ at position $t$.

In contrast, a standard regression-based detection model would use:

$$ \mathcal{L}_{\text{reg}} = \sum_{i} \| \mathbf{b}_i - \hat{\mathbf{b}}_i \|_2^2 $$

where $\mathbf{b}_i$ is the $i$-th ground-truth box and $\hat{\mathbf{b}}_i$ is the predicted box. This only provides gradients to the regression head and the features it directly consumes — a much shallower gradient path.

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

### 4.2 Multi-Head Attention

The core attention operation in both the LLM and vision encoder is **scaled dot-product attention**. Given queries $\mathbf{Q} \in \mathbb{R}^{T \times d_k}$, keys $\mathbf{K} \in \mathbb{R}^{T \times d_k}$, and values $\mathbf{V} \in \mathbb{R}^{T \times d_v}$:

$$ \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M} \right) \mathbf{V} $$

where $\mathbf{M}$ is the attention mask ($0$ for allowed positions, $-\infty$ for masked positions). **Multi-head attention** runs $H$ parallel attention heads and concatenates:

$$ \begin{aligned}
\text{head}_i &= \text{Attention}(\mathbf{Q} \mathbf{W}_i^Q, \mathbf{K} \mathbf{W}_i^K, \mathbf{V} \mathbf{W}_i^V) \\[2pt]
\text{MHA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \mathbf{W}^O
\end{aligned} $$

The LLM uses **Grouped Query Attention (GQA)** with 14 query heads and 2 key/value heads — reducing KV cache size during inference by sharing KV projections across groups of query heads. The vision encoder uses standard multi-head self-attention ($\mathbf{Q} = \mathbf{K} = \mathbf{V}$).

### 4.3 LoRA (Low-Rank Adaptation)

LoRA freezes the pre-trained weight matrix $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$ and injects a trainable low-rank decomposition:

$$ \mathbf{W}' = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \mathbf{B} \mathbf{A} $$

where $\mathbf{B} \in \mathbb{R}^{d \times r}$, $\mathbf{A} \in \mathbb{R}^{r \times k}$, and the rank $r \ll \min(d, k)$. During training:

$$ \mathbf{h} = \mathbf{W}' \mathbf{x} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} \mathbf{B} \mathbf{A} \mathbf{x} $$

- **Scale factor**: $\alpha / r = 256 / 128 = 2$ controls the magnitude of the update
- **Initialization**: $\mathbf{A} \sim \mathcal{N}(0, \sigma^2)$, $\mathbf{B} = \mathbf{0}$ (so $\Delta \mathbf{W} = \mathbf{0}$ at start)
- Applied to all 6 linear projection types in each transformer layer: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Only BA matrices are trained; original W stays frozen
- Saves ~459M parameters from being trained

### 4.4 Positional Encoding

**LLM — 1D Rotary Position Embeddings (RoPE)**. RoPE applies a rotation to the query and key vectors based on their position $p$ in the sequence. For a vector $\mathbf{x}$ at position $p$, the rotary transformation for dimension pair $(2i, 2i+1)$ is:

$$ \begin{aligned}
    \text{RoPE}(\mathbf{x}, p)_{2i} &= x_{2i} \cos(p \theta_i) - x_{2i+1} \sin(p \theta_i) \\[2pt]
    \text{RoPE}(\mathbf{x}, p)_{2i+1} &= x_{2i} \sin(p \theta_i) + x_{2i+1} \cos(p \theta_i)
\end{aligned} $$

where $\theta_i = 10000^{-2i/d}$ for standard RoPE. With this formulation, the attention score between positions $p$ and $q$ depends only on their relative offset $(p - q)$, since:

$$ \text{RoPE}(\mathbf{x}, p)^\top \text{RoPE}(\mathbf{y}, q) = \mathbf{x}^\top \mathbf{R}_{p-q} \mathbf{y} $$

Qwen2.5 uses $\theta_i = 1000000^{-2i/d}$ (a larger base, extending the context to 32K tokens).

**SigLIP/SigLIP2**: Learned 1D absolute position embeddings of shape $(N_{\text{patches}}, 768)$. These are added to the patch embeddings before the first transformer layer. SigLIP2 interpolates position embeddings from the checkpoint to match the model's expected grid size.

**MoonViT — 2D RoPE**: Instead of 1D positions, RoPE is applied on a 2D grid with indices $(i, j)$ for each patch at grid position $(i, j)$. The rotation uses two separate frequencies for the height and width dimensions:

$$ \begin{aligned}
    \text{RoPE}_\text{2D}(\mathbf{x}, i, j) &= \text{RoPE}(\text{RoPE}(\mathbf{x}, i), j)
\end{aligned} $$

This decouples the $x$ and $y$ spatial dimensions, enabling the model to attend to spatial regions at any resolution without position embedding interpolation.

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
| **PBD support** | Yes | Yes | Yes |

### 5.2 SigLIP Path

Standard HuggingFace ViT. The input image $\mathbf{I} \in \mathbb{R}^{3 \times H \times W}$ is resized to $224 \times 224$ and normalized. It is then split into $N = HW / p^2$ non-overlapping patches of size $p \times p$ (here $p = 16$, so $N = 196$). Each patch is flattened and linearly projected:

$$ \mathbf{x}_i = \mathbf{W}_{\text{embed}} \cdot \text{flatten}(\mathbf{I}[:, i]) + \mathbf{p}_i, \quad \mathbf{W}_{\text{embed}} \in \mathbb{R}^{768 \times (3 \cdot 16 \cdot 16)} $$

where $\mathbf{p}_i$ is a learned position embedding. The resulting patch tokens $\mathbf{X} = [\mathbf{x}_1; \ldots; \mathbf{x}_N] \in \mathbb{R}^{N \times 768}$ pass through 24 transformer layers. Output is `(196, 768)` patch features. Simple, reliable, works out of the box.

### 5.3 SigLIP2 Path

SigLIP2 changes the patch embedding from Conv2d to Linear. The model expects each patch to be pre-flattened — so the raw image `(3, 224, 224)` must be manually unrolled into `(196, 768)` before the Linear layer. The manual patchify operation is:

$$ \mathbf{X}_{i,j} = \text{flatten}\left( \mathbf{I}[:, ip:(i+1)p, jp:(j+1)p] \right) \in \mathbb{R}^{3p^2} $$

for each grid position $(i, j)$ where $p = 16$. The resulting tensor $\mathbf{X} \in \mathbb{R}^{(H/p)(W/p) \times 3p^2}$ is then passed through the Linear embedding:

$$ \mathbf{z}_k = \mathbf{W}_{\text{embed}} \mathbf{x}_k, \quad \mathbf{W}_{\text{embed}} \in \mathbb{R}^{768 \times 768} $$

Note the weight shape mismatch: the checkpoint stores Conv2d weights $\mathbf{W}_{\text{conv}} \in \mathbb{R}^{768 \times 3 \times 16 \times 16}$, requiring a reshape $\mathbf{W}_{\text{embed}} = \text{reshape}(\mathbf{W}_{\text{conv}}, 768, 768)$ to load correctly.

### 5.4 MoonViT Path

A custom 27-layer ViT designed for high-resolution, multi-scale input:

- **2D RoPE**: Rotary position embeddings are computed on a 2D grid of size $(H/14, W/14)$ rather than a 1D sequence. For a patch at grid position $(i, j)$, the rotation is applied independently to the height and width axes:

$$ \mathbf{x}_{i,j}' = \text{RoPE}_h(\text{RoPE}_w(\mathbf{x}, i), j) $$

where each dimension has its own frequency set $\{\theta_k^{(h)}\}$ and $\{\theta_k^{(w)}\}$. This lets the model attend to spatial relationships at any resolution without position embedding interpolation.
- **Patch Merge**: After the 27 encoder layers, adjacent patches are merged using a 2×2 convolution. If the encoder produces features $\mathbf{Z} \in \mathbb{R}^{(H/14) \times (W/14) \times 1152}$, the merge operation produces:

$$ \mathbf{Z}'_{i,j} = \text{Conv2d}_{2\times2}(\mathbf{Z}_{2i:2i+2, 2j:2j+2}) \in \mathbb{R}^{4608} $$

Output has $1/4$ the spatial size but $4\times$ the channel width (4608 = 1152 × 4). These merged tokens form the final visual features that are projected to the LLM's embedding space.
- **Native Resolution**: Can process images at their original resolution (or tiles thereof), avoiding the information loss from aggressive downsampling.

PBD works with any vision encoder since it operates on the projected visual features in the LLM's embedding space, independent of the encoder architecture.

---

## 6. Projector

The projector (also called the connector or MLP bridge) maps vision features from the VE's dimension to the LLM's 896-dim embedding space.

### 6.1 Architecture

**For SigLIP/SigLIP2 (768 → 896):**

$$ \begin{aligned}
    \mathbf{h} &= \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1, \quad \mathbf{W}_1 \in \mathbb{R}^{896 \times 768} \\[2pt]
    \mathbf{h} &= \text{GELU}(\mathbf{h}) \\[2pt]
    \mathbf{z} &= \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2, \quad \mathbf{W}_2 \in \mathbb{R}^{896 \times 896}
\end{aligned} $$

- 2 layers, no bias in practice ($\mathbf{b}_1 = \mathbf{b}_2 = 0$), no normalization
- ~1.4M parameters ($768 \times 896 + 896 \times 896 = 1,376,256$)

**For MoonViT (4608 → 896):**

$$ \begin{aligned}
    \mathbf{h} &= \text{LayerNorm}(\mathbf{x}), \quad \mathbf{x} \in \mathbb{R}^{4608} \\[2pt]
    \mathbf{h} &= \text{GELU}(\mathbf{W}_1 \mathbf{h}), \quad \mathbf{W}_1 \in \mathbb{R}^{896 \times 4608} \\[2pt]
    \mathbf{z} &= \mathbf{W}_2 \mathbf{h}, \quad \mathbf{W}_2 \in \mathbb{R}^{896 \times 896}
\end{aligned} $$

- 2 layers with LayerNorm before first linear
- ~4.2M parameters ($4608 \times 896 + 896 \times 896 = 4,931,584$)

The GELU activation is defined as:

$$ \text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right] $$

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

Each bounding box is defined by its top-left and bottom-right corners in pixel coordinates:

$$ \mathbf{b} = (x_1, y_1, x_2, y_2) \quad \text{where } 0 \leq x_1 < x_2 \leq W,\; 0 \leq y_1 < y_2 \leq H $$

These continuous pixel values are quantized into 1001 integer bins $[0, 1000]$ using the image dimensions $(W, H)$:

$$ \begin{aligned}
    d_1 &= \left\lfloor \frac{x_1 \cdot 1000}{W} \right\rceil \\[2pt]
    d_2 &= \left\lfloor \frac{y_1 \cdot 1000}{H} \right\rceil \\[2pt]
    d_3 &= \left\lfloor \frac{x_2 \cdot 1000}{W} \right\rceil \\[2pt]
    d_4 &= \left\lfloor \frac{y_2 \cdot 1000}{H} \right\rceil
\end{aligned} $$

where $\lfloor \cdot \rceil$ denotes rounding to the nearest integer. The resulting integers are clamped to $[0, 1000]$ and mapped to vocabulary token IDs:

$$ \text{token\_id}(d_i) = \text{coord\_start} + d_i = 151670 + d_i $$

Example serialization:
```
<ref>cat</ref><box><432><219><687><544></box>
```

**Coordinate quantization error**. The maximum spatial quantization error at input resolution $R$ is:

$$ \Delta = \frac{R}{1000} $$

For $R = 224$, $\Delta \approx 0.224$ pixels — sub-pixel precision, negligible for detection tasks where IoU-based evaluation tolerates such errors. For MoonViT processing at native 448×448, $\Delta \approx 0.448$ pixels.

### 7.3 Coordinate Decoding (Inference)

The reverse operation recovers pixel-space coordinates from token-space bins:

$$ \begin{aligned}
    x_1 &= \frac{d_1 \cdot W}{1000} \\[2pt]
    y_1 &= \frac{d_2 \cdot H}{1000} \\[2pt]
    x_2 &= \frac{d_3 \cdot W}{1000} \\[2pt]
    y_2 &= \frac{d_4 \cdot H}{1000}
\end{aligned} $$

This denormalization is performed per-image using each image's actual dimensions $(W, H)$, so the model correctly handles images of any resolution despite always predicting in the fixed $[0, 1000]$ token space.

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
- **SwiGLU activation** in the feed-forward network (intermediate size = 4864):

$$ \text{SwiGLU}(\mathbf{x}) = \text{Swish}(\mathbf{W}_1 \mathbf{x}) \odot (\mathbf{W}_2 \mathbf{x}), \quad \text{Swish}(x) = x \cdot \sigma(x) $$

The full FFN is:
$$ \text{FFN}(\mathbf{x}) = \mathbf{W}_O \left( \text{Swish}(\mathbf{W}_G \mathbf{x}) \odot \mathbf{W}_U \mathbf{x} \right) $$

where $\mathbf{W}_G, \mathbf{W}_U \in \mathbb{R}^{4864 \times 896}$ are the gate and up projections, $\mathbf{W}_O \in \mathbb{R}^{896 \times 4864}$ is the down projection, and $\odot$ is element-wise multiplication.
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

The language modeling head is an **untied linear layer**: $\text{LMHead}(\mathbf{h}) = \mathbf{W}_{\text{lm}} \mathbf{h}$ where $\mathbf{W}_{\text{lm}} \in \mathbb{R}^{152673 \times 896}$. "Untied" means $\mathbf{W}_{\text{lm}}$ is a separate weight matrix, independent from the input embedding matrix $\mathbf{E} \in \mathbb{R}^{152673 \times 896}$. The logits for position $t$ are:

$$ \mathbf{z}_t = \mathbf{W}_{\text{lm}} \mathbf{h}_t $$

and the predicted token is $\hat{y}_t = \arg\max \mathbf{z}_t$. In the tied case ($\mathbf{W}_{\text{lm}} = \mathbf{E}^\top$), the LM head shares weights with the embedding layer, but this is undesirable here because:

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

Let the total sequence length be $T = C + B$, where $C$ is the context length (user prompt + image features) and $B = 6$ is the block size. Define the attention mask $\mathbf{M} \in \mathbb{R}^{T \times T}$ as:

$$ \mathbf{M}_{i,j} = \begin{cases}
0, & \text{if } j \leq i \text{ and } i < C \quad \text{(causal within context)} \\[2pt]
0, & \text{if } j < C \text{ and } i \geq C \quad \text{(block → all context)} \\[2pt]
0, & \text{if } j \geq C \text{ and } i \geq C \quad \text{(non-causal within block)} \\[2pt]
-\infty, & \text{otherwise}
\end{cases} $$

In attention matrix form:

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

- Context tokens ($i < C$) use standard causal masking: each attends only to itself and previous context tokens
- Prediction tokens ($i \geq C$) see ALL context tokens (full visibility into the prompt)
- Prediction tokens see ALL other prediction tokens (non-causal within the block)
- Each prediction token can condition on the others, enabling coordinated box prediction

The attention scores are computed as:

$$ \text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M} \right) \mathbf{V} $$

where $d_k$ is the head dimension and $-\infty$ entries in $\mathbf{M}$ cause the softmax to output zero attention weight.

**2. Weighted Coordinate Averaging**

Instead of taking the argmax for each coordinate position, `decode_bbox_avg()` uses a **weighted average** of the top-$k$ logits. Let $\ell_1, \ldots, \ell_{1001}$ be the logits for the 1001 coordinate tokens at a given position. The top-$k$ candidate values and their logits are:

$$ \mathcal{C}_k = \left\{ (v_j, \ell_j) \mid \ell_j \in \text{top-}k(\ell_1, \ldots, \ell_{1001}) \right\} $$

A softmax is applied over the selected logits:

$$ p_j = \frac{\exp(\ell_j / \tau)}{\sum_{m=1}^{k} \exp(\ell_m / \tau)} $$

where $\tau$ is the temperature (default $\tau = 1.0$, greedy when $\tau \to 0$). The weighted average coordinate value is:

$$ \hat{v} = \left\lfloor \frac{ \sum_{j=1}^{k} p_j \cdot v_j }{ \sum_{j=1}^{k} p_j } \right\rceil $$

In **fast** mode this average is always used. In **hybrid** mode, the block is flagged as `error_box` if the top-1 probability is low and the candidate spread is wide:

$$ \text{flag}\ = \begin{cases}
\text{error\_box}, & \text{if } p_{\max} < 0.9 \ \wedge\ \text{spread} > 60 \\
\text{coord\_box}, & \text{otherwise}
\end{cases} $$

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

The training objective is the **masked cross-entropy loss** over the vocabulary. Let $\mathbf{h}_t \in \mathbb{R}^{896}$ be the LLM's hidden state at position $t$, and $\mathbf{W}_{\text{lm}} \in \mathbb{R}^{152673 \times 896}$ be the LM head weight matrix. The logits for position $t$ are:

$$ \mathbf{z}_t = \mathbf{W}_{\text{lm}} \mathbf{h}_t \in \mathbb{R}^{152673} $$

The predicted probability of token $v$ at position $t$ is:

$$ p_t(v) = \frac{\exp(z_{t,v})}{\sum_{j=1}^{152673} \exp(z_{t,j})} $$

Let $y_t \in \{0, \ldots, 152672\}$ be the ground-truth token ID at position $t$, and let $\mathcal{M} = \{t \mid y_t \neq -100\}$ be the set of non-masked positions. The loss is:

$$ \mathcal{L} = -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \log p_t(y_t) $$

Key detail: loss is **masked** — only computed on the assistant (GPT) response portion of the sequence:

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

A random long-edge resize strategy. Given an image of dimensions $(W, H)$, let $L = \max(W, H)$. With 50% probability, a target long-edge length $L'$ is sampled uniformly:

$$ L' \sim \mathcal{U}[640, 2560] $$

The image is resized preserving aspect ratio:

$$ s = \frac{L'}{L}, \quad (W', H') = (sW, sH) $$

Then all images are resized to the model's input size $(R, R)$ (e.g., 224×224) using Lanczos interpolation. This provides scale diversity without distorting aspect ratios.

Finally, pixel values are normalized for the vision encoder:

$$ \mathbf{x}_{\text{norm}} = \frac{\mathbf{x}_{\text{tensor}} - \mu}{\sigma}, \quad \mu = \sigma = 0.5 $$

mapping $[0, 1]$ to $[-1, 1]$.

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
   - If mode ≠ slow → `generate_pbd()` (parallel)
   - Otherwise → `model.generate()` (autoregressive)
4. **Text decoding**: Token IDs → text
5. **Box parsing**: Regex extract patterns:
   - Primary: `<ref>.*?</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>`
   - Fallbacks for malformed output
6. **Coordinate denormalization**: `coord × img_dim / 1000`
7. **Output**: `List[[x1, y1, x2, y2]]` in pixel coordinates

### 11.2 Evaluation Metrics

**Intersection-over-Union (IoU)**. For a predicted box $\mathbf{b}_p$ and ground-truth box $\mathbf{b}_g$ defined by their corners $(x_1, y_1, x_2, y_2)$:

$$ \begin{aligned}
    \mathbf{b}_p \cap \mathbf{b}_g &= \max(0, \min(x_{p,2}, x_{g,2}) - \max(x_{p,1}, x_{g,1})) \\[2pt]
    &\quad \times \max(0, \min(y_{p,2}, y_{g,2}) - \max(y_{p,1}, y_{g,1})) \\[2pt]
    \mathbf{b}_p \cup \mathbf{b}_g &= A_p + A_g - (\mathbf{b}_p \cap \mathbf{b}_g) \\[2pt]
    \text{IoU}(\mathbf{b}_p, \mathbf{b}_g) &= \frac{ \mathbf{b}_p \cap \mathbf{b}_g }{ \mathbf{b}_p \cup \mathbf{b}_g }
\end{aligned} $$

where $A_p$ and $A_g$ are the areas of the predicted and ground-truth boxes.

**Average Precision (AP)**. A detection is considered a true positive if $\text{IoU} \geq \tau$ for some threshold $\tau$. Predictions are ranked by confidence score, and precision-recall curve is computed. AP at threshold $\tau$ is the 11-point interpolated average precision:

$$ \text{AP}_\tau = \frac{1}{11} \sum_{r \in \mathcal{R}} \max_{\tilde{r} \geq r} p(\tilde{r}) $$

where $\mathcal{R} = \{0.0, 0.1, \ldots, 1.0\}$ are 11 equally-spaced recall levels and $p(r)$ is the precision at recall $r$.

**Mean AP (COCO standard)**:

$$ \text{mAP} = \frac{1}{10} \sum_{\tau \in \mathcal{T}} \text{AP}_\tau, \quad \mathcal{T} = \{0.50, 0.55, \ldots, 0.95\} $$

**Precision, Recall, F1** at a given IoU threshold $\tau$:

$$ \begin{aligned}
    \text{Precision}_\tau &= \frac{\text{TP}_\tau}{\text{TP}_\tau + \text{FP}_\tau} \\[2pt]
    \text{Recall}_\tau &= \frac{\text{TP}_\tau}{\text{TP}_\tau + \text{FN}_\tau} \\[2pt]
    \text{F1}_\tau &= 2 \cdot \frac{\text{Precision}_\tau \cdot \text{Recall}_\tau}{\text{Precision}_\tau + \text{Recall}_\tau}
\end{aligned} $$

| Metric | What it measures |
|---|---|
| **mAP** | Mean AP across IoU thresholds 0.50:0.05:0.95 |
| **AP@0.50** | PASCAL VOC standard (50% IoU) |
| **AP@0.75** | Strict localization (75% IoU) |
| **mIoU** | Mean pairwise IoU |
| **Precision / Recall / F1** | Per-threshold classification metrics |

### 11.3 Visualization

The `visualize_prediction()` function draws GT boxes (green) and predicted boxes (red) on the image with IoU scores. Supports `--visualize N` flag to save N sample outputs.

---

## 12. Configuration & Variants

### 12.1 Supported Configurations

| Variant | VE | LLM | Trainable | VRAM | PBD |
|---|---|---|---|---|---|---|
| SigLIP Base | `siglip-base-patch16-224` (93M) | Qwen2.5-0.5B | ~37M | ~4 GB | Yes |
| SigLIP2 Base | `siglip2-base-patch16-224` (98M) | Qwen2.5-0.5B | ~37M | ~4 GB | Yes |
| MoonViT | MoonViT (408M) | Qwen2.5-0.5B | ~44M | ~8 GB | Yes |
| + VE LoRA | Any + LoRA r=16 | Qwen2.5-0.5B | ~40–47M | +1 GB | Yes |
| + Packing | Any | Qwen2.5-0.5B | ~37M | ~4–8 GB | Yes |

### 12.2 Trade-offs

- **SigLIP vs MoonViT**: SigLIP is lighter and faster for inference. MoonViT supports native resolution and is 4× larger in the vision encoder. Both support PBD.
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
