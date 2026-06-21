# Architecture

## Overview

EdgeLocate is a VLM for open-vocabulary object detection. Given an image and a text prompt, it autoregressively generates bounding box coordinates as discrete tokens.

```mermaid
flowchart LR
    subgraph Inputs
        I[Image: 224 x 224 x 3] --> VE[SigLIP-Base-Patch16-224<br/>92.88M params, frozen]
        T[Text Prompt] --> Tokenizer[Qwen2.5 Tokenizer<br/>vocab: 151644 base tokens]
        Tokenizer --> Emb[Token Embeddings<br/>seq_len x 896]
    end

    subgraph Vision Pipeline
        VE --> VP[196 patch features<br/>each 768-d]
        VP --> Proj[MLP Projector<br/>1.49M params, trainable<br/>Linear 768-896 + GELU + Linear 896-896]
    end

    Proj --> Merge{Merge at lpipeimagepipe token<br/>replace 1 embed with 196 visual tokens}

    subgraph LLM Backbone
        direction TB
        Merge --> LLM[Qwen2.5-0.5B-Instruct<br/>494.03M base params<br/>LoRA r=64-128: +35M trainable<br/>target: q k v o gate up down proj]
        LLM --> Attn[Self-Attention x24 layers<br/>with LoRA adapters]
        Attn --> FFN[FFN x24 layers<br/>gate/up/down with LoRA]
        FFN --> HS[Hidden States<br/>merged_seq_len x 896]
    end

    HS --> LMH[LM Head<br/>152669 x 896, untrained, trainable<br/>no weight tying]

    subgraph Outputs
        LMH --> Tokens[Logits: 152669 classes<br/>argmax -> token IDs]
        Tokens --> Detok[Detokenize to text]
        Detok --> Parse[Regex parse:<br/>box(d+)(d+)(d+)(d+)/box]
        Parse --> Boxes[Bounding Boxes<br/>x1 y1 x2 y2 in 0-1000<br/>divide by 1000 for normalized coords]
    end
```

```mermaid
flowchart TB
    subgraph Training Step
        direction TB
        A[Image: 224x224x3] --> B[SigLIP forward: 196x768]
        B --> C[MLP Projector: 196x768 to 196x896]
        C --> D[input_ids: tokenized conversation]
        D --> E[Locate image token lpipeimagepipe at pos k]
        E --> F[Slice embeddings: emb0:k-1, visual, embk+1:]
        F --> G[Merged seq len: T + 195<br/>each position: 896-d]
        G --> H[Expand labels: insert -100 for<br/>196 visual positions]
        H --> I[LLM+LoRA forward on merged embeds]
        I --> J[Logits: merged_seq_len x 152669]
        J --> K[Cross-entropy loss with shifted labels<br/>ignoring -100 positions]
        K --> L[Backprop through LM head, LoRA,<br/>projector (VE frozen)]
        L --> M[Update: LoRA adapters, LM head,<br/>embed_tokens, projector weights]
    end

    subgraph Inference Step
        direction TB
        N[Image: 224x224x3] --> O[SigLIP + Projector: 196x896]
        P[Text prompt] --> Q[Tokenize + apply_chat_template]
        Q --> R[input_ids with lpipeimagepipe token]
        O --> S[Merge: text embeds + visual embeds]
        R --> S
        S --> T[llm.generate with inputs_embeds]
        T --> U[KV cache accumulates full context]
        U --> V[Greedy sampling by default<br/>temperature=0]
        V --> W[Stop at lpipeim_endpipe or max_new_tokens]
        W --> X[Decode token IDs to text]
        X --> Y[Regex find all box(d+)(d+)(d+)(d+)/box]
        Y --> Z[Convert each group of 4 digits to<br/>x1 y1 x2 y2 ints in 0-1000]
    end

    Training Step -.-> |same VE + projector|Inference Step
```

## Components

### Vision Encoder (SigLIP-Base-Patch16-224)

- **Params**: 92.88M (frozen)
- **Output**: 196 patch features, each 768-d
- **Selection**: `vision_select_layer=-1` (final layer)

### MLP Projector

- **Params**: 1.49M (trainable)
- **Structure**: `Linear(768, 896) → GELU → Linear(896, 896)`
- **Purpose**: Projects visual features from VE space (768) to LLM space (896)

### LLM (Qwen2.5-0.5B-Instruct)

- **Params**: 494.03M base
- **Trainable via LoRA** (r=64–128): ~35M additional params
- **LoRA targets**: `q_proj, k_proj, v_proj, o_proj, gate_proj, down_proj, up_proj`
- **LoRA init**: A = Kaiming uniform, B = zeros (standard PEFT)
- **LM head**: Untied from input embeddings (`tie_word_embeddings=False`), separately trainable (152669 × 896)

### Vocabulary

| Token | ID | Count |
|---|---|---|
| Base Qwen vocab | 0–151643 | 151644 |
| `<|image|>` | 151665 | 1 |
| `<box>` | 151666 | 1 |
| `</box>` | 151667 | 1 |
| `<0>` – `<1000>` | 151668–152668 | 1001 |
| **Total** | | **152669** |

Coordinate tokens `<n>` map to integer bin `n` in range [0, 1000], representing the normalized coordinate `n / 1000`.

## Visual Feature Injection

The model uses `inputs_embeds` mode: the single `<|image|>` token embedding is replaced by 196 projected visual patch embeddings (each 896-d). The sequence becomes:

```
[sys_tokens, user_tokens, |image|, visual_1...visual_196, prompt_tokens, assistant_tokens]
```

The LLM processes the full merged sequence with KV cache, so hidden states at `assistant` positions are conditioned on visual content.

## Loss

Standard cross-entropy on all tokens. Labels are masked so only the assistant response (starting from `\n` after `assistant`) contributes to the loss. Visual token positions (the 196 injected patches) are set to `-100` (ignored).

## Training Loop

```
For each batch:
  1. Load image → SigLIP → 196×768 patch features
  2. Project via MLP → 196×896 visual tokens
  3. Replace <|image|> in input_ids with visual tokens → merged sequence
  4. Expand labels: insert -100 for visual token positions
  5. LLM forward pass (with LoRA) on merged embeddings
  6. Cross-entropy between logits and expanded labels
  7. Backprop through LM head → LLM+LoRA → projector → VE (VE frozen, no grad)
```

## Inference

Standard `llm.generate()` with `inputs_embeds`:

1. Convert image → 196 visual patch embeddings via VE + projector
2. Merge into text embeddings at `<|image|>` position
3. Pass `inputs_embeds` and `attention_mask` to `llm.generate()`
4. Autoregressively sample tokens until `<|im_end|>` or `max_new_tokens`
5. Parse `<box><d1><d2><d3><d4></box>` from generated text

## File Structure

```
locany/
  __init__.py          # Public API exports
  config.py            # ModelConfig, TrainingConfig, DataConfig, InferenceConfig
  model.py             # LocateAnythingForDetection, create_model, load_model_from_dir
  dataset.py           # DetectionDataset, parse_sharegpt_line
  training.py          # setup_training, DetectionDataCollator, save_model
  inference.py         # DetectionInferenceEngine, visualize_boxes
  eval.py              # compute_iou, compute_precision_recall, evaluate_model
  utils.py             # Token definitions, helpers, parse_boxes_from_text
  create_sample_data.py  # Synthetic dataset generator
train.py               # CLI entry point
infer.py               # Inference pipeline script
```

## Save/Load Format

When LoRA is enabled, saving produces:

- `adapter_config.json` / `adapter_model.safetensors` — LoRA A/B matrices
- `non_llm.pt` — projector, lm_head, embed_tokens weights (non-LoRA trainable params)
- `locany_config.json` — ModelConfig for reconstruction

Loading: creates base model without LoRA, wraps with `PeftModel.from_pretrained`, then loads `non_llm.pt` (with key remapping for PEFT's `base_model.model.` prefix).
