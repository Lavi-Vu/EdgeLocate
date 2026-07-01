# EdgeLocate Architecture

## Overview

EdgeLocate is a vision-language model for referring expression comprehension (REC) and detection, ported from NVIDIA's [LocateAnything (EagleVL)](https://github.com/NVIDIA/LocateAnything). It combines a vision encoder, a small language model, and PBD (Point-Beam-Detection) decoding for high-accuracy object detection at <1B parameters.

## Components

### Vision Encoder (`VisionEncoderWrapper`)

Two variants are supported:

- **SigLIP** (`google/siglip-base-patch16-224`): Standard ViT-based vision encoder. Returns `(B, N_patches, D)` features. Used directly as input to the projector.

- **MoonViT** (`moonshotai/MoonViT-SO-400M`): Hierarchical ViT from Moonshot AI with 2D RoPE and patch merging (27 layers, 1152 dim). Returns multi-scale features that go through a `patch_merger` to produce `(B, N, 4D)` output. Detected automatically via HF config `model_type == "moonvit"`.

- **SigLIP2** (`google/siglip2-base-patch16-224`): Flexible-resolution ViT requiring `pixel_attention_mask` and `spatial_shapes` arguments.

Auto-detection logic:
1. Check `raw.model_type == "moonvit"` → MoonViT path
2. Else → standard SigLIP/SigLIP2 path (based on model name string)

### Projector (`MLPProjector`, `MoonViTProjector`)

Maps vision features to the LLM's embedding space (Qwen2.5-0.5B, 896 dim):

- **MLPProjector**: 2-layer MLP (GELU, no bias). Input dim: `hidden_size → 896`. Used with SigLIP/MoonViT.
- **MoonViTProjector**: 3-layer MLP with LayerNorm and Dropout. Input: hidden_size*4 (merged MoonViT features).

Both projectors cast input to match their parameter dtype at runtime, avoiding dtype mismatches under `nn.DataParallel`.

### Language Model (`Qwen2.5-0.5B`)

The base LLM handles tokenized inputs with special detection tokens (box, ref, coord, null) embedded directly into the vocabulary.

### PBD Decoding (`generate_pbd`)

PBD (Point-Beam-Detection) generates bounding boxes through caption generation with averaging over N hypotheses:

1. Expand each beam output by `avg_window` samples
2. Parse `<bbox>` tokens from each sample
3. Cluster predictions via IoU-based grouping
4. Average boxes within each cluster
5. Average final predictions across clusters if `avg_across_clusters=True`

MTP (Multi-Token Prediction) attention mask can be enabled via `create_mtp_attention_mask()` for predicting N tokens per position.

### Sequence Packing (`PackedDetectionDataset`, `PackedDataCollator`)

Multiple sequences packed into a single training example for efficiency:

- `PackedDetectionDataset`: Appends sequences up to `max_seq_len`, separated by EOS tokens
- `PackedDataCollator`: 1D packing with `pack_1d()` — single `input_ids` tensor per batch, no attention mask needed

## Files

| File | Purpose |
|------|---------|
| `model.py` | Main model: dual VE, projectors, `generate_pbd()`, packed forward |
| `modeling_vit.py` | MoonViT VE: 27-layer ViT with 2D RoPE, patch merge, class-drop |
| `generate_utils.py` | PBD sampling, caption averaging, MTP attention mask |
| `config.py` | Model/Training/Data/Inference configs + CLI args |
| `utils.py` | Token IDs, `setup_tokenizer()`, box parsing, logger |
| `dataset.py` | `DetectionDataset`, `PackedDetectionDataset` |
| `training.py` | Collators, `setup_training()`, VE checkpointing |
| `inference.py` | Inference engine with hybrid VE paths |

## Token Scheme

- `IMAGE`=151665, `BOX_START`=151666, `BOX_END`=151667
- `REF_START`=151668, `REF_END`=151669, `COORD_START`=151670
- `NULL`=152671, `TEXT_MASK`=152672 (8 extra tokens, total 152673)

## Model Config

Key `ModelConfig` fields:

- `model_name`: vision encoder name (SigLIP/MoonViT/SigLIP2)
- `llm_model_name`: LLM name (default: `Qwen/Qwen2.5-0.5B`)
- `use_lora`, `ve_lora_r`: LoRA for LLM and VE
- `attn_implementation`: `sdpa` (default), `flash_attention_2`
- `rope_scaling`: RoPE extension for longer sequences
- `avg_window`: PBD averaging window (default: 3)
- `pattern_h`: PBD pattern columns (default: 6)
- `no_avg_for_ref`: skip averaging for referring expressions
