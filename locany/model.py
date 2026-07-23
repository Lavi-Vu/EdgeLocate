from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
    GenerationConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import ModelConfig
from .utils import logger

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class MLPProjector(nn.Module):
    """3-layer MLP projector with LayerNorm, matching reference LocateAnything3B architecture.
    
    Reference uses: LayerNorm(vit_hidden*4) -> Linear(vit_hidden*4, llm_hidden) -> GELU -> Linear(llm_hidden, llm_hidden)
    For non-MoonViT encoders (no pixel_shuffle), we use: LayerNorm(vit_hidden) -> Linear(vit_hidden, llm_hidden) -> GELU -> Linear(llm_hidden, llm_hidden)
    """

    def __init__(self, ve_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(ve_hidden_size),
            nn.Linear(ve_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class VisionEncoderWrapper(nn.Module):
    """Wrapper for vision encoder models.

    Supports:
      - Google SigLIP / SigLIP2 (via SiglipVisionModel / Siglip2VisionModel)
      - Apple MobileCLIP (via AutoModel with trust_remote_code)
      - Moonshot MoonViT (native-resolution, auto-computes grid_hws)
      - Any HF-compatible vision encoder (via AutoModel)
    """

    def __init__(self, model_name: str, select_layer: int = -1, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.select_layer = select_layer
        self.dtype = dtype
        self.model_name = model_name
        self._load_encoder()
        self._patch_size = getattr(self.encoder.config, "patch_size", 14)

    def _load_encoder(self):
        name = self.model_name.lower()
        if "moonvit" in name:
            import sys as _sys
            try:
                self.encoder = AutoModel.from_pretrained(
                    self.model_name, dtype=self.dtype, trust_remote_code=True,
                )
            except AttributeError as e:
                if "all_tied_weights_keys" not in str(e):
                    raise
                for _mod_name, _mod in _sys.modules.copy().items():
                    if "moonvit" in _mod_name.lower() and hasattr(_mod, "MoonVitPretrainedModel"):
                        if not hasattr(_mod.MoonVitPretrainedModel, 'all_tied_weights_keys'):
                            _mod.MoonVitPretrainedModel.all_tied_weights_keys = {}
                        break
                self.encoder = AutoModel.from_pretrained(
                    self.model_name, dtype=self.dtype, trust_remote_code=True,
                )
            self._fix_moonvit_forward()
        elif "siglip2" in name:
            try:
                from transformers import Siglip2VisionModel
                self.encoder = Siglip2VisionModel.from_pretrained(
                    self.model_name, dtype=self.dtype, ignore_mismatched_sizes=True,
                )
            except (ImportError, OSError, ValueError):
                self.encoder = AutoModel.from_pretrained(
                    self.model_name, dtype=self.dtype, trust_remote_code=True,
                )
        elif "siglip" in name:
            try:
                from transformers import SiglipVisionModel
                self.encoder = SiglipVisionModel.from_pretrained(
                    self.model_name, dtype=self.dtype, ignore_mismatched_sizes=True,
                )
            except (OSError, ValueError):
                self.encoder = AutoModel.from_pretrained(
                    self.model_name, dtype=self.dtype, trust_remote_code=True,
                )
        elif "mobileclip" in name:
            self.encoder = AutoModel.from_pretrained(
                self.model_name, dtype=self.dtype, trust_remote_code=True,
            )
        else:
            self.encoder = AutoModel.from_pretrained(
                self.model_name, dtype=self.dtype, trust_remote_code=True,
            )

        self.hidden_size = self.encoder.config.hidden_size
        if "moonvit" in name:
            grid_dim = getattr(self.encoder.config, "init_pos_emb_height", 64)
            ps = self.encoder.config.patch_size
            self.image_size = grid_dim * ps
        else:
            self.image_size = getattr(self.encoder.config, "image_size", 224)

    def _fix_moonvit_forward(self):
        """Monkey-patch MoonViT's patch_embed to output a single concatenated sequence (B*N, D)."""
        orig_forward = self.encoder.patch_embed.forward

        def patched_forward(self_patch, x, grid_hws):
            x = self_patch.proj(x)
            B, D, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B * H * W, D)
            x = self_patch.pos_emb(x, grid_hws)
            return x

        import types
        self.encoder.patch_embed.forward = types.MethodType(patched_forward, self.encoder.patch_embed)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.training):
            if "moonvit" in self.model_name.lower():
                B = pixel_values.shape[0]
                ps = self._patch_size
                h = pixel_values.shape[2] // ps
                w = pixel_values.shape[3] // ps
                grid_hws = torch.tensor([[h, w]] * B, device=pixel_values.device, dtype=torch.long)
                outputs = self.encoder(pixel_values, grid_hws)
                # MoonViT with patched forward returns (B, num_patches, D) or (B*num_patches, D)
                if isinstance(outputs, (list, tuple)):
                    # Take the last hidden state if list is returned
                    feat = outputs[-1] if outputs else None
                else:
                    feat = outputs
                # Ensure shape is (B, num_patches, D) for projector
                if feat.dim() == 2:
                    feat = feat.unsqueeze(0)
                return feat
            else:
                outputs = self.encoder(pixel_values, output_hidden_states=True)
                if self.select_layer == -1:
                    feat = outputs.last_hidden_state
                else:
                    feat = outputs.hidden_states[self.select_layer]
                return feat

    @property
    def num_patches(self) -> int:
        if "moonvit" in self.model_name.lower():
            ps = self._patch_size
            isz = self.image_size
            return (isz // ps) ** 2
        if "siglip" in self.model_name.lower():
            ps = getattr(self.encoder.config, "patch_size", 16)
            isz = self.image_size
            return (isz // ps) ** 2
        ps = getattr(self.encoder.config, "patch_size", 16)
        isz = self.image_size
        n = (isz // ps) ** 2
        return n + 1 if getattr(self.encoder.config, "num_cls_tokens", 1) > 0 else n

    def get_hidden_size(self) -> int:
        return self.hidden_size


class LocateAnythingForDetection(PreTrainedModel):
    """VLM for detection: Vision Encoder + MLP Projector + LLM (with LoRA).
    
    Box coordinates are generated as discrete tokens (<0>–<1000>) via the LM head.
    """

    def __init__(self, config: ModelConfig):
        hf_config = AutoConfig.from_pretrained(config.llm_model)
        super().__init__(hf_config)
        self.model_config = config

        dtype = DTYPE_MAP[config.torch_dtype]

        self.vision_encoder = VisionEncoderWrapper(
            config.ve_model,
            select_layer=config.vision_select_layer,
            dtype=dtype,
        )

        ve_hidden = self.vision_encoder.get_hidden_size()
        self.projector = MLPProjector(ve_hidden, config.llm_hidden_size).to(dtype=dtype)

        llm_config = AutoConfig.from_pretrained(config.llm_model, trust_remote_code=True)
        llm_config.tie_word_embeddings = config.use_lora and not config.freeze_llm
        llm_kwargs = dict(config=llm_config, dtype=dtype, attn_implementation=config.attn_implementation)
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_model, **llm_kwargs)

        self.image_token_id = None

        self._apply_freezing(config)
        self._apply_lora(config)

        if config.use_lora and not config.freeze_llm:
            self.llm.tie_weights = lambda: None

    def _apply_freezing(self, config: ModelConfig):
        if config.freeze_vision_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            logger.info("Froze vision encoder")
        if config.freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
            logger.info("Froze LLM backbone")
        else:
            logger.info("LLM backbone is trainable")
        if config.freeze_mlp:
            for p in self.projector.parameters():
                p.requires_grad = False
            logger.info("Froze MLP projector")
        else:
            logger.info("MLP projector is trainable")

    def _apply_lora(self, config: ModelConfig):
        if not config.use_lora or config.freeze_llm:
            return
        try:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                'gate_proj', 'down_proj', 'up_proj'],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.enable_input_require_grads()
            for name, param in self.llm.named_parameters():
                if "lm_head" in name or "embed_tokens" in name:
                    param.requires_grad = True
            self.llm.print_trainable_parameters()
            logger.info(f"Applied LoRA (r={config.lora_r}, alpha={config.lora_alpha})")
        except ImportError:
            logger.warning("PEFT not installed, skipping LoRA")

    def set_image_token_id(self, tokenizer):
        from .utils import SPECIAL_TOKENS
        self.image_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["image"])
        logger.info(f"Set image_token_id={self.image_token_id}")

    def merge_visual_features(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor]:
        """Replace image token positions with projected visual features."""
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        if self.image_token_id is None:
            return None, attention_mask, input_ids

        vis_feats = self.vision_encoder(pixel_values)
        vis_feats = self.projector(vis_feats)
        num_image_tokens = vis_feats.shape[1]

        img_positions = (input_ids == self.image_token_id).nonzero(as_tuple=False)
        batch_indices = img_positions[:, 0]
        seq_indices = img_positions[:, 1]

        text_embeds = self.llm.get_input_embeddings()(input_ids)

        new_embeds_list = []
        new_mask_list = []
        new_ids_list = []

        for b in range(batch_size):
            mask = batch_indices == b
            if mask.any():
                img_pos = seq_indices[mask][0].item()
                before = text_embeds[b, :img_pos]
                after = text_embeds[b, img_pos + 1:]
                new_embeds = torch.cat([before, vis_feats[b], after], dim=0)

                before_ids = input_ids[b, :img_pos]
                vis_ids = torch.full((num_image_tokens,), self.image_token_id,
                                     device=device, dtype=torch.long)
                after_ids = input_ids[b, img_pos + 1:]
                new_ids = torch.cat([before_ids, vis_ids, after_ids], dim=0)

                before_mask = attention_mask[b, :img_pos]
                vis_mask = torch.ones(num_image_tokens, device=device, dtype=attention_mask.dtype)
                after_mask = attention_mask[b, img_pos + 1:]
                new_mask = torch.cat([before_mask, vis_mask, after_mask], dim=0)
            else:
                new_embeds = text_embeds[b]
                new_ids = input_ids[b]
                new_mask = attention_mask[b]

            new_embeds_list.append(new_embeds)
            new_ids_list.append(new_ids)
            new_mask_list.append(new_mask)

        max_len = max(e.shape[0] for e in new_embeds_list)
        padded_embeds = []
        padded_ids = []
        padded_mask = []

        for emb, ids, m in zip(new_embeds_list, new_ids_list, new_mask_list):
            pad_len = max_len - emb.shape[0]
            if pad_len > 0:
                pad_e = torch.zeros(pad_len, emb.shape[1], device=device, dtype=emb.dtype)
                padded_embeds.append(torch.cat([emb, pad_e], dim=0))
                pad_ids = torch.zeros(pad_len, device=device, dtype=torch.long)
                padded_ids.append(torch.cat([ids, pad_ids], dim=0))
                pad_m = torch.zeros(pad_len, device=device, dtype=m.dtype)
                padded_mask.append(torch.cat([m, pad_m], dim=0))
            else:
                padded_embeds.append(emb[:max_len])
                padded_ids.append(ids[:max_len])
                padded_mask.append(m[:max_len])

        return torch.stack(padded_embeds), torch.stack(padded_mask), torch.stack(padded_ids)

    def _expand_labels_for_visual(
        self,
        labels: torch.LongTensor,
        input_ids: torch.LongTensor,
        num_visual_tokens: int,
    ) -> torch.LongTensor:
        """Expand labels to match merged sequence length by inserting -100 for visual tokens.
        
        Args:
            labels: Original labels tensor (B, seq_len)
            input_ids: Original input_ids tensor (B, seq_len)
            num_visual_tokens: Actual number of visual tokens from the vision encoder output
        """
        if self.image_token_id is None:
            return labels

        new_labels_list = []
        for b in range(labels.shape[0]):
            lbl = labels[b]
            ids = input_ids[b]
            new_lbl = []
            for i in range(len(ids)):
                new_lbl.append(lbl[i].item())
                if ids[i] == self.image_token_id:
                    for _ in range(num_visual_tokens - 1):
                        new_lbl.append(-100)
            new_labels_list.append(torch.tensor(new_lbl, device=labels.device, dtype=torch.long))
        return torch.stack(new_labels_list)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward pass that bypasses PEFT/CausalLM wrappers for gradient flow.

        Reference (Eagle/Embodied) calls the inner transformer model directly and
        computes cross-entropy loss manually. This avoids gradient flow issues when
        PEFT's CausalLM wrapper receives inputs_embeds.

        Unlike the reference (which replaces image tokens IN-PLACE at constant seq
        length), EdgeLocate expands 1 <|image|> token to N visual tokens, growing
        the sequence. We handle this expansion here with proper label/mask/position
        expansion.
        """
        return_dict = return_dict if return_dict is not None else True
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        IGNORE_INDEX = -100

        has_images = (pixel_values is not None and input_ids is not None
                      and self.image_token_id is not None)
        B, orig_len = input_ids.shape

        # --- Extract and project visual features ---
        num_visual_tokens = 0
        vis_feats = None
        if has_images:
            vis_feats = self.vision_encoder(pixel_values)
            vis_feats = self.projector(vis_feats)
            num_visual_tokens = vis_feats.shape[1]

        # --- Build expanded input_embeds, labels, attention_mask ---
        if has_images and num_visual_tokens > 0:
            input_embeds, attention_mask, expanded_labels = self._expand_sequence(
                input_ids, attention_mask, labels, vis_feats, num_visual_tokens,
            )
        else:
            input_embeds = self.llm.get_input_embeddings()(input_ids)
            expanded_labels = labels

        # --- Build position ids ---
        max_len = input_embeds.shape[1]
        if position_ids is None:
            position_ids = torch.arange(max_len, device=input_ids.device).unsqueeze(0).expand(B, -1)

        # --- Call inner transformer directly (bypass PEFT + CausalLM wrappers) ---
        if hasattr(self.llm, 'base_model') and hasattr(self.llm.base_model, 'model'):
            inner_model = self.llm.base_model.model.model  # Qwen2Model
        else:
            inner_model = self.llm.model  # Qwen2Model

        outputs = inner_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state

        # --- Compute loss manually via lm_head projection ---
        if expanded_labels is not None:
            if hasattr(self.llm, 'base_model') and hasattr(self.llm.base_model, 'model'):
                lm_head = self.llm.base_model.model.lm_head
            else:
                lm_head = self.llm.lm_head

            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = expanded_labels[..., 1:].contiguous()
            shift_hidden = shift_hidden.view(-1, shift_hidden.shape[-1])
            shift_labels = shift_labels.view(-1)

            loss = torch.nn.functional.cross_entropy(
                lm_head(shift_hidden), shift_labels,
                ignore_index=IGNORE_INDEX, reduction='mean',
            )
        else:
            loss = None

        if not return_dict:
            return (loss,) if loss is not None else (hidden_states,)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )

    def _expand_sequence(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        labels: torch.LongTensor,
        vis_feats: torch.Tensor,
        num_visual_tokens: int,
    ) -> Tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor]:
        """Expand single <|image|> token into N visual tokens per sample.

        For each sample in the batch, finds the <|image|> position, replaces it
        with the projected visual features (N tokens), and pads all samples to
        the same length. Labels at visual token positions are set to -100.
        """
        IGNORE_INDEX = -100
        B, orig_len, C = input_ids.shape
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        new_embeds_list = []
        new_mask_list = []
        new_labels_list = []

        for b in range(B):
            ids_b = input_ids[b]
            mask_b = attention_mask[b]
            labels_b = labels[b] if labels is not None else None

            img_pos = (ids_b == self.image_token_id).nonzero(as_tuple=False)
            if img_pos.numel() == 0:
                new_embeds_list.append(text_embeds[b])
                new_mask_list.append(mask_b)
                if labels_b is not None:
                    new_labels_list.append(labels_b)
                continue

            img_pos_idx = img_pos[0].item()
            before = text_embeds[b, :img_pos_idx]
            after = text_embeds[b, img_pos_idx + 1:]
            new_embeds_list.append(torch.cat([before, vis_feats[b], after], dim=0))

            before_mask = mask_b[:img_pos_idx]
            vis_mask = torch.ones(num_visual_tokens, device=mask_b.device, dtype=mask_b.dtype)
            after_mask = mask_b[img_pos_idx + 1:]
            new_mask_list.append(torch.cat([before_mask, vis_mask, after_mask], dim=0))

            if labels_b is not None:
                before_lbl = labels_b[:img_pos_idx]
                vis_lbl = torch.full((num_visual_tokens,), IGNORE_INDEX,
                                     device=labels_b.device, dtype=labels_b.dtype)
                after_lbl = labels_b[img_pos_idx + 1:]
                new_labels_list.append(torch.cat([before_lbl, vis_lbl, after_lbl], dim=0))

        max_len = max(e.shape[0] for e in new_embeds_list)

        padded_embeds = []
        padded_mask = []
        padded_labels = []

        for i in range(len(new_embeds_list)):
            emb = new_embeds_list[i]
            pad_len = max_len - emb.shape[0]
            if pad_len > 0:
                padded_embeds.append(torch.cat([
                    emb, torch.zeros(pad_len, C, device=emb.device, dtype=emb.dtype)
                ], dim=0))
            else:
                padded_embeds.append(emb)

            m = new_mask_list[i]
            if pad_len > 0:
                padded_mask.append(torch.cat([
                    m, torch.zeros(pad_len, device=m.device, dtype=m.dtype)
                ], dim=0))
            else:
                padded_mask.append(m)

            if i < len(new_labels_list):
                l = new_labels_list[i]
                if pad_len > 0:
                    padded_labels.append(torch.cat([
                        l, torch.full((pad_len,), IGNORE_INDEX, device=l.device, dtype=l.dtype)
                    ], dim=0))
                else:
                    padded_labels.append(l)

        input_embeds = torch.stack(padded_embeds)
        attention_mask = torch.stack(padded_mask)
        labels_out = torch.stack(padded_labels) if padded_labels else labels

        return input_embeds, attention_mask, labels_out

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """Standard autoregressive generation with visual features."""
        if pixel_values is not None and input_ids is not None and self.image_token_id is not None:
            merged_embeds, merged_mask, _ = self.merge_visual_features(
                pixel_values, input_ids, attention_mask,
            )
            inputs = dict(inputs_embeds=merged_embeds, attention_mask=merged_mask)
        else:
            inputs = dict(input_ids=input_ids, attention_mask=attention_mask)

        if 'use_cache' not in generate_kwargs:
            generate_kwargs['use_cache'] = True

        outputs = self.llm.generate(
            generation_config=generation_config,
            **inputs,
            **generate_kwargs,
        )
        return outputs

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llm.set_output_embeddings(new_embeddings)

    def tie_weights(self):
        pass

    def get_trainable_params(self) -> Dict[str, nn.Parameter]:
        params = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                params[name] = p
        return params

    def check_gradient_flow(self) -> Dict[str, bool]:
        """Check that gradients flow through projector and inner model after backward."""
        result = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                result[name] = p.grad is not None and p.grad.abs().sum().item() > 0
        return result


def create_model(config: ModelConfig) -> LocateAnythingForDetection:
    logger.info(f"Creating model with LLM={config.llm_model}, VE={config.ve_model}")
    model = LocateAnythingForDetection(config)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created: {n_total/1e6:.2f}M total, {n_trainable/1e6:.2f}M trainable")
    return model


def _safe_load_state_dict(model, state_dict: dict, label: str = ""):
    """Load state_dict, skipping size-mismatched keys with a warning."""
    model_state = model.state_dict()
    to_load = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state:
            if isinstance(v, torch.Tensor) and v.shape == model_state[k].shape:
                to_load[k] = v
            elif isinstance(v, torch.Tensor):
                skipped.append(f"{k}: checkpoint {list(v.shape)} vs model {list(model_state[k].shape)}")
            else:
                to_load[k] = v
        else:
            to_load[k] = v
    if skipped:
        logger.warning(f"Skipped {len(skipped)} size-mismatched keys {label}:")
        for s in skipped:
            logger.warning(f"  {s}")
    model.load_state_dict(to_load, strict=False)


def load_model_from_dir(model_dir: str, tokenizer) -> LocateAnythingForDetection:
    """Load a saved model from directory.

    Supports both LoRA adapter + non-LLM weights and full merged model formats.
    """
    import json
    import os
    import torch

    # Try to load saved config
    cfg_path = os.path.join(model_dir, "locany_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg_dict = json.load(f)
        model_cfg = ModelConfig.from_dict(cfg_dict)
    else:
        model_cfg = ModelConfig()

    # Create model WITHOUT LoRA (we'll load it from the adapter)
    model_cfg.use_lora = False
    model = create_model(model_cfg)
    old_vocab = model.llm.get_input_embeddings().weight.shape[0]
    new_vocab = len(tokenizer)
    if new_vocab > old_vocab:
        model.llm.resize_token_embeddings(new_vocab, mean_resizing=False)
    model.image_token_id = tokenizer.convert_tokens_to_ids("<|image|>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load LoRA adapter if present
    adapter_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(adapter_path):
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, model_dir)
        model = model.to(device)
        logger.info(f"Loaded LoRA adapter from {model_dir}")

    # Load non-LoRA weights (projector, possibly lm_head, embeddings)
    non_llm_path = os.path.join(model_dir, "non_llm.pt")
    if os.path.exists(non_llm_path):
        other_state = torch.load(non_llm_path, map_location=device)
        fixed_state = {}
        for k, v in other_state.items():
            if k.startswith("llm.base_model.model."):
                fixed_state[k.replace("llm.base_model.model.", "llm.")] = v
            else:
                fixed_state[k] = v
        _safe_load_state_dict(model, fixed_state, label=f"from {non_llm_path}")
        logger.info(f"Loaded non-LoRA weights from {non_llm_path}")
    else:
        # Try full model.safetensors (legacy format)
        full_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(full_path):
            from safetensors.torch import load_file
            state_dict = load_file(full_path)
            _safe_load_state_dict(model, state_dict, label=f"from {full_path}")
            logger.info(f"Loaded full model from {full_path}")

    return model
