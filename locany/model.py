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
    """2-layer MLP projector from vision encoder space to LLM space."""

    def __init__(self, ve_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.model = nn.Sequential(
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
      - Any HF-compatible vision encoder (via AutoModel)
    """

    def __init__(self, model_name: str, select_layer: int = -1, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.select_layer = select_layer
        self.dtype = dtype
        self.model_name = model_name
        self._load_encoder()

    def _load_encoder(self):
        name = self.model_name.lower()
        if "siglip2" in name:
            try:
                from transformers import Siglip2VisionModel
                self.encoder = Siglip2VisionModel.from_pretrained(
                    self.model_name, dtype=self.dtype, ignore_mismatched_sizes=True,
                )
            except (ImportError, OSError, ValueError):
                from transformers import AutoModel
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
        self.image_size = getattr(self.encoder.config, "image_size", 224)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.training):
            outputs = self.encoder(pixel_values, output_hidden_states=True)
            if self.select_layer == -1:
                feat = outputs.last_hidden_state
            else:
                feat = outputs.hidden_states[self.select_layer]
            return feat

    @property
    def num_patches(self) -> int:
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
    ) -> torch.LongTensor:
        """Expand labels to match merged sequence length by inserting -100 for visual tokens."""
        if self.image_token_id is None:
            return labels
        ve = self.vision_encoder
        ps = getattr(ve.encoder.config, "patch_size", 16)
        isz = ve.image_size
        num_patches = (isz // ps) ** 2
        num_cls = getattr(ve.encoder.config, "num_cls_tokens", 0)
        if "siglip" in ve.model_name.lower():
            num_cls = 0
        num_vis = num_patches + num_cls
        new_labels_list = []
        for b in range(labels.shape[0]):
            lbl = labels[b]
            ids = input_ids[b]
            new_lbl = []
            for i in range(len(ids)):
                new_lbl.append(lbl[i].item())
                if ids[i] == self.image_token_id:
                    for _ in range(num_vis - 1):
                        new_lbl.append(-100)
            new_labels_list.append(torch.tensor(new_lbl, device=labels.device, dtype=torch.long))
        return torch.stack(new_labels_list)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else True
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if pixel_values is not None and input_ids is not None and self.image_token_id is not None:
            merged_embeds, merged_mask, merged_ids = self.merge_visual_features(
                pixel_values, input_ids, attention_mask,
            )
            if merged_embeds is None:
                merged_embeds = None
                merged_mask = attention_mask
                merged_ids = input_ids
        else:
            merged_embeds = None
            merged_mask = attention_mask
            merged_ids = input_ids

        if labels is not None and pixel_values is not None and self.image_token_id is not None and (input_ids == self.image_token_id).any():
            expanded_labels = self._expand_labels_for_visual(labels, input_ids)
        else:
            expanded_labels = labels

        llm_kwargs = dict(
            attention_mask=merged_mask,
            labels=expanded_labels,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if merged_embeds is not None:
            llm_kwargs["inputs_embeds"] = merged_embeds
        else:
            llm_kwargs["input_ids"] = merged_ids

        outputs = self.llm(**llm_kwargs, **kwargs)

        if not return_dict:
            return outputs
        return CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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


def create_model(config: ModelConfig) -> LocateAnythingForDetection:
    logger.info(f"Creating model with LLM={config.llm_model}, VE={config.ve_model}")
    model = LocateAnythingForDetection(config)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created: {n_total/1e6:.2f}M total, {n_trainable/1e6:.2f}M trainable")
    return model


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
        model.load_state_dict(fixed_state, strict=False)
        logger.info(f"Loaded non-LoRA weights from {non_llm_path}")
    else:
        # Try full model.safetensors (legacy format)
        full_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(full_path):
            from safetensors.torch import load_file
            state_dict = load_file(full_path)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded full model from {full_path}")

    return model
