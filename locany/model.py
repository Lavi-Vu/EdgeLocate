from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig, AutoModel, AutoModelForCausalLM,
    PreTrainedModel, GenerationConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import ModelConfig
from .utils import logger
from .generate_utils import get_token_ids_from_config, sample_tokens, handle_pattern

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class MLPProjector(nn.Module):
    def __init__(self, ve_hidden_size: int, llm_hidden_size: int, num_layers: int = 2):
        super().__init__()
        if num_layers == 2:
            self.model = nn.Sequential(
                nn.Linear(ve_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
        else:
            layers = [nn.LayerNorm(ve_hidden_size), nn.Linear(ve_hidden_size, llm_hidden_size), nn.GELU()]
            for _ in range(num_layers - 1):
                layers.extend([nn.Linear(llm_hidden_size, llm_hidden_size), nn.GELU()])
            self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MoonViTProjector(nn.Module):
    """MLP projector for MoonViT with LayerNorm + 4x channel handling (patch merge)."""

    def __init__(self, vit_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * 4),
            nn.Linear(vit_hidden_size * 4, llm_hidden_size),
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
    def __init__(self, model_name: str, select_layer: int = -1, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.select_layer = select_layer
        self.dtype = dtype
        self.model_name = model_name
        self.is_moonvit = "moonvit" in model_name.lower()
        if not self.is_moonvit:
            try:
                from .modeling_vit import MoonViTConfig
                cfg_check = MoonViTConfig.from_pretrained(model_name)
                self.is_moonvit = hasattr(cfg_check, 'merge_kernel_size')
            except Exception:
                pass
        if self.is_moonvit:
            self._load_moonvit()
        else:
            self._load_standard_encoder()
        self._patch_size = getattr(self.encoder.config, "patch_size", 14)

    def _load_moonvit(self):
        from .modeling_vit import MoonVitPretrainedModel, MoonViTConfig
        config = MoonViTConfig.from_pretrained(self.model_name)
        config._attn_implementation = 'sdpa'
        self.encoder = MoonVitPretrainedModel(config)
        self.hidden_size = config.hidden_size
        self.image_size = getattr(config, 'init_pos_emb_height', 64) * config.patch_size
        self.merge_kernel_size = config.merge_kernel_size

    def _load_standard_encoder(self):
        name = self.model_name.lower()
        if "siglip2" in name:
            try:
                from transformers import Siglip2VisionModel
                self.encoder = Siglip2VisionModel.from_pretrained(self.model_name, dtype=self.dtype, ignore_mismatched_sizes=True)
            except (ImportError, OSError, ValueError):
                self.encoder = AutoModel.from_pretrained(self.model_name, dtype=self.dtype, trust_remote_code=True)
        elif "siglip" in name:
            try:
                from transformers import SiglipVisionModel
                self.encoder = SiglipVisionModel.from_pretrained(self.model_name, dtype=self.dtype, ignore_mismatched_sizes=True)
            except (OSError, ValueError):
                self.encoder = AutoModel.from_pretrained(self.model_name, dtype=self.dtype, trust_remote_code=True)
        elif "mobileclip" in name:
            self.encoder = AutoModel.from_pretrained(self.model_name, dtype=self.dtype, trust_remote_code=True)
        else:
            self.encoder = AutoModel.from_pretrained(self.model_name, dtype=self.dtype, trust_remote_code=True)
        self.hidden_size = self.encoder.config.hidden_size
        self.image_size = getattr(self.encoder.config, "image_size", 224)

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.is_moonvit:
            grid_hws = kwargs.get('grid_hws')
            if grid_hws is None:
                B = pixel_values.shape[0]
                ps = self._patch_size
                h = pixel_values.shape[2] // ps
                w = pixel_values.shape[3] // ps
                grid_hws = torch.tensor([[h, w]] * B, device=pixel_values.device, dtype=torch.long)
            outputs = self.encoder(pixel_values, grid_hws=grid_hws)
            if isinstance(outputs, list):
                if len(outputs) == 1:
                    outputs = outputs[0].unsqueeze(0)
                else:
                    outputs = torch.stack(outputs, dim=0)
            return outputs
        else:
            outputs = self.encoder(pixel_values, output_hidden_states=True)
            if self.select_layer == -1:
                return outputs.last_hidden_state
            return outputs.hidden_states[self.select_layer]

    def get_hidden_size(self) -> int:
        return self.hidden_size

    def get_num_patches(self, img_h: int, img_w: int) -> int:
        if self.is_moonvit:
            ps = self._patch_size
            kh, kw = self.merge_kernel_size
            return (img_h // ps // kh) * (img_w // ps // kw)
        ps = getattr(self.encoder.config, "patch_size", 16)
        return (img_h // ps) * (img_w // ps)


def create_mtp_attention_mask(context_len: int, block_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create attention mask for MTP: causal on context, non-causal within the prediction block."""
    total_len = context_len + block_size
    mask = torch.full((1, 1, total_len, total_len), float('-inf'), device=device, dtype=dtype)
    for i in range(total_len):
        if i < context_len:
            mask[0, 0, i, :i + 1] = 0.0
        else:
            mask[0, 0, i, :context_len] = 0.0
            mask[0, 0, i, context_len:] = 0.0
    return mask


class LocateAnythingForDetection(PreTrainedModel):
    def __init__(self, config: ModelConfig):
        hf_config = AutoConfig.from_pretrained(config.llm_model)
        super().__init__(hf_config)
        self.model_config = config
        dtype = DTYPE_MAP[config.torch_dtype]

        self.vision_encoder = VisionEncoderWrapper(
            config.ve_model, select_layer=config.vision_select_layer, dtype=dtype,
        )

        ve_hidden = self.vision_encoder.get_hidden_size()
        self.is_moonvit = self.vision_encoder.is_moonvit

        if self.is_moonvit:
            self.projector = MoonViTProjector(ve_hidden, config.llm_hidden_size).to(dtype=dtype)
        else:
            self.projector = MLPProjector(ve_hidden, config.llm_hidden_size,
                                          num_layers=config.mlp_connector_layers).to(dtype=dtype)

        llm_config = AutoConfig.from_pretrained(config.llm_model, trust_remote_code=True)
        llm_config.tie_word_embeddings = config.use_lora and not config.freeze_llm
        llm_kwargs = dict(config=llm_config, dtype=dtype, attn_implementation=config.attn_implementation)
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_model, **llm_kwargs)

        self.image_token_id = None
        self.token_ids = None

        self._apply_freezing(config)
        self._apply_lora(config)
        self._apply_backbone_lora(config)

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
        if config.freeze_mlp:
            for p in self.projector.parameters():
                p.requires_grad = False
            logger.info("Froze MLP projector")

    def _apply_lora(self, config: ModelConfig):
        if not config.use_lora or config.freeze_llm:
            return
        try:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                'gate_proj', 'down_proj', 'up_proj'],
                bias="none", task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.enable_input_require_grads()
            for name, param in self.llm.named_parameters():
                if "lm_head" in name or "embed_tokens" in name:
                    param.requires_grad = True
            self.llm.print_trainable_parameters()
            logger.info(f"Applied LLM LoRA (r={config.lora_r})")
        except ImportError:
            logger.warning("PEFT not installed, skipping LoRA")

    def _apply_backbone_lora(self, config: ModelConfig):
        if not config.use_backbone_lora or config.freeze_vision_encoder:
            return
        try:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora, lora_dropout=0.05,
                target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj',
                                'mlp.fc1', 'mlp.fc2'],
                bias="none",
            )
            self.vision_encoder.encoder = get_peft_model(self.vision_encoder.encoder, lora_config)
            self.vision_encoder.encoder.print_trainable_parameters()
            logger.info(f"Applied backbone LoRA (r={config.use_backbone_lora})")
        except ImportError:
            logger.warning("PEFT not installed, skipping backbone LoRA")

    def set_image_token_id(self, tokenizer):
        from .utils import SPECIAL_TOKENS
        self.image_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["image"])
        self.token_ids = get_token_ids_from_config(self.model_config)
        logger.info(f"Set image_token_id={self.image_token_id}")

    def extract_visual_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vis_feats = self.vision_encoder(pixel_values)
        vis_feats = self.projector(vis_feats)
        return vis_feats

    def merge_visual_features(
        self, pixel_values: torch.Tensor, input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor]:
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        if self.image_token_id is None:
            return None, attention_mask, input_ids

        vis_feats = self.extract_visual_features(pixel_values)
        num_image_tokens = vis_feats.shape[1] if vis_feats.dim() == 3 else 1
        if vis_feats.dim() == 2:
            vis_feats = vis_feats.unsqueeze(0)
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

    def _expand_labels_for_visual(self, labels: torch.LongTensor, input_ids: torch.LongTensor,
                                   n_vis_tokens: Optional[int] = None) -> torch.LongTensor:
        if self.image_token_id is None:
            return labels
        if n_vis_tokens is None:
            ve = self.vision_encoder
            ps = ve._patch_size
            if self.is_moonvit:
                kh, kw = ve.merge_kernel_size
                isz = ve.image_size
                num_vis = (isz // ps // kh) * (isz // ps // kw)
            else:
                num_vis = (ve.image_size // ps) ** 2
        else:
            num_vis = n_vis_tokens

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

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None,
                labels=None, output_hidden_states=None, return_dict=None,
                position_ids=None, sub_sample_lengths=None, **kwargs):
        return_dict = return_dict if return_dict is not None else True
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        is_packed = sub_sample_lengths is not None

        if not is_packed and pixel_values is not None and input_ids is not None and self.image_token_id is not None:
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

        if labels is not None and not is_packed and pixel_values is not None and self.image_token_id is not None and (input_ids == self.image_token_id).any():
            if self.is_moonvit and pixel_values is not None:
                ps = self.vision_encoder._patch_size
                kh, kw = self.vision_encoder.merge_kernel_size
                actual_h = pixel_values.shape[2] // ps // kh
                actual_w = pixel_values.shape[3] // ps // kw
                n_vis = actual_h * actual_w
            else:
                n_vis = None
            expanded_labels = self._expand_labels_for_visual(labels, input_ids, n_vis_tokens=n_vis)
        else:
            expanded_labels = labels

        llm_kwargs = dict(
            labels=expanded_labels,
            output_hidden_states=output_hidden_states, return_dict=True,
        )

        if is_packed:
            llm_kwargs["input_ids"] = input_ids
            llm_kwargs["position_ids"] = position_ids
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            causal_mask = torch.full((1, 1, seq_len, seq_len), float('-inf'), device=device, dtype=merged_embeds.dtype if merged_embeds is not None else torch.float32)
            cumsum = sub_sample_lengths.cumsum(dim=1)
            prev = 0
            for b in range(batch_size):
                for i in range(sub_sample_lengths.shape[1]):
                    end = cumsum[b, i].item()
                    causal_mask[b, 0, prev:end, :end] = 0.0
                    for j in range(prev, end):
                        causal_mask[b, 0, j, :j + 1] = 0.0
                    prev = end
            llm_kwargs["attention_mask"] = causal_mask
        else:
            if merged_embeds is not None:
                llm_kwargs["inputs_embeds"] = merged_embeds
            else:
                llm_kwargs["input_ids"] = merged_ids
            llm_kwargs["attention_mask"] = merged_mask

        outputs = self.llm(**llm_kwargs, **kwargs)

        if not return_dict:
            return outputs
        return CausalLMOutputWithPast(
            loss=outputs.loss, logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(self, pixel_values=None, input_ids=None, attention_mask=None,
                 generation_config=None, **generate_kwargs):
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
            generation_config=generation_config, **inputs, **generate_kwargs,
        )
        return outputs

    def generate_pbd(self, pixel_values=None, input_ids=None, attention_mask=None,
                      tokenizer=None, generation_mode='hybrid', max_new_tokens=512,
                      temperature=0.0, top_p=1.0, block_size=6, keep_k_avg=4, verbose=False):
        """Parallel Box Decoding (PBD) generation with MTP. Supports hybrid/fast/slow modes."""
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        assert batch_size == 1, "PBD only supports batch_size=1"

        vis_feats = self.extract_visual_features(pixel_values)
        if vis_feats.dim() == 2:
            vis_feats = vis_feats.unsqueeze(0)

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        img_pos = (input_ids[0] == self.image_token_id).nonzero(as_tuple=False)
        if len(img_pos) > 0:
            img_pos = img_pos[0].item()
            merged_embeds = torch.cat([text_embeds[:, :img_pos], vis_feats, text_embeds[:, img_pos+1:]], dim=1)
            merged_mask = torch.ones(1, merged_embeds.shape[1], device=device, dtype=attention_mask.dtype)
            context_len = merged_embeds.shape[1]
        else:
            merged_embeds = text_embeds
            merged_mask = attention_mask
            context_len = seq_len

        generated = input_ids.clone()
        use_mtp = generation_mode in ('fast', 'hybrid')
        tok_ids = self.token_ids or get_token_ids_from_config(self.model_config)
        im_end_token_id = tok_ids['im_end_token_id']
        box_end_token_id = tok_ids['box_end_token_id']

        full_pos_ids = torch.arange(0, context_len + max_new_tokens + block_size, device=device).unsqueeze(0)

        gen_len = 0
        cur_embeds = merged_embeds
        cur_mask = merged_mask

        while gen_len < max_new_tokens:
            if use_mtp:
                ctx_len = cur_embeds.shape[1]
                mask_emb = self.llm.get_input_embeddings()(
                    torch.tensor([[0]], device=device)
                ).expand(-1, block_size, -1)
                full_embeds = torch.cat([cur_embeds, mask_emb], dim=1)
                pos_ids = full_pos_ids[:, :ctx_len + block_size].clone()
                pos_ids[0, ctx_len:] = ctx_len - 1
                attn_mask = create_mtp_attention_mask(ctx_len, block_size, device, cur_embeds.dtype)

                with torch.no_grad():
                    outputs = self.llm(
                        inputs_embeds=full_embeds, attention_mask=attn_mask,
                        position_ids=pos_ids, use_cache=False, return_dict=True,
                    )

                next_logits = outputs.logits[:, -block_size:, :]
                _, _, x0, box_avg = sample_tokens(
                    next_logits, generated, tok_ids,
                    temperature=temperature, top_p=top_p,
                    keep_k_avg=keep_k_avg, generation_mode=generation_mode,
                )
                is_box_empty = (box_avg[0] == 0).all()
                new_tokens = x0[0] if is_box_empty else box_avg[0]
                out = handle_pattern(new_tokens, tok_ids, generation_mode)

                if out['type'] == 'im_end':
                    break

                is_error = (out['type'] == 'error_box')
                if is_error and generation_mode == 'hybrid':
                    use_mtp = False

                out_tokens = out['tokens']
                out_ids = torch.tensor([out_tokens], device=device)
                gen_len += out_ids.shape[1]
                cur_embeds = torch.cat([cur_embeds, self.llm.get_input_embeddings()(out_ids)], dim=1)
                cur_mask = torch.cat([cur_mask, torch.ones(1, out_ids.shape[1], device=device)], dim=1)
                generated = torch.cat([generated, out_ids], dim=1)
            else:
                ctx_len = cur_embeds.shape[1]
                last_emb = cur_embeds[:, -1:, :]
                pos_ids = full_pos_ids[:, ctx_len - 1:ctx_len]
                with torch.no_grad():
                    outputs = self.llm(
                        inputs_embeds=last_emb, attention_mask=cur_mask,
                        position_ids=pos_ids, use_cache=False, return_dict=True,
                    )

                next_logits = outputs.logits[:, -1:, :]
                _, _, x0, _ = sample_tokens(
                    next_logits, generated, tok_ids,
                    temperature=temperature, top_p=top_p,
                )
                next_token = x0[0]
                next_id = next_token[0].item()

                if next_id == im_end_token_id:
                    generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                    break

                gen_len += 1
                next_emb = self.llm.get_input_embeddings()(next_token.unsqueeze(0))
                cur_embeds = torch.cat([cur_embeds, next_emb], dim=1)
                cur_mask = torch.cat([cur_mask, torch.ones(1, 1, device=device)], dim=1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

                if generation_mode == 'hybrid' and next_id == box_end_token_id:
                    use_mtp = True

        return generated[:, seq_len:]

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
        return {name: p for name, p in self.named_parameters() if p.requires_grad}


def create_model(config: ModelConfig) -> LocateAnythingForDetection:
    logger.info(f"Creating model with LLM={config.llm_model}, VE={config.ve_model}")
    model = LocateAnythingForDetection(config)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created: {n_total/1e6:.2f}M total, {n_trainable/1e6:.2f}M trainable")
    return model


def _safe_load_state_dict(model, state_dict: dict, label: str = ""):
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
    import json, os, torch
    cfg_path = os.path.join(model_dir, "locany_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg_dict = json.load(f)
        model_cfg = ModelConfig.from_dict(cfg_dict)
    else:
        model_cfg = ModelConfig()

    model_cfg.use_lora = False
    model = create_model(model_cfg)
    old_vocab = model.llm.get_input_embeddings().weight.shape[0]
    new_vocab = len(tokenizer)
    if new_vocab > old_vocab:
        model.llm.resize_token_embeddings(new_vocab, mean_resizing=False)
    model.image_token_id = tokenizer.convert_tokens_to_ids("<|image|>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(adapter_path):
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, model_dir)
        model = model.to(device)
        logger.info(f"Loaded LoRA adapter from {model_dir}")

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
        full_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(full_path):
            from safetensors.torch import load_file
            state_dict = load_file(full_path)
            _safe_load_state_dict(model, state_dict, label=f"from {full_path}")
            logger.info(f"Loaded full model from {full_path}")

    return model
