import torch
import torch.nn.functional as F
import torch.distributions as dists
from typing import Dict, Optional, List, Tuple


def get_token_ids_from_config(config) -> Dict[str, int]:
    token_ids = {}
    token_ids['box_start_token_id'] = getattr(config, 'box_start_token_id', 151666)
    token_ids['box_end_token_id'] = getattr(config, 'box_end_token_id', 151667)
    token_ids['coord_start_token_id'] = getattr(config, 'coord_start_token_id', 151670)
    token_ids['coord_end_token_id'] = getattr(config, 'coord_end_token_id', 152670)
    token_ids['ref_start_token_id'] = getattr(config, 'ref_start_token_id', 151668)
    token_ids['ref_end_token_id'] = getattr(config, 'ref_end_token_id', 151669)
    token_ids['none_token_id'] = getattr(config, 'none_token_id', 4064)
    token_ids['null_token_id'] = getattr(config, 'null_token_id', 152671)
    token_ids['im_end_token_id'] = getattr(config, 'im_end_token_id', 151645)
    return token_ids


def top_p_logits(logits: torch.Tensor, top_p: float = None) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def top_k_logits(logits: torch.Tensor, top_k: int = None) -> torch.Tensor:
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)


def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, repetition_penalty: float = 1.0) -> torch.Tensor:
    if repetition_penalty == 1.0:
        return logits
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)
        squeeze_back = True
    else:
        squeeze_back = False
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device
    token_mask = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=device)
    for b in range(batch_size):
        unique_tokens = input_ids[b].unique()
        valid_tokens = unique_tokens[(unique_tokens >= 0) & (unique_tokens < vocab_size)]
        if valid_tokens.numel() > 0:
            token_mask[b, valid_tokens] = True
    token_mask = token_mask.unsqueeze(1).expand(-1, seq_len, -1)
    positive = logits > 0
    negative = ~positive
    logits = torch.where(token_mask & positive, logits / repetition_penalty, logits)
    logits = torch.where(token_mask & negative, logits * repetition_penalty, logits)
    if squeeze_back:
        logits = logits.squeeze(1)
    return logits


def sample_tokens(logits, generated, token_ids, **generate_kwargs):
    batch_size, seq_len, vocab_size = logits.shape
    repetition_penalty = generate_kwargs.get('repetition_penalty', 1.0)
    temperature = generate_kwargs.get('temperature', 0)
    top_p = generate_kwargs.get('top_p', None)
    top_k_val = generate_kwargs.get('top_k', None)

    if repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated, repetition_penalty)
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k_val is not None:
        logits = top_k_logits(logits, top_k_val)

    probs = torch.softmax(logits, dim=-1)
    if temperature > 0:
        x0 = dists.Categorical(probs=probs).sample()
        confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if seq_len == 1:
        return probs, confidence, x0, None

    box_avg = []
    fallback_box = torch.zeros(1, dtype=x0.dtype, device=x0.device)
    for b in range(batch_size):
        decoded_box = decode_bbox_avg(
            logits[b], probs[b], token_ids,
            keep_k=generate_kwargs.get('keep_k_avg', 4),
            generation_mode=generate_kwargs.get('generation_mode', 'hybrid'),
        )
        if decoded_box is not None:
            box_avg.append(decoded_box)
        else:
            box_avg.append(fallback_box)
    box_avg = torch.stack(box_avg)
    return probs, confidence, x0, box_avg


def is_valid_box_frame(probs, token_ids, start_thresh=0.6, end_thresh=0.2, topk=5):
    box_start_token_id = token_ids['box_start_token_id']
    box_end_token_id = token_ids['box_end_token_id']
    null_token_id = token_ids['null_token_id']
    im_end_token_id = token_ids['im_end_token_id']
    none_token_id = token_ids['none_token_id']

    p_start = probs[0, box_start_token_id]
    if p_start >= start_thresh:
        if (probs[1, none_token_id] > 0.2 and
            probs[2, box_end_token_id] > 0.2 and
            probs[3, null_token_id] > 0.1 and
            probs[4, null_token_id] > 0.1):
            return 'empty_box'

    end_target_ids = torch.tensor([box_end_token_id, null_token_id, im_end_token_id], device=probs.device)
    if probs[5, end_target_ids].sum() >= end_thresh:
        return 'legal_box'
    return 'illegal_box'


def decode_bbox_avg(logits, probs, token_ids, keep_k=5, start_thresh=0.7, end_thresh=0.2, generation_mode='hybrid'):
    coord_start_token_id = token_ids['coord_start_token_id']
    coord_end_token_id = token_ids['coord_end_token_id']
    box_start_token_id = token_ids['box_start_token_id']
    box_end_token_id = token_ids['box_end_token_id']
    none_token_id = token_ids['none_token_id']
    device = logits.device

    box_type = is_valid_box_frame(probs, token_ids, start_thresh=start_thresh, end_thresh=end_thresh, topk=keep_k)
    if box_type == 'empty_box':
        return torch.tensor([
            box_start_token_id, none_token_id, box_end_token_id,
            token_ids['null_token_id'], token_ids['null_token_id'], token_ids['null_token_id']
        ], dtype=torch.long, device=device)
    elif box_type == 'illegal_box':
        return None

    pos_probs, pos_ids = torch.topk(probs[1:5], k=keep_k, dim=-1)
    mask = (pos_ids >= coord_start_token_id) & (pos_ids <= coord_end_token_id)
    has_valid = mask.any(dim=-1)
    if not has_valid.all():
        return None

    first_valid_idx = mask.long().argmax(dim=-1, keepdim=True)
    first_valid_probs = pos_probs.gather(-1, first_valid_idx).squeeze(-1)
    first_valid_ids = pos_ids.gather(-1, first_valid_idx).squeeze(-1)

    if generation_mode == 'hybrid':
        valid_counts = mask.sum(dim=-1)
        LARGE_NUM, SMALL_NUM = 999999, -999999
        valid_ids_for_max = torch.where(mask, pos_ids, torch.tensor(SMALL_NUM, device=device))
        valid_ids_for_min = torch.where(mask, pos_ids, torch.tensor(LARGE_NUM, device=device))
        valid_max = valid_ids_for_max.max(dim=-1)[0]
        valid_min = valid_ids_for_min.min(dim=-1)[0]
        is_abnormal = (first_valid_probs < 0.9) & (valid_counts > 1) & ((valid_max - valid_min) > 60)
        final_coords = torch.where(is_abnormal, torch.tensor(0, device=device), first_valid_ids)
    else:
        final_coords = first_valid_ids

    start_t = torch.tensor([box_start_token_id], dtype=final_coords.dtype, device=device)
    end_t = torch.tensor([box_end_token_id], dtype=final_coords.dtype, device=device)
    return torch.cat([start_t, final_coords, end_t])


def decode_ref(logits, probs, token_ids, keep_k=5, start_thresh=0.6):
    ref_start_token_id = token_ids.get('ref_start_token_id')
    coord_start_token_id = token_ids['coord_start_token_id']
    coord_end_token_id = token_ids['coord_end_token_id']
    device = probs.device
    L = probs.size(0)

    if probs[0, ref_start_token_id] < start_thresh:
        return None

    pos_probs, pos_ids = torch.topk(probs[1:], k=keep_k, dim=-1)
    is_coord = (pos_ids >= coord_start_token_id) & (pos_ids <= coord_end_token_id)
    is_valid = ~is_coord
    has_valid = is_valid.any(dim=-1)
    if not has_valid.all():
        return None

    first_valid_idx = is_valid.long().argmax(dim=-1, keepdim=True)
    final_text_ids = pos_ids.gather(-1, first_valid_idx).squeeze(-1)
    start_t = torch.tensor([ref_start_token_id], dtype=final_text_ids.dtype, device=device)
    return torch.cat([start_t, final_text_ids])


def handle_pattern(x0, token_ids, generation_mode='hybrid'):
    null_token_id = token_ids['null_token_id']
    im_end_token_id = token_ids['im_end_token_id']
    box_start_token_id = token_ids['box_start_token_id']
    box_end_token_id = token_ids['box_end_token_id']
    none_token_id = token_ids['none_token_id']
    coord_start_token_id = token_ids['coord_start_token_id']
    coord_end_token_id = token_ids['coord_end_token_id']
    ref_end_token_id = token_ids['ref_end_token_id']

    x0 = x0.tolist()

    if x0[0] == null_token_id or x0[0] == im_end_token_id:
        return {"type": "im_end", "tokens": [im_end_token_id]}
    elif x0[:2] == [box_start_token_id, none_token_id]:
        return {"type": "empty_box", "tokens": [box_start_token_id, none_token_id, box_end_token_id]}
    elif x0[0] == box_start_token_id:
        coord_ix = 1
        for coord in x0[1:5]:
            if coord_start_token_id <= coord <= coord_end_token_id:
                coord_ix += 1
            else:
                break
        if coord_ix == 5 and x0[5] == box_end_token_id:
            return {"type": "coord_box", "tokens": x0}
        elif coord_ix == 3 and x0[3] == box_end_token_id:
            return {"type": "point_box", "tokens": x0[:4]}
        else:
            if generation_mode == 'fast':
                return {"type": "coord_box", "tokens": x0}
            else:
                return {"type": "error_box", "tokens": x0[:coord_ix]}
    else:
        for i, token in enumerate(x0):
            if token == null_token_id:
                x0 = x0[:i]
                break
        if len(x0) >= 2 and x0[-1] == x0[-2] == ref_end_token_id:
            x0 = x0[:-1]
        return {"type": "ref_object", "tokens": x0}
