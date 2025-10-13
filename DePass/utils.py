import torch
from contextlib import contextmanager
from typing import Dict, List, Tuple, Union
from torch import nn
import torch.nn.functional as F
import math
  
def capture_sdpa():
    # Capture the scaled dot product attention function
    original_sdpa = F.scaled_dot_product_attention
    masks, dropouts, causals = [], [], [] 
    def my_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        masks.append(attn_mask)
        dropouts.append(dropout_p)
        causals.append(is_causal)

        return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

    return my_sdpa, lambda: (masks, dropouts, causals)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:    
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def get_rmsnorm_scaling(norm_layer, hidden_states):
    """
    Compute the scaling factors for RMSNorm.
    """
    target_dtype = hidden_states.dtype
    weight = norm_layer.weight.to(device=hidden_states.device, dtype=target_dtype)
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    scale_factor = torch.rsqrt(variance + norm_layer.variance_epsilon)
    return weight * scale_factor.to(dtype=target_dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_llama(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_qwen(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def compute_attention_weights_mha(state_norm, attn_layer, layer_idx, attn_mask, dropout_p, is_causal, apply_rotary_pos_emb,model_name):
    q_proj, k_proj = attn_layer.q_proj, attn_layer.k_proj
    query_states = q_proj(state_norm)
    key_states = k_proj(state_norm)    
    bsz, seq_len, hidden_dim = query_states.shape
    num_heads = attn_layer.num_heads
    head_dim = hidden_dim // num_heads
    position_ids = torch.arange(seq_len, dtype=torch.long, device=state_norm.device).unsqueeze(0)
    query_states = query_states.view(bsz, seq_len, attn_layer.num_heads, attn_layer.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, seq_len, attn_layer.num_key_value_heads, attn_layer.head_dim).transpose(1, 2)
    if model_name == "llama":
        cos, sin = attn_layer.rotary_emb(state_norm, position_ids)
    elif model_name == "qwen":
        cos, sin = attn_layer.rotary_emb(state_norm, seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states = repeat_kv(key_states, attn_layer.num_heads // attn_layer.num_key_value_heads)
    _, _, q_len, _ = query_states.shape
    L, S = query_states.size(-2), key_states.size(-2)
    scale_factor = 1 / math.sqrt(query_states.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query_states.dtype, device=query_states.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query_states.device).tril(diagonal=0)
        attn_bias.masked_fill_(~temp_mask, float("-inf"))
        attn_bias = attn_bias.to(query_states.dtype)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(~attn_mask, float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias
    attn_weight = query_states @ key_states.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight

        
@contextmanager
def hook_manager(model: nn.Module):

    block_inputs: Dict[int, torch.Tensor] = {}
    attn_intermediates: Dict[int, torch.Tensor] = {}
    final_outputs: Dict[int, torch.Tensor] = {}
    hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    def get_block_pre_hook(layer_idx: int):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor]):
            block_inputs[layer_idx] = inputs[0]
        return hook

    def get_self_attn_hook(layer_idx: int):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor], output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
            attn_out = output[0] if isinstance(output, tuple) else output
            device = attn_out.device
            default_zeros = torch.zeros_like(attn_out, device=device)
            block_input = block_inputs.get(layer_idx, default_zeros)
            if block_input.device != device:
                block_input = block_input.to(device)
            attn_intermediates[layer_idx] = block_input + attn_out.clone()
        return hook

    def get_block_post_hook(layer_idx: int):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor):
            final_outputs[layer_idx] = output
        return hook

    for idx, block in enumerate(model.model.layers):
        hook_handles.append(block.register_forward_pre_hook(get_block_pre_hook(idx)))
        hook_handles.append(block.self_attn.register_forward_hook(get_self_attn_hook(idx)))
        hook_handles.append(block.register_forward_hook(get_block_post_hook(idx)))

    try:
        yield block_inputs, attn_intermediates, final_outputs
    finally:
        for h in hook_handles:
            h.remove()

def get_stream_from_prompt(tokenizer, model, prompt: str) -> List:
    input_ids = tokenizer.encode(prompt)
    input_ts = torch.tensor(input_ids, dtype=torch.int64, device=model.device).unsqueeze(0)
    pre_norm_hidden = {}
    def capture_pre_norm_hook(module, input, output):
        pre_norm_hidden['last'] = input[0].detach()
    handle = model.model.norm.register_forward_hook(capture_pre_norm_hook)
    with hook_manager(model) as (block_inputs, attn_intermediates, final_outputs):
        outputs = model(input_ts, output_hidden_states=True)
    handle.remove()
    hidden_states = outputs.hidden_states
    attn_intermediate_list = [attn_intermediates[i] for i in sorted(attn_intermediates.keys())]
    stream = []
    for i in range(len(attn_intermediate_list)):
        stream.append(hidden_states[i])
        stream.append(attn_intermediate_list[i])
    stream.append(pre_norm_hidden['last'])

    return stream, attn_intermediate_list, hidden_states


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

def from_states_to_probs(states, lm_head, tokenizer, topk=5):
    """convert states(d dimension)to token ids and probabilities(vocabulary size)
    Args:
        states: torch.Tensor, shape=(d, )
        tokenizer: transformers.Tokenizer
        layer_idx: int, layer index
        topk: int, topk
    Returns:
        token_probs: dict, token probabilities
    """
    logits = lm_head(states)
    traj_log_probs = logits.log_softmax(dim=-1).squeeze()
    topk_values, topk_indices = torch.topk(traj_log_probs, k=topk)
    probs = torch.exp(topk_values)
    token_probs = []
    for idx, prob in zip(topk_indices.cpu(), probs.cpu()):
        token = tokenizer.decode(idx)
        token_probs.append((idx.item(), token, prob.item()))
    return token_probs