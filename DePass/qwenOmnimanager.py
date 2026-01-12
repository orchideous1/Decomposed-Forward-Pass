import torch
from typing import Optional, Union, List, Callable, Tuple
from torch import nn
from typing import Dict, List, Tuple, Union
from contextlib import contextmanager
import math
import torch
from tqdm import tqdm
from functools import partial
import torch.nn.functional as F
from DePass.utils import (
    capture_sdpa,
    get_rmsnorm_scaling,
    compute_attention_weights_mha,
    repeat_kv,
    hook_manager
)
from DePass.manager import decomposed_state_manager
import sys
sys.path.append("/home/linyiwu/OmniZip")
from qwen_omni_utils import process_mm_info
import pdb



def process_omni_inputs(thinker_model, inputs):
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    input_features = inputs.get("input_features")
    pixel_values = inputs.get("pixel_values")
    pixel_values_videos = inputs.get("pixel_values_videos")
    image_grid_thw = inputs.get("image_grid_thw")
    video_grid_thw = inputs.get("video_grid_thw")
    feature_attention_mask = inputs.get("feature_attention_mask")
    audio_feature_lengths = inputs.get("audio_feature_lengths")
    use_audio_in_video = True
    video_second_per_grid = inputs.get("video_second_per_grid")

    inputs_embeds = thinker_model.get_input_embeddings()(input_ids)

    if input_ids is not None and input_ids.shape[1] != 1:
        if input_features is not None:
            audio_features = thinker_model.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_mask = (
                (input_ids == thinker_model.config.audio_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if pixel_values is not None:
            image_embeds = thinker_model.get_image_features(pixel_values, image_grid_thw)
            image_mask = (
                (input_ids == thinker_model.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = thinker_model.get_video_features(pixel_values_videos, video_grid_thw)
            video_mask = (
                (input_ids == thinker_model.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
    else:
        audio_feature_lengths = None

    position_ids = None
    if attention_mask is not None:
        # delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
        position_ids, _ = thinker_model.get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask,
            use_audio_in_video,
            audio_feature_lengths,
            video_second_per_grid,
        )
        if position_ids is not None:
            position_ids = position_ids.to(inputs_embeds.device)


    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids
    }

def get_stream_from_inputs(model, inputs) -> Tuple[List, List, List, Dict, Dict]:
    # input_ts = torch.tensor(input_ids, dtype=torch.int64, device=model.device).unsqueeze(0)
    with torch.no_grad():
        processed_inputs = process_omni_inputs(model.thinker, inputs)

    pre_norm_hidden = {}
    def capture_pre_norm_hook(module, input, output):
        pre_norm_hidden['last'] = input[0].detach().cpu()
    handle = model.thinker.model.norm.register_forward_hook(capture_pre_norm_hook)
    with hook_manager(model.thinker) as (block_inputs, attn_intermediates, final_outputs, attn_position_embeddings):
        with torch.no_grad():
            outputs = model.thinker.model(**processed_inputs, output_hidden_states=True)
    handle.remove()
    hidden_states = outputs.hidden_states
    attn_intermediate_list = [attn_intermediates[i] for i in sorted(attn_intermediates.keys())]
    stream = []
    for i in range(len(attn_intermediate_list)):
        stream.append(hidden_states[i].detach().cpu())
        stream.append(attn_intermediate_list[i])
    stream.append(pre_norm_hidden['last'])

    return stream, attn_intermediate_list, hidden_states, processed_inputs, attn_position_embeddings


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_qwenomni(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class qwenOmnimanager(decomposed_state_manager):
    def __init__(self, model, processor, mlp_softmax_temp=0.1,ignore_special_token=True, mlp_decomposed_function="softmax"):
        self.model = model
        self.text_model = model.thinker.model
        self.processor = processor
        self.num_heads = model.thinker.config.text_config.num_attention_heads
        self.num_key_value_heads = model.thinker.config.text_config.num_key_value_heads
        self.num_layers = model.thinker.config.text_config.num_hidden_layers
        self.hidden_dim = model.thinker.config.text_config.hidden_size
        self.head_dim = self.hidden_dim // self.num_heads
        self.mlp_softmax_temp = mlp_softmax_temp
        self.ignore_special_token = ignore_special_token
        self.attention_weight_compute = compute_attention_weights_mha
        self.model_name = "qwen"

        if mlp_decomposed_function == "softmax":
            self.mlp_decomposed_compute = self.mlp_decomposed_compute_softmax
        elif mlp_decomposed_function == "relu":
            self.mlp_decomposed_compute = self.mlp_decomposed_compute_relu
        elif mlp_decomposed_function == "max":
            self.mlp_decomposed_compute = self.mlp_decomposed_compute_max
        elif mlp_decomposed_function == "linear":
            self.mlp_decomposed_compute = self.mlp_decomposed_compute_linear
        elif mlp_decomposed_function == "taylor":
            self.mlp_decomposed_compute = self.mlp_decomposed_compute_taylor
            
        

        self.apply_rotary_pos_emb = apply_rotary_pos_emb_qwenomni

    def _prepare_inputs_for_omni(self, conversation):
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        print(text)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        return inputs
    
    def test_forward(self, conversation):
        inputs = self.prepare_inputs_for_omni(conversation)
        with torch.no_grad():
            gen_out = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        use_audio_in_video=True,
                        return_audio=False,
                    )
        gen_ids = gen_out
        if isinstance(gen_out, tuple):
            # 当 return_audio=True 时，OmniZip 模型返回 (sequences, wav)
            gen_ids, wav_tensor = gen_out
        elif hasattr(gen_out, "sequences"):
            gen_ids = gen_out.sequences

        generated_text = self.processor.batch_decode(
            gen_ids[:, (inputs["input_ids"].shape[1]-5) :],

        )# [0].split("<|im_end|>")[0]
        print(generated_text)

    def from_states_to_probs(self, states, topk=5):
        """convert states(d dimension)to token ids and probabilities(vocabulary size)
        Args:
            states: torch.Tensor, shape=(d, )
            tokenizer: transformers.Tokenizer
            layer_idx: int, layer index
            topk: int, topk
        Returns:
            token_probs: dict, token probabilities
        """
        logits = self.model.thinker.lm_head(states)
        traj_log_probs = logits.log_softmax(dim=-1).squeeze()
        topk_values, topk_indices = torch.topk(traj_log_probs, k=topk)
        probs = torch.exp(topk_values)
        token_probs = []
        for idx, prob in zip(topk_indices.cpu(), probs.cpu()):
            token = self.processor.batch_decode(
                        idx,
                        skip_special_tokens=True,
                    )
            token_probs.append((idx.item(), token, prob.item()))
        return token_probs

    def _get_states(self, inputs):
        """Get the states of the model for a given prompt.(Including the post attention states)
        Args:
            prompt (_str_): The input prompt for the model.
        """
        original_sdpa = F.scaled_dot_product_attention
        my_sdpa, get_captured = capture_sdpa()
        F.scaled_dot_product_attention = my_sdpa
        states, after_attn_list, hidden_states, processed_inputs, attn_position_embeddings = get_stream_from_inputs(self.model, inputs)
        attribute_len = states[0][0].shape[0]
        F.scaled_dot_product_attention = original_sdpa
        masks, dropouts, causals = get_captured()
        self.states = states
        self.masks = masks
        self.dropouts = dropouts
        self.causals = causals
        self.attribute_len = attribute_len
        self.processed_inputs = processed_inputs
        self.attn_position_embeddings = attn_position_embeddings
        return states


    def get_last_layer_decomposed_state(self, conversation, single_compute_token=1):
        """
        Get states from prompt and compute the attribute state from the first layer.
        """
        inputs = self._prepare_inputs_for_omni(conversation)
        self._get_states(inputs)
        states = self.states
        # 修改调用为 compute_modality_decomposed_state，并传入 input_ids
        attribute_state, mod_names = self.compute_modality_decomposed_state(
            inputs["input_ids"][0], 
            start_layer_idx=0, 
            single_compute_token=single_compute_token
        )
        return states, attribute_state, mod_names
    
    def _get_modality_masks(self, input_ids):
        audio_ids_mask = input_ids == self.model.config.thinker_config.audio_token_index
        image_ids_mask = input_ids == self.model.config.thinker_config.image_token_index
        video_ids_mask = input_ids == self.model.config.thinker_config.video_token_index

        return [audio_ids_mask, image_ids_mask, video_ids_mask]
    

    def compute_modality_decomposed_state(self, input_ids, start_layer_idx=0, single_compute_token=10):
        modality_masks = self._get_modality_masks(input_ids)
        masks = self.masks
        dropouts = self.dropouts
        causals = self.causals
        states = self.states
        attribute_len = self.attribute_len

        # 1. 解析输入的 mask, 构造动态模态列表
        # 这里假设 modality_masks 里的顺序是 [Audio, Image, Video]
        # Text 模态默认是剩余的部分
        standard_names = ["Audio", "Image", "Video"]
        active_masks = {}
        
        # 记录所有非文本模态的总 Mask
        total_special_mask = torch.zeros(attribute_len, dtype=torch.bool, device=self.model.device)
        
        for i, mask in enumerate(modality_masks):
            if i < len(standard_names) and mask.any():
                active_masks[standard_names[i]] = mask
                total_special_mask |= mask

        # 自动添加 Text 模态 (即非 Audio/Image/Video 的部分)
        text_mask = ~total_special_mask
        if text_mask.any():
            active_masks["Text"] = text_mask
            
        mod_names = list(active_masks.keys())
        final_modality_states = []
        # 将起始层状态移至模型设备，确保后续减法运算在同一设备上
        start_state = states[2*start_layer_idx][0].to(self.model.device)

        # 外层循环：模态 (Modality) - 若模态不存在已在上一步过滤
        for name in mod_names:
            mask = active_masks[name]
            indices = torch.where(mask)[0]
            
            # 模态累加器 (置于 CPU 节省显存)
            mod_accumulator = torch.zeros(attribute_len, self.hidden_dim, device="cpu", dtype=states[0].dtype)
            
            # 内层循环：原始分块 Token 计算 (解决 N 维度 OOM)
            forward_num = (len(indices) + single_compute_token - 1) // single_compute_token
            
            pbar = tqdm(range(forward_num), desc=f"Computing {name}")
            for cur_num in pbar:
                start_i = cur_num * single_compute_token
                end_i = min((cur_num + 1) * single_compute_token, len(indices))
                cur_indices = indices[start_i:end_i]
                chunk_len = len(cur_indices)
                
                # 初始化分块 attribute_state: [chunk_len, N, D]
                attribute_state = torch.zeros(
                    chunk_len, attribute_len, self.hidden_dim, 
                    dtype=states[0].dtype, device=self.model.device
                )
                for i, idx in enumerate(cur_indices):
                    attribute_state[i, idx, :] = start_state[idx, :]
                
                # 兼容特殊 token 处理逻辑
                if self.ignore_special_token and cur_num != 0:
                    special_token_state = torch.zeros(1, attribute_len, self.hidden_dim, 
                                                   dtype=states[0].dtype, device=self.model.device)
                    special_token_state[0, 0, :] = start_state[0, :]
                    attribute_state = torch.cat([special_token_state, attribute_state], dim=0)

                # 添加残差 left_state 保证线性守恒
                attribute_state_sum = torch.sum(attribute_state, dim=0)
                left_state = start_state - attribute_state_sum
                attribute_state = torch.cat([attribute_state, left_state.unsqueeze(0)], dim=0)

                # 逐层前向传播
                with torch.no_grad():
                    for layer_idx in range(start_layer_idx, self.num_layers):
                        # print(f"layer {layer_idx}")
                        # Attention 分解
                        s_attn = states[layer_idx*2]
                        attribute_state = self._attn_decomposed_compute(layer_idx, s_attn, masks, dropouts, causals, attribute_state)
                        # MLP 分解
                        s_mlp = states[layer_idx*2 + 1]
                        attribute_state = self.mlp_decomposed_compute(
                            layer_idx, s_mlp, attribute_state, self.mlp_softmax_temp, 
                            ignore_special_token=self.ignore_special_token, cur_num=cur_num
                        )

                # 剥离辅助分量
                attribute_state = attribute_state[:-1, :, :]
                if self.ignore_special_token and cur_num != 0:
                    attribute_state = attribute_state[1:, :, :]
                
                # 聚合本块贡献到模态
                mod_accumulator += attribute_state.sum(dim=0).cpu()

            final_modality_states.append(mod_accumulator)

        # 整理最终 [M, N, D] 矩阵并应用 Final Norm
        final_state = torch.stack(final_modality_states).to(self.model.device)
        with torch.no_grad():
            state_final = states[-1][0].to(final_state.device)
            ln = self.model.thinker.model.norm
            W_diag_elements = get_rmsnorm_scaling(ln, state_final)
            final_state = final_state * W_diag_elements

        return final_state, mod_names
    
    def _attn_decomposed_compute(self,layer_idx,state,masks,dropouts,causals,attribute_state):
        """
        Decomposed Attention module computation.
        Args:
            layer_idx (int): The layer index for the Attention module.
            state (torch.Tensor): The input state tensor with shape: N, D
                - N: sequence length (number of tokens)
                - D: hidden dimension size (model's hidden state dimension)
            masks (list): A list of attention masks for each layer.
            dropouts (list): A list of dropout probabilities for each layer.
            causals (list): A list of causal flags for each layer.
            attribute_state (torch.Tensor): The attribute state tensor with shape: N, M, D
                - N: sequence length (number of tokens)
                - M: DePass component number
                - D: hidden dimension size (model's hidden state dimension)
        Returns:
            attribute_state (torch.Tensor): The updated attribute state tensor with shape: N, M, D
                - N: sequence length (number of tokens)
                - M: DePass component number
                - D: hidden dimension size (model's hidden state dimension)
        """
        layer = self.text_model.layers[layer_idx]
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj
        ln = layer.input_layernorm
        state_norm = ln(state)
        
        # 使用捕获到的 position_embeddings 和配置中的 mrope_section
        pos_embeddings = self.attn_position_embeddings.get(layer_idx)
        mrope_section = self.model.thinker.config.text_config.rope_scaling['mrope_section']
        
        attn_weight = self.attention_weight_compute(
            state_norm, layer.self_attn, layer_idx, 
            masks[layer_idx], dropouts[layer_idx], causals[layer_idx],
            self.apply_rotary_pos_emb, self.model_name,
            position_embeddings=pos_embeddings, mrope_section=mrope_section
        )
        W_diag_elements = get_rmsnorm_scaling(ln, state).to(attribute_state.device)
        attribute_state_norm = attribute_state * W_diag_elements
        target_device = attribute_state.device
        target_dtype = v_proj.weight.dtype
        v_proj_weight = v_proj.weight.view(self.num_key_value_heads, self.head_dim, self.hidden_dim).to(device=target_device, dtype=target_dtype)
        o_proj_weight = o_proj.weight.view(self.hidden_dim, self.num_heads, self.head_dim).to(device=target_device, dtype=target_dtype)
        attn_weight = attn_weight[0].to(device=target_device, dtype=target_dtype)
        attribute_state_norm = attribute_state_norm.to(dtype=target_dtype)
        attribute_values = torch.einsum("kqi,nhi->knqh", attribute_state_norm, v_proj_weight)
        attribute_values = repeat_kv(attribute_values, self.num_heads // self.num_key_value_heads)
        vo_attribute = torch.einsum("knqh,jnh->kqnj", attribute_values, o_proj_weight)
        attribute_state += torch.einsum("iknd, nqk -> iqd", vo_attribute, attn_weight)
        return attribute_state
    
    def mlp_decomposed_compute_softmax(self, layer_idx, state, attribute_state, mlp_softmax_temp=0.1, ignore_special_token=True, cur_num=None):
        """
        Decomposed MLP module computation.
        Neurons Distributed based on softmax contributing coefficients.
        Args:
            layer_idx (int): The layer index for the MLP module.
            state (torch.Tensor): The input state tensor with shape: N, D
                - N: sequence length (number of tokens)
                - D: hidden dimension size (model's hidden state dimension)
            attribute_state (torch.Tensor): The attribute state tensor with shape: N, M, D
                - N: sequence length (number of tokens)
                - M: DePass component number
                - D: hidden dimension size (model's hidden state dimension)
            mlp_softmax_temp (float, optional): The temperature for softmax. Defaults to 0.1.
            ignore_special_token (bool, optional): Whether to ignore special tokens. Defaults to True.
            cur_num (int, optional): The current token index. Defaults to None.
        Returns:
            attribute_state (torch.Tensor): The updated attribute state tensor with shape: N, M, D
                - N: sequence length (number of tokens)
                - M: DePass component number
                - D: hidden dimension size (model's hidden state dimension)
        """
        
        layer = self.text_model.layers[layer_idx]
        target_device = attribute_state.device
        ln = layer.post_attention_layernorm
        gate_proj = layer.mlp.gate_proj
        target_dtype = gate_proj.weight.dtype
        state = state.to(device=target_device, dtype=target_dtype)
        state_norm = ln(state)
        W_diag_elements = get_rmsnorm_scaling(ln, state).to(device=target_device)
        W_diag_elements = W_diag_elements.to(dtype=target_dtype)        
        attribute_state_norm = attribute_state.to(dtype=target_dtype) * W_diag_elements
        gate_ratio = gate_proj(attribute_state_norm).transpose(-2, -1)
        ori_gate = gate_proj(state_norm).transpose(-2, -1)
        # gate_ratio = torch.cat([gate_ratio, ori_gate - gate_ratio.sum(0)])
        # Process the speacial token
        if ignore_special_token:
            last_minus_first = gate_ratio[-1:]
            gate_ratio = torch.cat([
                torch.zeros_like(gate_ratio[:1]),
                gate_ratio[1:-1],          
                last_minus_first  
            ], dim=0)
        elif ignore_special_token and cur_num == None:
            raise ValueError("ignore_special_token is True, but cur_num is None. Provide cur_num.")
        gate_ratio[0,:,0] = 1.0
        gate_ratio[-1,:,:] = torch.where(torch.abs(gate_ratio[-1,:,:]) < 1e-5, torch.zeros_like(gate_ratio[-1,:,:]), gate_ratio[-1,:,:])
        zero_cols = (gate_ratio[:,:,:] == 0).all(dim=0)
        gate_ratio[-1,:,:] = torch.where(zero_cols, torch.ones_like(gate_ratio[-1,:,:]),gate_ratio[-1,:,:])
        mask = (gate_ratio != 0).float()
        gate_ratio.masked_fill_(mask == 0, float('-inf'))
        gate_ratio = torch.softmax(gate_ratio / mlp_softmax_temp, dim=0) * mask
        attribute_mlp_values = torch.zeros_like(attribute_state)
        mlp_token_compute_num = state_norm.shape[1]
        num_batches = (state_norm.shape[1] + mlp_token_compute_num - 1) // mlp_token_compute_num

        for i in range(num_batches):
            start_idx = i * mlp_token_compute_num
            end_idx = min((i + 1) * mlp_token_compute_num, state_norm.shape[1])
            mlp = layer.mlp
            mlp_gate = mlp.act_fn(mlp.gate_proj(state_norm[:,start_idx:end_idx,:])) * mlp.up_proj(state_norm[:,start_idx:end_idx,:])   # self.get_per_ffn2_values(state_norm[:,start_idx:end_idx,:], layer_idx).to(target_device)
            gate_slice = gate_ratio[:, :, start_idx:end_idx].to(target_device)
            mlp_gate = mlp_gate.squeeze(0)
            gate_slice_trans = gate_slice.permute(0, 2, 1).to(mlp_gate.device)
            weight = layer.mlp.down_proj.weight
            mlp_gate_expanded = mlp_gate.unsqueeze(0).expand(gate_slice_trans.size(0), -1, -1) 
            mlp_gate = mlp_gate_expanded * gate_slice_trans
            k, n, i = mlp_gate.shape
            d = weight.size(0)
            mlp_gate = mlp_gate.view(k * n, i)
            target_dtype = weight.dtype
            mlp_gate = mlp_gate.to(dtype=target_dtype)
            output = mlp_gate @ weight.T 
            output = output.view(k, n, d)
            attribute_mlp_values[:, start_idx:end_idx, :] += output.to(attribute_mlp_values.device)
        attribute_state += attribute_mlp_values
        return attribute_state