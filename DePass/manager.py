
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
    apply_rotary_pos_emb_llama,
    apply_rotary_pos_emb_qwen,
    compute_attention_weights_mha,
    repeat_kv,
    get_stream_from_prompt,
)

class decomposed_state_manager():
    def __init__(self, model,tokenizer,mlp_softmax_temp=0.1,ignore_special_token=True, mlp_decomposed_function="softmax"):
        
        self.model = model
        self.tokenizer = tokenizer
        self.num_heads = self.model.config.num_attention_heads
        self.num_key_value_heads = self.model.config.num_key_value_heads
        self.num_layers = self.model.config.num_hidden_layers
        self.head_dim = self.model.config.hidden_size // self.num_heads
        self.hidden_dim = self.model.config.hidden_size
        self.mlp_softmax_temp = mlp_softmax_temp
        self.ignore_special_token = ignore_special_token
        self.attention_weight_compute = compute_attention_weights_mha
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
            
        
        if hasattr(model.config, "model_type"):
            model_type = model.config.model_type.lower()
            if "llama" in model_type:
                self.apply_rotary_pos_emb = apply_rotary_pos_emb_llama
                self.model_name = "llama"
            elif "qwen" in model_type:
                self.apply_rotary_pos_emb = apply_rotary_pos_emb_qwen
                self.model_name = "qwen"
            else:
                self.apply_rotary_pos_emb = apply_rotary_pos_emb_llama
                self.model_name = "llama"
                print(f"Warning: Unknown model type {model_type}, using default LLaMA rotary position embedding")
                     
    def get_states(self, prompt):
        """Get the states of the model for a given prompt.(Including the post attention states)
        Args:
            prompt (_str_): The input prompt for the model.
        """
        original_sdpa = F.scaled_dot_product_attention
        my_sdpa, get_captured = capture_sdpa()
        F.scaled_dot_product_attention = my_sdpa
        states, after_attn_list, hidden_states = get_stream_from_prompt(self.tokenizer, self.model, prompt)
        attribute_len = states[0][0].shape[0]
        F.scaled_dot_product_attention = original_sdpa
        masks, dropouts, causals = get_captured()
        self.states = states
        self.masks = masks
        self.dropouts = dropouts
        self.causals = causals
        self.attribute_len = attribute_len
        return states
    
    def compute_decomposed_state(self,start_layer_idx,single_compute_token=None):
        """
        Token-Level Initialization at the given layer and compute the last layer attribute state.
        
        Args:
            start_layer_idx (int): The layer index for the Token-Level Initialization.
        Returns:
            attribute_state (torch.Tensor): The computed attribute state tensor with shape:
                - N: sequence length (number of tokens)
                - N: sequence length (number of tokens) 
                - D: hidden dimension size (model's hidden state dimension)
                Shape: (N, N, D)
        """
        masks = self.masks
        dropouts = self.dropouts
        causals = self.causals
        states = self.states
        attribute_len = self.attribute_len
        if single_compute_token is None:
            single_compute_token = attribute_len + 1
        forward_num = math.ceil(self.attribute_len/single_compute_token)
        attribute_state_list = []
        with torch.no_grad():
            for cur_num in range(forward_num):
                # Initialization of attribute state at the given layer
                start_token_idx = cur_num * single_compute_token
                end_token_idx = min((cur_num + 1) * single_compute_token, attribute_len)
                cur_compute_token_len = end_token_idx - start_token_idx
                attribute_state = torch.zeros(cur_compute_token_len, attribute_len, self.model.config.hidden_size, dtype=next(self.model.parameters()).dtype,device=self.model.device)
                idx_range = torch.arange(start_token_idx, end_token_idx)
                local_idx = idx_range - start_token_idx
                attribute_state[local_idx, idx_range, :] = states[2*start_layer_idx][0][idx_range, :]
                if self.ignore_special_token and cur_num !=0 :
                    special_token_state = torch.zeros(1, attribute_state.shape[1], attribute_state.shape[2], dtype=next(self.model.parameters()).dtype,).to(self.model.device)
                    special_token_state[0,0,:] = states[2*start_layer_idx][0][0, :]
                    attribute_state = torch.cat([special_token_state, attribute_state], dim=0)
                attribute_state_sum = torch.sum(attribute_state, dim=0)
                left_state = states[2*start_layer_idx][0] - attribute_state_sum
                attribute_state = torch.cat([attribute_state,left_state.unsqueeze(0)], dim=0)
                # Compute the last layer attribute state
                for layer_idx in range(start_layer_idx, self.num_layers):
                    # Attn module computation
                    state = states[layer_idx*2]
                    attribute_state = self.attn_decomposed_compute(layer_idx, state, masks, dropouts, causals, attribute_state)
                    # MLP module computation
                    state = states[layer_idx*2 + 1]
                    attribute_state = self.mlp_decomposed_compute(layer_idx, state, attribute_state,self.mlp_softmax_temp, ignore_special_token=self.ignore_special_token, cur_num=cur_num)
                attribute_state = attribute_state[:-1,:,:]
                if self.ignore_special_token and cur_num !=0 :
                    attribute_state = attribute_state[1:,:,:]
                # Last layernorm computation
                state=states[-1][0].to(attribute_state.device)
                ln = self.model.model.norm
                W_diag_elements = get_rmsnorm_scaling(ln, state)
                attribute_state = attribute_state * W_diag_elements
                attribute_state_list.append(attribute_state.clone().detach().cpu())
            attribute_state = torch.cat(attribute_state_list, dim=0)
            attribute_state=attribute_state.transpose(0, 1)
        return attribute_state
    
    def get_middle_to_last_decomposed_state(self, prompt, start_layer_idx):
        """
        Get states from prompt and compute the attribute state from the start layer index.
        """
        self.get_states(prompt)
        states = self.states
        attribute_state = self.compute_decomposed_state(start_layer_idx)
        return attribute_state, states

    def get_last_layer_decomposed_state(self, prompt,single_compute_token=None):
        """
        Get states from prompt and compute the attribute state from the first layer.
        """
        self.get_states(prompt)
        states = self.states
        attribute_state = self.compute_decomposed_state(0, single_compute_token=single_compute_token)
        return attribute_state, states
    
    def get_middel_to_middle_decomposed_state(self, prompt, start_layer_idx, end_layer_idx, single_compute_token=None):
        self.get_states(prompt)
        masks = self.masks
        dropouts = self.dropouts
        causals = self.causals
        states = self.states
        attribute_len = self.attribute_len
        if single_compute_token is None:
            single_compute_token = attribute_len + 1
        forward_num = math.ceil(self.attribute_len/single_compute_token)
        attribute_state_list = []
        with torch.no_grad():
            for cur_num in range(forward_num):
                # Initialization of attribute state at the given layer
                start_token_idx = cur_num * single_compute_token
                end_token_idx = min((cur_num + 1) * single_compute_token, attribute_len)
                cur_compute_token_len = end_token_idx - start_token_idx
                attribute_state = torch.zeros(cur_compute_token_len, attribute_len, self.model.config.hidden_size, dtype=next(self.model.parameters()).dtype,device=self.model.device)
                idx_range = torch.arange(start_token_idx, end_token_idx)
                local_idx = idx_range - start_token_idx
                attribute_state[local_idx, idx_range, :] = states[2*start_layer_idx][0][idx_range, :]
                if self.ignore_special_token and cur_num !=0 :
                    special_token_state = torch.zeros(1, attribute_state.shape[1], attribute_state.shape[2], dtype=next(self.model.parameters()).dtype,).to(self.model.device)
                    special_token_state[0,0,:] = states[2*start_layer_idx][0][0, :]
                    attribute_state = torch.cat([special_token_state, attribute_state], dim=0)
                attribute_state_sum = torch.sum(attribute_state, dim=0)
                left_state = states[2*start_layer_idx][0] - attribute_state_sum
                attribute_state = torch.cat([attribute_state,left_state.unsqueeze(0)], dim=0)
                # Compute the last layer attribute state
                for layer_idx in range(start_layer_idx, end_layer_idx):
                    # Attn module computation
                    state = states[layer_idx*2]
                    attribute_state = self.attn_decomposed_compute(layer_idx, state, masks, dropouts, causals, attribute_state)
                    # MLP module computation
                    state = states[layer_idx*2 + 1]
                    attribute_state = self.mlp_decomposed_compute(layer_idx, state, attribute_state,self.mlp_softmax_temp, ignore_special_token=self.ignore_special_token, cur_num=cur_num)
                attribute_state = attribute_state[:-1,:,:]
                if self.ignore_special_token and cur_num !=0 :
                    attribute_state = attribute_state[1:,:,:]
                attribute_state_list.append(attribute_state.clone().detach().cpu())
            attribute_state = torch.cat(attribute_state_list, dim=0)
            attribute_state = attribute_state.transpose(0, 1)
        return attribute_state, states
    
    def all_layer_to_init_attribute_state(self, prompt,single_compute_token=None):
        self.get_states(prompt)
        masks = self.masks
        dropouts = self.dropouts
        causals = self.causals
        states = self.states
        attribute_len = self.attribute_len
        if single_compute_token is None:
            single_compute_token = attribute_len + 1
        forward_num = math.ceil(self.attribute_len/single_compute_token)
        attribute_state_list = []
        with torch.no_grad():
            for cur_num in range(forward_num):
                start_token_idx = cur_num * single_compute_token
                end_token_idx = min((cur_num + 1) * single_compute_token, attribute_len)
                cur_compute_token_len = end_token_idx - start_token_idx
                attribute_state = torch.zeros(cur_compute_token_len, attribute_len, self.model.config.hidden_size,dtype=next(self.model.parameters()).dtype).to(self.model.device)
                idx_range = torch.arange(start_token_idx, end_token_idx)
                local_idx = idx_range - start_token_idx
                attribute_state[local_idx, idx_range, :] = states[0][0][idx_range, :]
                if self.ignore_special_token and cur_num !=0 :
                    special_token_state = torch.zeros(1, attribute_state.shape[1], attribute_state.shape[2],dtype=next(self.model.parameters()).dtype).to(self.model.device)
                    special_token_state[0,0,:] = states[0][0][0, :]
                    attribute_state = torch.cat([special_token_state, attribute_state], dim=0)
                attribute_state_sum = torch.sum(attribute_state, dim=0)
                left_state = states[0][0] - attribute_state_sum
                attribute_state = torch.cat([attribute_state,left_state.unsqueeze(0)], dim=0)
                all_layer_attribute_state_list = []
                all_layer_attribute_state_list.append(attribute_state[:-1,:,:].clone().detach().cpu())
                for layer_idx in range(self.num_layers):
                    state = states[layer_idx*2]
                    attribute_state = self.attn_decomposed_compute(layer_idx, state, masks, dropouts, causals, attribute_state)                    
                    state = states[layer_idx*2 + 1]
                    attribute_state = self.mlp_decomposed_compute(layer_idx, state, attribute_state,self.mlp_softmax_temp, ignore_special_token=self.ignore_special_token, cur_num=cur_num)
                    if self.ignore_special_token and cur_num !=0 :
                        all_layer_attribute_state_list.append(attribute_state[1:-1,:,:].clone().detach().cpu())
                    else:
                        all_layer_attribute_state_list.append(attribute_state[:-1,:,:].clone().detach().cpu())
                attribute_state_list.append(torch.stack(all_layer_attribute_state_list, dim=0))
            attribute_state = torch.cat(attribute_state_list, dim=1)
            attribute_state=attribute_state.transpose(1, 2)
        return attribute_state, states

    def get_mlp_neuron_decomposed_state(self, prompt, start_layer_idx, compute_num=None):
        """
        MLP Neuron Level DePass.
        Initializes the attribute state at the given layer and computes the last layer attribute state.

        Args:
            prompt (str): The input prompt for the model.
            start_layer_idx (int): The layer index for the MLP neuron-Level Initialization.
            compute_num (int, optional): The number of computation. If None, it will be set to 1.

        Returns:
            attribute_state (torch.Tensor): The computed attribute state tensor with shape:
                - N: sequence length (number of tokens)
                - M: MLP neuron number (number of neurons in the MLP layer) 
                - D: hidden dimension size (model's hidden state dimension)
                Shape: (N, M, D)
        """
        self.get_states(prompt)
        masks = self.masks
        dropouts = self.dropouts
        causals = self.causals
        states = self.states
        attribute_len = self.attribute_len
        attribute_state_list = []
        with torch.no_grad():
            # Divide the intermediate size by the number of splits
            intermediate_size = self.model.config.intermediate_size    
            if compute_num is None:
                num_splits = 1
            else:
                num_splits = compute_num
            chunk_size = math.ceil(intermediate_size / num_splits)
            for cur_neuron_num in range(num_splits):
                # Initialization of attribute state at the given layer
                start_neuron_idx = cur_neuron_num * chunk_size
                end_neuron_idx = min((cur_neuron_num + 1) * chunk_size, intermediate_size)
                attribute_state = torch.zeros(end_neuron_idx - start_neuron_idx + 1, attribute_len, self.model.config.hidden_size,dtype=next(self.model.parameters()).dtype).to(self.model.device)
                layer = self.model.model.layers[start_layer_idx]
                target_device = attribute_state.device
                state_ori = states[2*start_layer_idx+1][0].to(target_device)
                state_after_mlp = states[2*start_layer_idx+2][0].to(target_device)
                x = layer.post_attention_layernorm(state_ori)
                mlp = layer.mlp
                gate_proj = mlp.gate_proj(x)
                up_proj = mlp.up_proj(x)
                intermediate = mlp.act_fn(gate_proj) * up_proj 
                intermediate = intermediate.squeeze(0)
                down_weights = mlp.down_proj.weight
                expanded_down_weights = down_weights.T.unsqueeze(0).expand(intermediate.size(0), -1, -1)
                mlp_output = intermediate.unsqueeze(-1) * expanded_down_weights
                mlp_decompose = mlp_output[:,start_neuron_idx: end_neuron_idx,:].to(target_device)
                attribute_state[0,:,:] = state_ori - mlp_decompose.sum(1)
                attribute_state[1:,:,:] = mlp_decompose.transpose(0, 1)
                # Compute the last layer attribute state
                for layer_idx in range(start_layer_idx + 1, self.num_layers):
                    # Attn module computation
                    state = states[layer_idx*2]
                    attribute_state = self.attn_decomposed_compute(layer_idx, state, masks, dropouts, causals, attribute_state)
                    # MLP module computation
                    state = states[layer_idx*2 + 1]
                    attribute_state = self.mlp_decomposed_compute(layer_idx, state, attribute_state,self.mlp_softmax_temp, ignore_special_token=False, cur_num=0)
                # Last layernorm computation
                if cur_neuron_num !=0 :
                    attribute_state = attribute_state[1:,:,:]
                state=states[-1][0].to(attribute_state.device)
                ln = self.model.model.norm
                W_diag_elements = get_rmsnorm_scaling(ln, state)
                attribute_state = attribute_state * W_diag_elements
                attribute_state_list.append(attribute_state.clone().detach().cpu())
            attribute_state = torch.cat(attribute_state_list, dim=0)
            attribute_state=attribute_state.transpose(0, 1)
        return attribute_state
    
    def get_layer_module_decomposed_state(self, prompt, start_layer_idx, single_compute_token=None ,type=None):
        """
        Module-Level DePass.
        Initializes the attribute state at the given layer and computes the last layer attribute state.

        Args:
            prompt (str): The input prompt for the model.
            start_layer_idx (int): The layer index for the Module-Level Initialization.
            type (str, optional): Module type. It can be "attn" or "mlp" or "attn_head".

        Returns:
            attribute_state (torch.Tensor): The computed attribute state tensor with shape:
                - N: sequence length (number of tokens)
                - M: module number (number of modules in the layer) 
                - D: hidden dimension size (model's hidden state dimension)
                Shape: (N, M, D)
        """
        self.get_states(prompt)
        masks = self.masks
        dropouts = self.dropouts
        causals = self.causals
        states = self.states
        attribute_len = self.attribute_len
        if single_compute_token is None:
            single_compute_token = attribute_len + 1
        forward_num = math.ceil(self.attribute_len/single_compute_token)
        attribute_state_list = []
        with torch.no_grad():
            for cur_num in range(forward_num):
                start_token_idx = cur_num * single_compute_token
                end_token_idx = min((cur_num + 1) * single_compute_token, attribute_len)
                attribute_state = torch.zeros(2, attribute_len, self.model.config.hidden_size,dtype=next(self.model.parameters()).dtype,).to(self.model.device,)
                layer = self.model.model.layers[start_layer_idx]
                # Attention module level initialization
                if type == "attn":
                    target_device = attribute_state.device
                    state_ori = states[2*start_layer_idx][0].to(target_device)
                    state_after_attn = states[2*start_layer_idx+1][0].to(target_device)
                    attribute_state[0,:,:] = state_ori
                    attribute_state[1,:,:] = state_after_attn-state_ori
                    state = states[start_layer_idx*2 + 1]
                    attribute_state = self.mlp_decomposed_compute(start_layer_idx, state, attribute_state,self.mlp_softmax_temp, ignore_special_token=False, cur_num=cur_num)
                # MLP module level initialization 
                elif type == "mlp":
                    target_device = attribute_state.device
                    state_ori = states[2*start_layer_idx+1][0].to(target_device)
                    state_after_mlp = states[2*start_layer_idx+2][0].to(target_device)
                    attribute_state[0,:,:]= state_ori
                    attribute_state[1,:,:]= state_after_mlp-state_ori
                # Attention head level initialization
                elif type == "attn_head":
                    attribute_state = torch.zeros(self.num_heads + 1, attribute_len, self.model.config.hidden_size, dtype=next(self.model.parameters()).dtype).to(self.model.device)
                    target_device = attribute_state.device
                    state_ori = states[2*start_layer_idx][0].to(target_device)
                    state_after_attn = states[2*start_layer_idx+1][0].to(target_device)
                    state = states[2*start_layer_idx]
                    attn_output = self.attn_head_contribution_compute(start_layer_idx, state, masks, dropouts, causals, attribute_state).to(target_device)
                    attribute_state[0,:,:] = state_ori
                    attribute_state[1:,:,:] = attn_output
                    state = states[2*start_layer_idx + 1]
                    attribute_state = self.mlp_decomposed_compute(start_layer_idx, state, attribute_state,self.mlp_softmax_temp, ignore_special_token=False, cur_num=cur_num)
                else:
                    raise ValueError("type must be 'attn' or 'mlp' or 'attn_head'")
                # Compute the last layer attribute state
                for layer_idx in range(start_layer_idx + 1, self.num_layers):
                    # Attn module computation
                    state = states[layer_idx*2]
                    attribute_state = self.attn_decomposed_compute(layer_idx, state, masks, dropouts, causals, attribute_state)
                    
                    # MLP module computation
                    state = states[layer_idx*2 + 1]
                    attribute_state = self.mlp_decomposed_compute(layer_idx, state, attribute_state,self.mlp_softmax_temp, ignore_special_token=False, cur_num=cur_num)
                if self.ignore_special_token and cur_num !=0 :
                    attribute_state = attribute_state[1:,:,:]
                # Last layernorm computation
                state = states[-1][0].to(attribute_state.device)
                ln = self.model.model.norm
                W_diag_elements = get_rmsnorm_scaling(ln, state)
                attribute_state = attribute_state * W_diag_elements
                attribute_state_list.append(attribute_state.clone().detach().cpu())
            attribute_state = torch.cat(attribute_state_list, dim=0)
            attribute_state = attribute_state.transpose(0, 1)
        return attribute_state

        
    def get_subspace_decomposed_state(self,prompt,start_layer_idx,attribute_state):
        """
        Subspace-Level DePass.
        Takes externally initialized attribute state at a specified layer and computes the final layer's attribution state.

        Args:
            prompt (str): The input prompt for the model.
            start_layer_idx (int): The layer index for the Subspace-Level Initialization.
            attribute_state (torch.Tensor shape: (M, N, D)): The attribute state tensor to be used for the computation.

        Returns:
            attribute_state (torch.Tensor): The computed attribute state tensor with shape:
                - N: sequence length (number of tokens)
                - M: subspace number (number of subspaces in the layer) 
                - D: hidden dimension size (model's hidden state dimension)
                Shape: (N, M, D)
        """
        self.get_states(prompt)
        masks = self.masks
        dropouts = self.dropouts
        causals = self.causals
        states = self.states
        attribute_state_list = []
        with torch.no_grad():
            attribute_state = attribute_state.to(self.model.device)
            for layer_idx in range(start_layer_idx + 1, self.num_layers):
                state = states[layer_idx*2]
                attribute_state = self.attn_decomposed_compute(layer_idx, state, masks, dropouts, causals, attribute_state)
                state = states[layer_idx*2 + 1]
                attribute_state = self.mlp_decomposed_compute(layer_idx, state, attribute_state,self.mlp_softmax_temp, ignore_special_token=False, cur_num=0)
            state = states[-1][0].to(attribute_state.device)
            ln = self.model.model.norm
            W_diag_elements = get_rmsnorm_scaling(ln, state)
            attribute_state = attribute_state * W_diag_elements
            attribute_state_list.append(attribute_state.clone().detach().cpu())
            attribute_state = torch.cat(attribute_state_list, dim=0)
            attribute_state=attribute_state.transpose(0, 1)
        return attribute_state
    
    
    def get_subspace_decomposed_state_all_layer(self,prompt,start_layer_idx,attribute_state):
        """
        Subspace-Level DePass.
        Takes externally initialized attribute state at a specified layer and computes the final layer's attribution state.

        Args:
            prompt (str): The input prompt for the model.
            start_layer_idx (int): The layer index for the Subspace-Level Initialization.
            attribute_state (torch.Tensor shape: (M, N, D)): The attribute state tensor to be used for the computation.

        Returns:
            attribute_state (torch.Tensor): The computed attribute state tensor with shape:
                - N: sequence length (number of tokens)
                - M: subspace number (number of subspaces in the layer) 
                - D: hidden dimension size (model's hidden state dimension)
                Shape: (N, M, D)
        """
        self.get_states(prompt)
        masks = self.masks
        dropouts = self.dropouts
        causals = self.causals
        states = self.states
        attribute_state_list = []
        with torch.no_grad():
            attribute_state = attribute_state.to(self.model.device)
            for layer_idx in range(start_layer_idx + 1, self.num_layers):
                state = states[layer_idx*2]
                attribute_state = self.attn_decomposed_compute(layer_idx, state, masks, dropouts, causals, attribute_state)
                state = states[layer_idx*2 + 1]
                attribute_state = self.mlp_decomposed_compute(layer_idx, state, attribute_state,self.mlp_softmax_temp, ignore_special_token=False, cur_num=0)
                attribute_state_list.append(attribute_state.transpose(0, 1).clone().detach().cpu())
            #Final RMSNorm
            state = states[-1][0].to(attribute_state.device)
            ln = self.model.model.norm
            W_diag_elements = get_rmsnorm_scaling(ln, state)
            attribute_state = attribute_state * W_diag_elements
            attribute_state = attribute_state.transpose(0, 1)
        return attribute_state_list
    
    
    def model_generate_cite(
            self, 
            prompt,
            inputs: Optional[torch.Tensor] = None,
            max_length: Optional[int] = None,
            min_length: Optional[int] = None,
            do_sample: Optional[bool] = None,
            num_beams: Optional[int] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            typical_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            bad_words_ids: Optional[List[int]] = None,
            force_words_ids: Optional[Union[List[int], List[List[int]]]] = None,
            bos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            length_penalty: Optional[float] = None,
            no_repeat_ngram_size: Optional[int] = None,
            encoder_no_repeat_ngram_size: Optional[int] = None,
            num_return_sequences: Optional[int] = None, 
        ):
        """
        Token-Level DePass.
        Generates text from the model using the provided prompt and use DePass to compute the attribute state for new tokens.

        Args:
            prompt (str): The input prompt for the model.
            inputs (torch.Tensor, optional): The input tensor for the model. Defaults to None.
            max_length (int, optional): The maximum length of the generated text. Defaults to None.
            min_length (int, optional): The minimum length of the generated text. Defaults to None.
            do_sample (bool, optional): Whether to use sampling. Defaults to None.
            num_beams (int, optional): The number of beams for beam search. Defaults to None.
            temperature (float, optional): The temperature for sampling. Defaults to None.
            top_k (int, optional): The number of top-k tokens to sample from. Defaults to None.
            top_p (float, optional): The cumulative probability for nucleus sampling. Defaults to None.
            typical_p (float, optional): The typical probability for sampling. Defaults to None.
            repetition_penalty (float, optional): The penalty for repeated tokens. Defaults to None.
            bad_words_ids (List[int], optional): A list of token IDs to avoid. Defaults to None.
            force_words_ids (Union[List[int], List[List[int]]], optional): A list of token IDs to force. Defaults to None.
            bos_token_id (int, optional): The beginning of sequence token ID. Defaults to None.
            pad_token_id (int, optional): The padding token ID. Defaults to None.
            eos_token_id (int, optional): The end of sequence token ID. Defaults to None.
            length_penalty (float, optional): The length penalty for beam search. Defaults to None.
            no_repeat_ngram_size (int, optional): The size of n-grams to avoid repeating. Defaults to None.
            encoder_no_repeat_ngram_size (int, optional): The size of n-grams to avoid repeating in the encoder. Defaults to None.
            num_return_sequences (int, optional): The number of return sequences. Defaults to None.
        

        Returns:
            generated_text (str): The generated text from the model.
            score_list (list): A list of dictionaries containing the token and its corresponding attribute score.
            attribute_state (torch.Tensor): The computed attribute state tensor with shape:
                - N: sequence length (number of tokens)
                - N: sequence length (number of tokens) 
                - D: hidden dimension size (model's hidden state dimension)
                Shape: (N, N, D)
            states (list): A list of tensors containing the states of the model.
        """
        # Get the generated text from the model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        assert num_return_sequences == 1 or num_return_sequences == None, "num_return_sequences must be 1"
        initial_input_length = inputs["input_ids"].shape[1]
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            typical_p=typical_p,
            repetition_penalty=repetition_penalty,
            bad_words_ids=bad_words_ids,
            force_words_ids=force_words_ids,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            num_return_sequences=num_return_sequences
        )
        output_length= outputs.shape[1]
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        attribute_state, states = self.get_last_layer_attribute_state(generated_text)
        # Compute the attribute score for the generated tokens
        score_list = []
        for i in range(initial_input_length,output_length):
            dict={}
            token_id= outputs[0][i]
            token = self.tokenizer.decode(token_id)
            attribute_score = self.compute_attribute_score(attribute_state, i-1, token_id)
            dict["token"]=token
            dict["attribute_score"]=attribute_score
            score_list.append(dict)
        return generated_text, score_list, attribute_state, states
        
    def compute_attribute_score(self,attribute_state,token_idx,decode_token_id):
        """
        Given the attribute state, token index, and decoding token ID, compute the attribute score.

        Args:
            attribute_state (torch.Tensor): The attribute state tensor with shape: N, M, D
                - N: sequence length (number of tokens)
                - M: DePass component number
                - D: hidden dimension size (model's hidden state dimension)

        Returns:
            attribute_score (torch.Tensor): The computed attribute score tensor with shape: M
                - M: DePass component number
        """
        lm_head = self.model.lm_head
        attribute_state = attribute_state[token_idx].to(device=self.model.device, dtype=lm_head.weight.dtype)
        attribute_logits = lm_head(attribute_state)
        attribute_score = attribute_logits[:, decode_token_id].float()
        return attribute_score
        
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
        target_device = attribute_state.device
        layer = self.model.model.layers[layer_idx]
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
            mlp_gate = self.get_per_ffn2_values(state_norm[:,start_idx:end_idx,:], layer_idx).to(target_device)
            gate_slice = gate_ratio[:, :, start_idx:end_idx].to(target_device)
            mlp_gate = mlp_gate.squeeze(0)
            gate_slice_trans = gate_slice.permute(0, 2, 1)
            weight = layer.mlp.down_proj.weight.to(target_device)
            mlp_gate_expanded = mlp_gate.unsqueeze(0).expand(gate_slice_trans.size(0), -1, -1) 
            mlp_gate = mlp_gate_expanded * gate_slice_trans
            k, n, i = mlp_gate.shape
            d = weight.size(0)
            mlp_gate = mlp_gate.view(k * n, i)
            target_dtype = weight.dtype
            mlp_gate = mlp_gate.to(dtype=target_dtype)
            output = mlp_gate @ weight.T 
            output = output.view(k, n, d)
            attribute_mlp_values[:, start_idx:end_idx, :] += output
        attribute_state += attribute_mlp_values
        return attribute_state
    
    def mlp_decomposed_compute_max(self, layer_idx, state, attribute_state, mlp_softmax_temp=0.1, ignore_special_token=True, cur_num=None):
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
        target_device = attribute_state.device
        layer = self.model.model.layers[layer_idx]
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
        max_indices = gate_ratio.argmax(dim=0)  # (M, D)
        one_hot_gate_ratio = torch.zeros_like(gate_ratio)
        one_hot_gate_ratio.scatter_(0, max_indices.unsqueeze(0), 1.0)
        gate_ratio = one_hot_gate_ratio
        attribute_mlp_values = torch.zeros_like(attribute_state)
        mlp_token_compute_num = state_norm.shape[1]
        num_batches = (state_norm.shape[1] + mlp_token_compute_num - 1) // mlp_token_compute_num
        
        for i in range(num_batches):
            start_idx = i * mlp_token_compute_num
            end_idx = min((i + 1) * mlp_token_compute_num, state_norm.shape[1])
            mlp_gate = self.get_per_ffn2_values(state_norm[:,start_idx:end_idx,:], layer_idx).to(target_device)
            gate_slice = gate_ratio[:, :, start_idx:end_idx].to(target_device)
            mlp_gate = mlp_gate.squeeze(0)
            gate_slice_trans = gate_slice.permute(0, 2, 1)
            weight = layer.mlp.down_proj.weight.to(target_device)
            mlp_gate_expanded = mlp_gate.unsqueeze(0).expand(gate_slice_trans.size(0), -1, -1) 
            mlp_gate = mlp_gate_expanded * gate_slice_trans
            k, n, i = mlp_gate.shape
            d = weight.size(0)
            mlp_gate = mlp_gate.view(k * n, i)
            target_dtype = weight.dtype
            mlp_gate = mlp_gate.to(dtype=target_dtype)
            output = mlp_gate @ weight.T 
            output = output.view(k, n, d)
            attribute_mlp_values[:, start_idx:end_idx, :] += output
        attribute_state += attribute_mlp_values
        return attribute_state
    
    def mlp_decomposed_compute_linear(self, layer_idx, state, attribute_state, mlp_softmax_temp=0.1, ignore_special_token=True, cur_num=None):
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
        target_device = attribute_state.device
        layer = self.model.model.layers[layer_idx]
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
        safe_gate_ratio = gate_ratio.clone()
        safe_gate_ratio[safe_gate_ratio == float('-inf')] = torch.finfo(gate_ratio.dtype).min
        min_val = safe_gate_ratio.min(dim=0, keepdim=True)[0]
        shifted = torch.relu(safe_gate_ratio - min_val) * mask
        norm = shifted.sum(dim=0, keepdim=True) + 1e-6
        gate_ratio = shifted / norm
        attribute_mlp_values = torch.zeros_like(attribute_state)
        mlp_token_compute_num = state_norm.shape[1]
        num_batches = (state_norm.shape[1] + mlp_token_compute_num - 1) // mlp_token_compute_num
        
        for i in range(num_batches):
            start_idx = i * mlp_token_compute_num
            end_idx = min((i + 1) * mlp_token_compute_num, state_norm.shape[1])
            mlp_gate = self.get_per_ffn2_values(state_norm[:,start_idx:end_idx,:], layer_idx).to(target_device)
            gate_slice = gate_ratio[:, :, start_idx:end_idx].to(target_device)
            mlp_gate = mlp_gate.squeeze(0)
            gate_slice_trans = gate_slice.permute(0, 2, 1)
            weight = layer.mlp.down_proj.weight.to(target_device)
            mlp_gate_expanded = mlp_gate.unsqueeze(0).expand(gate_slice_trans.size(0), -1, -1) 
            mlp_gate = mlp_gate_expanded * gate_slice_trans
            k, n, i = mlp_gate.shape
            d = weight.size(0)
            mlp_gate = mlp_gate.view(k * n, i)
            target_dtype = weight.dtype
            mlp_gate = mlp_gate.to(dtype=target_dtype)
            output = mlp_gate @ weight.T 
            output = output.view(k, n, d)
            attribute_mlp_values[:, start_idx:end_idx, :] += output
        attribute_state += attribute_mlp_values
        return attribute_state

    def mlp_decomposed_compute_relu(self, layer_idx, state, attribute_state, mlp_softmax_temp=0.1, ignore_special_token=True, cur_num=None):
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
        target_device = attribute_state.device
        layer = self.model.model.layers[layer_idx]
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
        
        gate_ratio = torch.relu(gate_ratio)
        norm = gate_ratio.sum(dim=0, keepdim=True) + 1e-6
        gate_ratio = gate_ratio / norm

        attribute_mlp_values = torch.zeros_like(attribute_state)
        mlp_token_compute_num = state_norm.shape[1]
        num_batches = (state_norm.shape[1] + mlp_token_compute_num - 1) // mlp_token_compute_num
        
        for i in range(num_batches):
            start_idx = i * mlp_token_compute_num
            end_idx = min((i + 1) * mlp_token_compute_num, state_norm.shape[1])
            mlp_gate = self.get_per_ffn2_values(state_norm[:,start_idx:end_idx,:], layer_idx).to(target_device)
            gate_slice = gate_ratio[:, :, start_idx:end_idx].to(target_device)
            mlp_gate = mlp_gate.squeeze(0)
            gate_slice_trans = gate_slice.permute(0, 2, 1)
            weight = layer.mlp.down_proj.weight.to(target_device)
            mlp_gate_expanded = mlp_gate.unsqueeze(0).expand(gate_slice_trans.size(0), -1, -1) 
            mlp_gate = mlp_gate_expanded * gate_slice_trans
            k, n, i = mlp_gate.shape
            d = weight.size(0)
            mlp_gate = mlp_gate.view(k * n, i)
            target_dtype = weight.dtype
            mlp_gate = mlp_gate.to(dtype=target_dtype)
            output = mlp_gate @ weight.T 
            output = output.view(k, n, d)
            attribute_mlp_values[:, start_idx:end_idx, :] += output
        attribute_state += attribute_mlp_values
        return attribute_state

    def mlp_decomposed_compute_taylor(self, layer_idx, state, attribute_state, mlp_softmax_temp=0.1, ignore_special_token=True, cur_num=None):
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
        target_device = attribute_state.device
        layer = self.model.model.layers[layer_idx]
        ln = layer.post_attention_layernorm
        gate_proj = layer.mlp.gate_proj
        target_dtype = gate_proj.weight.dtype
        state = state.to(device=target_device, dtype=target_dtype)
        state_norm = ln(state)
        W_diag_elements = get_rmsnorm_scaling(ln, state).to(device=target_device)
        W_diag_elements = W_diag_elements.to(dtype=target_dtype)        
        attribute_state_norm = attribute_state.to(dtype=target_dtype) * W_diag_elements
        mlp = layer.mlp
        gate = mlp.gate_proj(state_norm)
        gate_act = mlp.act_fn(gate)
        coef = gate_act/(gate + 1e-8)
        gate_attribute = mlp.gate_proj(attribute_state_norm)
        gate_attribute = gate_attribute * coef
        attribute_mlp_values = mlp.down_proj(gate_attribute*mlp.up_proj(state_norm))
        attribute_state += attribute_mlp_values.to(attribute_state.device)
        return attribute_state

    def attn_decomposed_compute(self,layer_idx,state,masks,dropouts,causals,attribute_state):
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
        layer = self.model.model.layers[layer_idx]
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj
        ln = layer.input_layernorm
        state_norm = ln(state)
        attn_weight = self.attention_weight_compute(state_norm, layer.self_attn, layer_idx, masks[layer_idx], dropouts[layer_idx], causals[layer_idx],self.apply_rotary_pos_emb,self.model_name)
        W_diag_elements = get_rmsnorm_scaling(ln, state)
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
    
    def get_per_ffn2_values(self, state, layer_idx):
        # Get neuron-level values for the MLP module
        layer = self.model.model.layers[layer_idx].mlp
        gate_up_output = layer.act_fn(layer.gate_proj(state)) * layer.up_proj(state)
        return gate_up_output

    def attn_head_contribution_compute(self,layer_idx,state,masks,dropouts,causals,attribute_state):
        # Get head-level values for the Attention module
        target_device = attribute_state.device
        layer = self.model.model.layers[layer_idx]
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj
        ln = layer.input_layernorm
        state_norm = ln(state).to(v_proj.weight.device)
        attn_weight = self.attention_weight_compute(state_norm, layer.self_attn, layer_idx, masks[layer_idx], dropouts[layer_idx], causals[layer_idx],self.apply_rotary_pos_emb,self.model_name)
        v_proj_weight = v_proj.weight.view(self.num_key_value_heads, self.head_dim, self.hidden_dim).to(target_device)
        o_proj_weight = o_proj.weight.view(self.hidden_dim, self.num_heads, self.head_dim).to(target_device)
        attn_weight = attn_weight[0].to(target_device)
        state_norm = state_norm.to(target_device)
        attribute_values = torch.einsum("kqi,nhi->knqh", state_norm, v_proj_weight)
        attribute_values = repeat_kv(attribute_values, self.num_heads // self.num_key_value_heads)
        vo_attribute = torch.einsum("knqh,jnh->kqnj", attribute_values, o_proj_weight)
        attn_output = torch.einsum("iknd, nqk -> inqd", vo_attribute, attn_weight)[0]
        return attn_output
