# %%
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from DePass import decomposed_state_manager
from DePass.utils import from_states_to_probs
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer,  Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import plotly.io as pio
from plotly.subplots import make_subplots

model_path = "/ssd/linyiwu/Qwen2.5-Omni-7B/" # The path to model
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2'
    
)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
model = model.eval() 
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

# %%
sys.path.append("/home/linyiwu/OmniZip")
from omnizip.note_modeling_qwen2_5_omni import (
    Qwen2_5OmniAttention,       # Eager
    Qwen2_5OmniFlashAttention2, # Flash Attention 2
    Qwen2_5OmniSdpaAttention    # SDPA
)
def swap_attention_module(module, layer_idx, replacing_attn):
    old_attn = module.self_attn
    old_config = module.self_attn.config
    current_device = old_attn.q_proj.weight.device
    current_dtype = old_attn.q_proj.weight.dtype

    new_attn = replacing_attn(old_config, layer_idx=layer_idx)
    new_attn.load_state_dict(old_attn.state_dict())
    new_attn.to(current_device, dtype=current_dtype)
    module.self_attn = new_attn
    print(f"Layer {layer_idx} attention swapped to {replacing_attn.__name__}")


for layer_idx, layer in enumerate(model.thinker.model.layers):
    swap_attention_module(layer, layer_idx, Qwen2_5OmniSdpaAttention)

# %%
# prepare data

video_path = "/ssd/linyiwu/worldsense/videos/FnOIAada.mp4"
question = "What is the number on the rear wing of the red vehicle at the start of the video?"
task_candidates = [               
    "A. 56.",
    "B. 50.",
    "C. 58.",
    "D. 59."
]
conversation = [
    {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": f"{question}\nChoose from the following options:\n" + "\n".join(task_candidates) + "\nPlease provide your answer by selecting the corresponding option and give your answer straight away."},
        ],
    },
]


# %%
from DePass import qwenOmnimanager
# print(type(decomposed_state_manager))
# print(type(qwenOmnimanager))
# with open("/home/linyiwu/OmniZip/print_result.md", "w") as f:
#     print(model.thinker.config.text_config, file=f)

DecomposedStateManager = qwenOmnimanager(model, processor, mlp_decomposed_function="softmax")


# %%
states, attribute_state, mod_names = DecomposedStateManager.get_last_layer_decomposed_state(conversation)




# %%



