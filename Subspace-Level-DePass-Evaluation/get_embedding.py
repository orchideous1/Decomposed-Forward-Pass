
import sys
import os
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root_dir)
from DePass import decomposed_state_manager
import os
import json
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import argparse

# python -u get_embedding.py \
#     --model_name "llama-2-7b-chat-hf" \
#     --model_path "/root/models/transformers/llama-2/llama-2-7b-chat-hf/" \
#     --dataset "counterfact.json" \
#     --device "auto" \
#     --output_dir "./results" \
#     --dtype "bfloat16" \
#     --layer_idx 20

def parse_args():
    parser = argparse.ArgumentParser(description='Get model answers')
    parser.add_argument('--model_name', type=str, default='llama_3_1_8b_instruct',
                        help='Name of the model')
    parser.add_argument('--model_path', type=str, 
                        default='/root/models/llama_3_1_8b_instruct/',
                        help='Path to the model')
    parser.add_argument('--dataset', type=str, default='counterfact',
                        help='Dataset filename')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'float32', 'bfloat16'],
                        help='Floating point precision type')
    parser.add_argument('--layer_idx', type=int, default=20, help='The layer to fuse languages')
    
    return parser.parse_args()


def normalize_model_name(name):
    """Normalize model name to standardized format"""
    name = name.lower().strip()
    name = name.replace("-hf", "")
    name = name.replace("meta-", "")
    return name

def get_top_predictions(outputs, tokenizer, top_k=5):
    """
    Get top k predictions and their probabilities from model outputs
    Args:
        outputs: model output from forward pass
        tokenizer: tokenizer for decoding predictions
        top_k: number of top predictions to return (default 5)
    Returns:
        list of tuples (prediction, probability)
    """
    logits = outputs.logits[:, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
    top_probs = top_probs[0].tolist()
    top_indices = top_indices[0].tolist()
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.decode(idx)
        predictions.append((token, float(prob)))
    return predictions


def get_state_predictions(model, tokenizer, state, k=5):
    logits = model.lm_head(state)
    traj_log_probs = torch.from_numpy(
        logits.log_softmax(dim=-1).squeeze().detach().cpu().numpy()
    )
    topk_indices = torch.topk(traj_log_probs, k=k)
    probs = torch.exp(traj_log_probs[topk_indices.indices])
    token_probs = []
    for idx, prob in zip(topk_indices.indices, probs):
        token = tokenizer.decode(idx)
        token_probs.append((idx.item(), token, prob.item()))
    
    return token_probs

def main():
    args = parse_args()
    model_path = args.model_path
    dataset_name = args.dataset.split('.')[0]
    layer_idx = args.layer_idx
    model_type = normalize_model_name(args.model_name)
    if "llama" in model_type:
        model_type = "llama"
    elif "qwen" in model_type:
        model_type = "qwen"
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    torch_dtype = dtype_map[args.dtype]
    if args.device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch_dtype
        )
        device = model.device
    else:
        device = torch.device(args.device)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype
        ).to(device)
    model.requires_grad_(False) 
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    
    data_path = os.path.join("./data", args.dataset)
    try:
        with open(data_path, 'r') as f:
            data_all = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        return
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {data_path}")
        return

    AttrStateManager = decomposed_state_manager(model, tokenizer)

    attribute_state_all = {}
    for data in tqdm(data_all, desc="Processing data", position=0, leave=True):
        hidden_states_all = {}
        for lang, prompt_data in data.items():
            if lang == "case_id":
                continue
            prompt = prompt_data["prompt"]
            answer = prompt_data["answer"]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            hidden_states_all[lang] = hidden_states
            predictions= get_top_predictions(outputs, tokenizer)
            prompt_data["predictions"] = predictions
        mean_state = torch.zeros(model.config.hidden_size).to(model.device)
        for lang, hidden_states in hidden_states_all.items():
            mean_state += hidden_states_all[lang][layer_idx][0][-1]
        mean_state /= len(hidden_states_all)
        for lang,prompt_data in data.items():
            if lang == "case_id":
                continue
            prompt = prompt_data["prompt"]
            answer = prompt_data["answer"]
            hidden_states = hidden_states_all[lang]
            attribute_state=torch.zeros(2,hidden_states[layer_idx].shape[1],hidden_states[layer_idx].shape[2]).to(model.device)
            attribute_state[0, :] = hidden_states[layer_idx][0].clone()
            language_embedding=attribute_state[0, -1] - mean_state
            attribute_state[0, :] = attribute_state[0, :] - language_embedding
            attribute_state[1, :] = language_embedding
            
            attribute_state = AttrStateManager.get_subspace_decomposed_state(prompt,start_layer_idx=layer_idx-1,attribute_state=attribute_state)
            prompt_data["lang_embedding"] = attribute_state[-1][1].cpu().numpy() 
            prompt_data["semantic_embedding"] = attribute_state[-1][0].cpu().numpy()
            
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(args.output_dir, f"{args.model_name}_{dataset_name}_results.pt")
    
    for data in data_all:
        for lang, prompt_data in data.items():
            if lang == "case_id":
                continue
            if "lang_embedding" in prompt_data:
                prompt_data["lang_embedding"] = torch.from_numpy(prompt_data["lang_embedding"])
            if "semantic_embedding" in prompt_data:
                prompt_data["semantic_embedding"] = torch.from_numpy(prompt_data["semantic_embedding"])
    torch.save(data_all, output_path)
    print(f"Results saved to {output_path}")
            
if __name__ == "__main__":
    main()