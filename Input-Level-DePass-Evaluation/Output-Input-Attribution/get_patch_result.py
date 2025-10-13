import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import numpy as np
import os
import json
import math

# python get_patch_result.py \
#     --model_name llama-2-7b-chat-hf \
#     --model_path /root/models/transformers/llama-2/llama-2-7b-chat-hf \
#     --patch_type patch_top \
#     --dataset known_1000.json \
#     --output_dir ./patch_results \
#     --device cuda:0 \
#     --dtype float16

def parse_args():
    parser = argparse.ArgumentParser(description='Patch result evaluation script')
    parser.add_argument('--model_name', type=str, default='llama-2-7b-chat-hf',
                        help='Name of the model')
    parser.add_argument('--model_path', type=str, 
                        default='/root/models/transformers/llama-2/llama-2-7b-chat-hf',
                        help='Path to the model')
    parser.add_argument('--patch_type', type=str, 
                        default='patch_top',
                        help='patch top or recover top')
    parser.add_argument('--dataset', type=str, 
                        default='known_1000.json',
                        help='dataset filename')
    parser.add_argument('--input_dir', type=str, 
                        default='./importance_scores',
                        help='Path to importance scores')
    parser.add_argument('--output_dir', type=str, 
                        default='./patch_results',
                        help='Path to save results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'float32', 'bfloat16'],
                        help='Floating point precision type')
    return parser.parse_args()


def patch_top(prompt, tokenizer, importance_scores, mask_percent):
    """
    Remove tokens with highest importance scores and return decoded prompt.
    Special tokens and the last token are excluded from masking.
    """
    if not isinstance(importance_scores, torch.Tensor):
        importance_scores = torch.tensor(importance_scores)
    if importance_scores.dim() > 1:
        importance_scores = importance_scores.squeeze()
    tokens_with_special = tokenizer.encode(prompt, add_special_tokens=True)
    tokens_without_special = tokenizer.encode(prompt, add_special_tokens=False)
    has_special_tokens = len(tokens_with_special) > len(tokens_without_special)
    seq_len = len(importance_scores)
    assert seq_len == len(tokens_with_special), f"Importance scores length ({seq_len}) != token length ({len(tokens_with_special)})"
    keep_mask = torch.ones(seq_len, dtype=torch.bool)
    if has_special_tokens:
        if seq_len <= 2:
            return prompt
        processable_scores = importance_scores[1:-1]
        processable_len = len(processable_scores)
        if processable_len > 0:
            k = max(1, int(processable_len * mask_percent))
            k = min(k, processable_len)
            topk_indices = processable_scores.topk(k).indices
            absolute_indices = topk_indices + 1
            keep_mask[absolute_indices] = False
        keep_mask[0] = True
        keep_mask[-1] = True
    else:
        if seq_len <= 1:
            return prompt
        processable_scores = importance_scores[:-1]
        processable_len = len(processable_scores)
        if processable_len > 0:
            k = max(1, int(processable_len * mask_percent))
            k = min(k, processable_len)
            topk_indices = processable_scores.topk(k).indices
            keep_mask[topk_indices] = False
        keep_mask[-1] = True
    kept_tokens = [tokens_with_special[i] for i in range(seq_len) if keep_mask[i]]
    if len(kept_tokens) == 0:
        return ""
    masked_prompt = tokenizer.decode(kept_tokens, skip_special_tokens=True)
    return masked_prompt

def recover_top(prompt, tokenizer, importance_scores, keep_percent):
    """
    Keep tokens with highest importance scores and remove others.
    Special tokens and the last token are always preserved.
    """
    if not isinstance(importance_scores, torch.Tensor):
        importance_scores = torch.tensor(importance_scores)
    if importance_scores.dim() > 1:
        importance_scores = importance_scores.squeeze()
    tokens_with_special = tokenizer.encode(prompt, add_special_tokens=True)
    tokens_without_special = tokenizer.encode(prompt, add_special_tokens=False)
    has_special_tokens = len(tokens_with_special) > len(tokens_without_special)
    seq_len = len(importance_scores)
    assert seq_len == len(tokens_with_special), f"Importance scores length ({seq_len}) != token length ({len(tokens_with_special)})"
    keep_mask = torch.zeros(seq_len, dtype=torch.bool)
    if has_special_tokens:
        if seq_len <= 2:
            return prompt
        processable_scores = importance_scores[1:-1]
        processable_len = len(processable_scores)
        
        if processable_len > 0:
            k = max(1, int(processable_len * keep_percent))
            k = min(k, processable_len)
            topk_indices = processable_scores.topk(k).indices
            absolute_indices = topk_indices + 1
            keep_mask[absolute_indices] = True
        keep_mask[0] = True
        keep_mask[-1] = True
    else:
        if seq_len <= 1:
            return prompt
        processable_scores = importance_scores[:-1]
        processable_len = len(processable_scores)
        if processable_len > 0:
            k = max(1, int(processable_len * keep_percent))
            k = min(k, processable_len)
            topk_indices = processable_scores.topk(k).indices
            keep_mask[topk_indices] = True
        keep_mask[-1] = True
    kept_tokens = [tokens_with_special[i] for i in range(seq_len) if keep_mask[i]]
    if len(kept_tokens) == 0:
        return ""
    filtered_prompt = tokenizer.decode(kept_tokens, skip_special_tokens=True)
    return filtered_prompt


def main():
    args = parse_args()
    model_name = args.model_name
    dataset = args.dataset
    patch_type = args.patch_type
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    torch_dtype = dtype_map[args.dtype]
    model_path = args.model_path
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    input_filename = f"{args.model_name}_{args.dataset}"
    data_path = os.path.join(args.input_dir, input_filename)
    with open(data_path, 'r') as f:
        data_all = json.load(f)
    if args.patch_type == "patch_top":
        mask_prompt = patch_top
    elif args.patch_type == "recover_top":
        mask_prompt = recover_top
    else:
        raise ValueError("Invalid patch type. Choose 'patch_top' or 'recover_top'.")
    types=[
        "all",
        "last",
        "rollout",
        "integrated_gradients",
        "signed",
        "norm",
        "DePass"
    ]
    
    percents = np.round(np.arange(0.1, 1.1, 0.1), decimals=1).tolist()

    from tqdm import tqdm

    data_add_noise = []
    for i, data in tqdm(enumerate(data_all), total=len(data_all), desc="Processing samples"):
        has_nan = False
        for type in types:
            if isinstance(data[type], list):
                if any(math.isnan(x) if isinstance(x, (int, float)) else False for x in data[type]):
                    has_nan = True
                    break
            elif isinstance(data[type], (int, float)) and math.isnan(data[type]):
                has_nan = True
                break
        
        if has_nan:
            print(f"Skipping sample {data['id']} due to NaN values")
            continue
        noise_data = {}
        noise_data["id"] = data["id"]
        noise_data["prompt"] = data["prompt"]
        noise_data["target_token"] = data["target_token"]
        prompt = data['prompt']
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        target_token = data['target_token']
        target_token_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
        with torch.no_grad():
            base_outputs = model(**inputs)
            base_logits = base_outputs.logits
        final_logits = base_logits[0, -1]
        clean_probs = torch.softmax(final_logits, dim=-1)
        target_prob_clean = clean_probs[target_token_id].item()
        noise_data["base_prob"] = target_prob_clean

        for k, percent in enumerate(percents):
            masks = []
            noise_percents = []
            for j, type in enumerate(types):
                masked_prompt = mask_prompt(prompt, tokenizer, data[type], percent)
                with torch.no_grad():
                    filtered_input_ids = tokenizer(masked_prompt, return_tensors="pt").to(device).input_ids
                    outputs = model(filtered_input_ids)
                    noisy_logits = outputs.logits[0, -1]
                    noisy_probs = torch.softmax(noisy_logits, dim=-1)
                    target_prob_noisy = noisy_probs[target_token_id].item()
                    noise_percent = abs(target_prob_clean - target_prob_noisy) / target_prob_clean * 100
                    if noise_percent > 100:
                        noise_percent = 100
                    noise_percents.append(noise_percent)
            for j, type in enumerate(types):
                noise_data[type + "_noise_" + str(percent)] = noise_percents[j]
        data_add_noise.append(noise_data)

    dataset = dataset.split(".")[0]
    file_name = f"{model_name}_{dataset}_{patch_type}.json"
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, file_name)
    with open(output_path, 'w') as f:
        json.dump(data_add_noise, f, indent=4)
    print(f"Results saved to {output_path}")



if __name__ == "__main__":
    main()