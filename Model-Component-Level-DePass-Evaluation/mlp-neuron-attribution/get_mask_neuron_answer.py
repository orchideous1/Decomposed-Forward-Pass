import sys
import os
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root_dir)
from DePass import decomposed_state_manager
import json
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import argparse

# nohup python -u get_mask_neuron_answer.py \
#     --model_name "llama-2-7b-chat-hf" \
#     --model_path "/root/models/transformers/llama-2/llama-2-7b-chat-hf/" \
#     --dataset "counterfact_data.json" \
#     --device "auto" \
#     --output_dir "./results" \
#     --dtype "bfloat16" \
#     --compute_num 12 \
#     --mask_top_percents  0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.20 1 \
#     --mask_bottom_percents 1 5 10 20 30 40 50 60 70 75 80 83 85 88 90 93 95 \
#     --max_samples 100 \
#     --results_exist False > output13b.log 2>&1


def parse_args():
    parser = argparse.ArgumentParser(description='Get model answers')
    parser.add_argument('--model_name', type=str, default='llama-2-7b-chat-hf',
                        help='Name of the model')
    parser.add_argument('--model_path', type=str, 
                        default='/root/models/transformers/llama-2/llama-2-7b-chat-hf',
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
    parser.add_argument('--mask_top_percents', type=float, nargs='+', default=[1.0],
                        help='List of percentages for masking (e.g., 1.0 5.0 10.0 for 1%, 5%, 10%)')
    parser.add_argument('--mask_bottom_percents', type=float, nargs='+', default=[1.0],
                        help='List of percentages for masking (e.g., 1.0 5.0 10.0 for 1%, 5%, 10%)')
    parser.add_argument('--compute_num', type=int, default=5,
                        help='number of compute')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='number of samples')
    parser.add_argument('--results_exist', 
                       type=lambda x: x.lower() in ['true', '1', 'yes'],  # Convert string to bool
                       default=False,
                       help='whether the results exist')
    
    return parser.parse_args()


def compute_neuron_importance(model, tokenizer, prompt):
    model.requires_grad_(True) 
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
    activation_store = {}
    gradient_store = {}
    def register_hooks(model):
        hook_handles = []        
        for layer_idx, block in enumerate(model.model.layers):
            mlp = block.mlp
            def make_hook(layer_idx):
                def forward_pre_hook(module, input):
                    activation = input[0].detach()
                    activation_store[layer_idx] = activation
                    def grad_hook(grad):
                        gradient_store[layer_idx] = grad.detach()
                    input[0].register_hook(grad_hook)
                return forward_pre_hook
            handle = mlp.down_proj.register_forward_pre_hook(make_hook(layer_idx))
            hook_handles.append(handle)
        return hook_handles
    hooks = register_hooks(model)    
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    loss = nn.CrossEntropyLoss()(logits, next_token)
    loss.backward()    
    target_position = len(inputs["input_ids"][0]) - 1
    importance_by_layer = {}
    for layer_idx in activation_store.keys():
        act = activation_store[layer_idx][0, target_position]
        grad = gradient_store[layer_idx][0, target_position]
        importance = (act * grad).abs().float().cpu()
        importance_by_layer[layer_idx] = importance
    for h in hooks:
        h.remove()        
    layers = sorted(importance_by_layer.keys())
    atp_importance = np.stack([importance_by_layer[layer].numpy() for layer in layers])
    coef_importance = np.stack([activation_store[layer][0, target_position].float().cpu().numpy() for layer in layers])
    weight_importance = []
    for layer_idx in activation_store.keys():
        mlp_weight = model.model.layers[layer_idx].mlp.down_proj.weight
        act = activation_store[layer_idx][0, target_position]
        weighted_activation = (mlp_weight * act.unsqueeze(0)).cpu()
        neuron_norm = torch.norm(weighted_activation, dim=0)
        weight_importance.append(neuron_norm.detach().float().numpy())
    norm_importance = np.stack(weight_importance)
    model.requires_grad_(False)
    return atp_importance, coef_importance, norm_importance

def apply_neuron_mask_and_generate(model, tokenizer, prompt, ground_truth, device, neuron_scores, mask_type, mask_percents):
    results = {}
    for mask_percent in mask_percents:
        if mask_type == "top":
            percentile = 100 - mask_percent
            threshold = np.percentile(neuron_scores, percentile)
            mask = neuron_scores >= threshold
        else:  # bottom
            percentile = mask_percent
            threshold = np.percentile(neuron_scores, percentile)
            mask = neuron_scores <= threshold
        indices = np.argwhere(mask)
        layer_indices = {}
        for layer_idx, neuron_idx in indices:
            if layer_idx not in layer_indices:
                layer_indices[layer_idx] = []
            layer_indices[layer_idx].append(neuron_idx)
        for layer_idx, neuron_indices in layer_indices.items():
            neuron_indices = torch.tensor(neuron_indices, device=device)
            model.model.layers[layer_idx].mlp.down_proj.weight[:, neuron_indices] = 0
        answer_masked = generate_answer(model, tokenizer, prompt, device)
        is_correct_masked = check_answer_match(answer_masked, ground_truth)
        results[f"answer_masked_{mask_type}_{mask_percent}"] = answer_masked
        results[f"correct_masked_{mask_type}_{mask_percent}"] = is_correct_masked    
    return results

def normalize_string(text):
    """Normalize a string by converting to lowercase and removing punctuation and extra whitespace"""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def check_answer_match(answer, ground_truth):
    """Check if normalized answer starts with normalized ground truth"""
    norm_answer = normalize_string(answer)
    norm_truth = normalize_string(ground_truth)
    return norm_answer.startswith(norm_truth)

def generate_answer(model, tokenizer, prompt, device, max_new_tokens=5):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(outputs[0][-max_new_tokens:], skip_special_tokens=True).strip()


def get_neuron_scores_decompose_based(AttrStateManager, prompt, decode_token,compute_num=1):
    scores=[]
    for layer_idx in range(0, AttrStateManager.model.config.num_hidden_layers):
        print(layer_idx)
        attr_state_neuron = AttrStateManager.get_mlp_neuron_decomposed_state(prompt, layer_idx, compute_num=compute_num)
        score = AttrStateManager.compute_attribute_score(attr_state_neuron, -1, decode_token)
        scores.append(score[1:].detach().cpu().numpy())
    scores = np.array(scores)
    return scores

def normalize_model_name(name):
    """Normalize model name to standardized format"""
    name = name.lower().strip()
    name = name.replace("-hf", "")
    name = name.replace("meta-", "")
    return name


def main():
    args = parse_args()
    model_path = args.model_path
    dataset_name = args.dataset.split('.')[0]
    compute_num = args.compute_num
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
    top_mask_percents = args.mask_top_percents
    bottom_mask_percents = args.mask_bottom_percents
    top_mask_percents.sort()
    bottom_mask_percents.sort()
    
    # === Data Loading ===
    data_path = os.path.join("../data", args.dataset)
    try:
        with open(data_path, 'r') as f:
            data_all = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        return
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {data_path}")
        return

    # === AttrStateManager Initialization ===
    AttrStateManager = decomposed_state_manager(model, tokenizer)
    
    
    original_params = {}
    for layer_idx in range(0, model.config.num_hidden_layers):
        current_device = model.model.layers[layer_idx].mlp.down_proj.weight.device
        original_params[layer_idx] = {
            'params': model.model.layers[layer_idx].mlp.down_proj.weight.clone().cpu(),
            'device': current_device
        }
    print(args.results_exist)
    if args.results_exist:
        input_path = os.path.join(args.output_dir, f"{args.model_name}_mask_{dataset_name}_results.json")
        with open(input_path, 'r') as f:
            results = json.load(f)
        existing_prompts = {result["prompt"] for result in results}
        print(f"Results loaded from {input_path}, {len(existing_prompts)} existing prompts found")
    else:
        results = []
        existing_prompts = set()

    max_samples = args.max_samples
    methods=['DePass','DePass_abs','random','atp', 'coef', 'norm']
    mask_types=['top', 'bottom']
    for data in data_all:
        case_id = data["case_id"]
        prompt = data["prompt"]
        if prompt in existing_prompts:
            continue
        ground_truth = data["answer"]
        answer = generate_answer(model, tokenizer, prompt, device)
        is_correct = check_answer_match(answer, ground_truth)
        
        print(f"len(results): {len(results)}")
        if len(results) >= max_samples:
            break
        
        if not is_correct:
            continue
        else:
            result = {
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                    "answer": answer,
                    "case_id": case_id,
                }
            answer_tokens = tokenizer.encode(answer)
            target_token = answer_tokens[1] if len(answer_tokens) > 0 else None
            importances={}
            importances['atp'], importances['coef'], importances['norm'] = compute_neuron_importance(model, tokenizer, prompt)
            importances['DePass'] = get_neuron_scores_decompose_based(AttrStateManager, prompt,  target_token, compute_num=compute_num)
            importances['DePass_abs'] = np.abs(importances['DePass'])
            importances['random'] = np.random.rand(*importances['DePass'].shape)
            
            for method in methods:
                for mask_type in mask_types:
                    if mask_type == 'top':
                        mask_percents = top_mask_percents
                    elif mask_type == 'bottom':
                        mask_percents = bottom_mask_percents
                    importance = importances[method]
                    mask_result = apply_neuron_mask_and_generate(model, tokenizer, prompt, ground_truth, device, importance, mask_type, mask_percents)
                    result[f"answer_masked_{method}_{mask_type}"] = mask_result
                                        
                    for layer_idx in range(0, model.config.num_hidden_layers):
                        orig_state = original_params[layer_idx]
                        model.model.layers[layer_idx].mlp.down_proj.weight.data.copy_(
                            orig_state['params'].to(orig_state['device'])
                        )
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, f"{args.model_name}_mask_{dataset_name}_results.json")
            importance_score_save_path = os.path.join(args.output_dir, f"scores/{dataset_name}/{args.model_name}/{case_id}.pt")
            if not os.path.exists(os.path.dirname(importance_score_save_path)):
                os.makedirs(os.path.dirname(importance_score_save_path), exist_ok=True)
            torch.save(importances['DePass'], importance_score_save_path)
            
            if not os.path.exists(output_path):
                with open(output_path, 'w') as f:
                    json.dump([], f)
            with open(output_path, 'r') as f:
                current_results = json.load(f)
            current_results.append(result)
            with open(output_path, 'w') as f:
                json.dump(current_results, f, indent=4)
            print(f"Result saved to {output_path}, current total: {len(current_results)}")
            results.append(result)


if __name__ == "__main__":
    main()