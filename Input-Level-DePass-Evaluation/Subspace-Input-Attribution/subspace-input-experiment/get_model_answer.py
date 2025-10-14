import sys
import os
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..'))
sys.path.append(project_root_dir)
from DePass import decomposed_state_manager
import os
import json
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


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
    parser.add_argument('--type', type=str, default='truthful', help='Type of probing (e.g., truthful, safety)')
    parser.add_argument('--output_dir', type=str, default='./results',
    
                        help='Directory to save results')
    parser.add_argument('--classifier_start_layer', type=int, default=10,
                        help='The start layer of classifier')
    parser.add_argument('--classifier_end_layer', type=int, default=-1,
                        help='The end layer of classifier')
    parser.add_argument('--max_mask_percent', type=float, default=0.5,
                        help='The maximun percent of mask tokens')
    parser.add_argument('--classifier_bound', type=bool, default=False,
                        help='whether to use classifier bound')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'float32', 'bfloat16'],
                        help='Floating point precision type')
    return parser.parse_args()


def get_all_token_truthful_prob(classifiers, states, start_layer, end_layer, device):
    predictions = []
    input_len = states[0].shape[1]
    for token_idx in range(input_len):
        layer_predictions = []
        for layer in range(start_layer, end_layer):
            classifier = classifiers[layer]
            prediction = classifier.predict_proba(states[2*layer][0][token_idx])
            prediction_tensor = torch.tensor(prediction, device=device)
            layer_predictions.append(prediction_tensor)
        avg_prediction = torch.mean(torch.stack(layer_predictions), dim=0)
        predictions.append(avg_prediction.cpu().numpy())
    return predictions


def get_untruthful_token_indices(predictions, max_mask_percent, num_mask_tokens=None, classifier_bound=False):
    predictions = np.array(predictions)
    num_tokens = len(predictions)
    if num_mask_tokens is None:
        num_mask_tokens = int(num_tokens * max_mask_percent) + 1
    if classifier_bound:
        low_prob_mask = (predictions < 0.5) & (np.arange(len(predictions)) != 0)
    else:
        low_prob_mask = (np.arange(len(predictions)) != 0)
    low_prob_indices = np.where(low_prob_mask)[0]
    if classifier_bound:
        actual_mask_tokens = min(num_mask_tokens, len(low_prob_indices))
    else:
        actual_mask_tokens = num_mask_tokens

    sorted_low_indices = low_prob_indices[np.argsort(predictions[low_prob_indices])]
    selected_indices = sorted_low_indices[:actual_mask_tokens]
    return selected_indices.tolist()


def get_decompose_based_mask(attr_state_all, classifiers,start_layer, end_layer, device, predictions,max_mask_percent, num_mask_tokens=None,classifier_bound=False):
    predictions = np.array(predictions)
    num_tokens = len(predictions)
    if num_mask_tokens is None:
        num_mask_tokens = int(num_tokens * max_mask_percent) + 1
    if classifier_bound:
        low_prob_mask = (predictions < 0.5) & (np.arange(len(predictions)) != 0)
    else:
        low_prob_mask = (np.arange(len(predictions)) != 0)
    low_prob_indices = np.where(low_prob_mask)[0]
    if len(low_prob_indices) == 0:
        return []
    all_token_logits = []    
    for token_idx in low_prob_indices:
        all_layer_logits = []
        for layer_idx in range(start_layer, end_layer):
            classifier = classifiers[layer_idx]
            w, b = classifier.get_weights_bias()
            attr_state = attr_state_all[layer_idx][token_idx]
            target_dtype = attr_state.dtype
            w = w.to(dtype=target_dtype, device=device)        
            logits = attr_state.to(device) @ w.T.to(device)
            logits = logits + b.to(device)
            all_layer_logits.append(logits)
        avg_logits = torch.mean(torch.stack(all_layer_logits), dim=0)
        all_token_logits.append(avg_logits)
    token_probs = predictions[low_prob_indices]
    weights = 1 - token_probs
    normalized_weights = weights / weights.sum()
    weights_tensor = torch.tensor(normalized_weights, device=device)
    stacked_logits = torch.stack(all_token_logits)
    weights_tensor = weights_tensor.view(-1, 1, 1)
    weighted_avg_logits = (stacked_logits * weights_tensor).sum(dim=0)
    weighted_avg_logits = weighted_avg_logits.squeeze(-1)
    mask_indices = get_mask_indices_from_logits(
        weighted_avg_logits,
        low_prob_indices,
        num_tokens,
        max_mask_percent,
        num_mask_tokens,
        classifier_bound
    )
    return mask_indices

def get_mask_indices_from_logits(weighted_avg_logits, low_prob_indices, num_tokens, max_mask_percent, num_mask_tokens=None, classifier_bound=False):
    if num_mask_tokens is None:
        num_mask_tokens = int(num_tokens * max_mask_percent) + 1
    logits_np = weighted_avg_logits.cpu().numpy()
    low_prob_indices = low_prob_indices[low_prob_indices != 0]
    valid_sorted_indices = np.argsort(logits_np)[:len(low_prob_indices)]
    if classifier_bound:
        actual_mask_tokens = min(num_mask_tokens, len(low_prob_indices))
    else:
        actual_mask_tokens=num_mask_tokens
    selected_indices = valid_sorted_indices[:actual_mask_tokens]
    return selected_indices.tolist()

def get_random_token_indices(predictions, max_mask_percent, num_mask_tokens=None):
    num_tokens = len(predictions)
    if num_mask_tokens is None:
        num_mask_tokens = int(num_tokens * max_mask_percent) + 1
    valid_indices = np.arange(1, num_tokens)
    actual_mask_tokens = min(num_mask_tokens, len(valid_indices))
    random_indices = np.random.choice(
        valid_indices, 
        size=actual_mask_tokens, 
        replace=False
    )
    random_indices.sort()    
    return random_indices.tolist()

def extract_information(prompt_str):
    if "Information:" not in prompt_str:
        return None
    try:
        info_start = prompt_str.index("Information:") + len("Information:")
        info_end = prompt_str.index("\nQuestion:")
        information = prompt_str[info_start:info_end].strip()
        return information
    except ValueError:
        return None

from transformers import AutoTokenizer

def get_masked_prompt(prompt, mask_token_indices, tokenizer):
    info = extract_information(prompt)
    if info is None:
        raise ValueError("Prompt does not contain 'Information:' section.")
    info_tokens = tokenizer.tokenize(info)
    for idx in mask_token_indices:
        if 0 <= idx < len(info_tokens):
            info_tokens[idx] = ""
    masked_info = tokenizer.convert_tokens_to_string(info_tokens)
    info_start = prompt.index("Information:") + len("Information:")
    info_end = prompt.index("\nQuestion:")
    masked_prompt = (
        prompt[:info_start]
        + " " + masked_info.strip()
        + prompt[info_end:]
    )
    return masked_prompt



def mask_token(prompt, AttrStateManager, classifiers, classifier_start_layer, classifier_end_layer, device, max_mask_percent,classifier_bound=False):
    info = extract_information(prompt)
    if info is None:
        raise ValueError("Prompt does not contain 'Information:' section.")
    attr_state_all, states = AttrStateManager.all_layer_to_init_attribute_state(info)
    truthful_prob = get_all_token_truthful_prob(classifiers, states, classifier_start_layer, classifier_end_layer, device)
    mask_self_based = get_untruthful_token_indices(truthful_prob, max_mask_percent,classifier_bound=classifier_bound)
    mask_decompose_based = get_decompose_based_mask(attr_state_all, classifiers, classifier_start_layer, classifier_end_layer, device, truthful_prob, max_mask_percent,classifier_bound=classifier_bound)
    mask_random_based = get_random_token_indices(truthful_prob, max_mask_percent)
    prompt_self_based_mask = get_masked_prompt(prompt, mask_self_based, AttrStateManager.tokenizer)
    prompt_decompose_based_mask = get_masked_prompt(prompt, mask_decompose_based, AttrStateManager.tokenizer)
    prompt_random_based_mask = get_masked_prompt(prompt, mask_random_based, AttrStateManager.tokenizer)
    return prompt_self_based_mask, prompt_decompose_based_mask, prompt_random_based_mask
    
    
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

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
def normalize_model_name(name):
    """Normalize model name to standardized format"""
    name = name.lower().strip()
    name = name.replace("-hf", "")
    name = name.replace("meta-", "")
    return name

def main():
    args = parse_args()
    model_path = args.model_path
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    model_type = normalize_model_name(args.model_name)
    if "llama" in model_type:
        model_type = "llama"
    elif "qwen" in model_type:
        model_type = "qwen"
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
    classifier_path = f'../classifier-training/{args.type}/classifiers/{args.model_name}_classifiers.pt'
    try:
        classifiers = torch.load(classifier_path)
        print(f"Successfully loaded classifiers from {classifier_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load classifiers: {e}")

    num_layer = model.config.num_hidden_layers
    if args.classifier_end_layer < 0:
        args.classifier_end_layer = num_layer + args.classifier_end_layer
    AttrStateManager = decomposed_state_manager(model, tokenizer)

    # === Processing Prompts ===
    prompt_all = ['prompt_init', 'prompt_correct', 'prompt_wrong']
    results = []
    dataset_name = args.dataset.split('.')[0]
    for data in tqdm(data_all, desc="Processing data"):
        if dataset_name == "counterfact_data":
            result = {'case_id': data['case_id'], 'subject': data['subject'],
                    'target': data['target'], 'target_new': data['target_new']}
        elif dataset_name == "truthfulqa_data":
            result = {'case_id': data['case_id'], 'correct_option': data['correct_option'],}
        for prompt_type in prompt_all:
            prompt = data[prompt_type]
            result[prompt_type] = prompt
            result[f"{prompt_type}_answer"] = generate_answer(model, tokenizer, prompt, device)

            if prompt_type == "prompt_wrong":
                mask_self, mask_decompose, mask_random = mask_token(
                    prompt, AttrStateManager, classifiers,
                    args.classifier_start_layer, args.classifier_end_layer,
                    device, args.max_mask_percent, classifier_bound=args.classifier_bound
                )
                result[f"{prompt_type}_self_based_mask"] = mask_self
                result[f"{prompt_type}_self_based_mask_answer"] = generate_answer(model, tokenizer, mask_self, device)

                result[f"{prompt_type}_DePass_based_mask"] = mask_decompose
                result[f"{prompt_type}_DePass_based_mask_answer"] = generate_answer(model, tokenizer, mask_decompose, device)

                result[f"{prompt_type}_random_based_mask"] = mask_random
                result[f"{prompt_type}_random_based_mask_answer"] = generate_answer(model, tokenizer, mask_random, device)

        results.append(result)

    # === Save Results ===
    dataset_name = args.dataset.split('.')[0]
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.model_name}_{dataset_name}_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")

    
    
if __name__ == "__main__":
    main()