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

from baseline_attribution_methods.attention import AttentionImportanceScoreEvaluator
from baseline_attribution_methods.grad import GradientImportanceScoreEvaluator



import argparse
import os
import json
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# python get_importance_score.py \
#     --model_name llama-2-7b-chat-hf \
#     --model_path /root/models/transformers/llama-2/llama-2-7b-chat-hf \
#     --dataset known_1000.json \
#     --device cuda:0 \
#     --output_dir ./results \
#     --dtype bfloat16

def parse_args():
    parser = argparse.ArgumentParser(description='Attribution evaluation script')
    parser.add_argument('--model_name', type=str, default='llama-2-7b-chat-hf',
                        help='Name of the model')
    parser.add_argument('--model_path', type=str, 
                        default='/root/models/transformers/llama-2/llama-2-7b-chat-hf',
                        help='Path to the model')
    parser.add_argument('--dataset', type=str, default='known_1000.json',
                        help='Dataset filename')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--output_dir', type=str, default='./importance_scores',
                        help='Directory to save results')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float16', 'float32', 'bfloat16'],
                        help='Floating point precision type')
    return parser.parse_args()

def normalize_model_name(name):
    """Normalize model name to standardized format"""
    name = name.lower().strip()
    name = name.replace("-hf", "")
    name = name.replace("meta-", "")
    return name

def main():
    args = parse_args()
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    torch_dtype = dtype_map[args.dtype]
    model_name = args.model_name
    model_path = args.model_path
    dataset = args.dataset
    model_type = normalize_model_name(args.model_name)
    if "llama" in model_type:
        model_type = "llama"
    elif "qwen" in model_type:
        model_type = "qwen"
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
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_path = os.path.join("./data", dataset)
    try:
        with open(data_path, 'r') as f:
            data_all = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {data_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {data_path}")
    
    if dataset=="known_1000.json" or dataset=="known.json":
        processed_data = []
        for i, data in enumerate(data_all):
            dict={}
            dict['id'] = data['known_id']
            dict['prompt'] = data['prompt']
            dict['target_token']= data['attribute']
            dict['relation_id']= data['relation_id']
            processed_data.append(dict)
    else:
        processed_data = data_all
    attn_type=[
        "all",
        "last",
        "rollout"
    ]

    grad_type=[
        "integrated_gradients",
        "signed",
        "norm"
    ]

    importance_score_evaluator_list = {}
    for i in range(len(attn_type)):
        importance_score_evaluator_list[attn_type[i]] = AttentionImportanceScoreEvaluator(
                model=model,
                tokenizer=tokenizer,
                attn_type=attn_type[i]
            )
    for i in range(len(grad_type)):
        importance_score_evaluator_list[grad_type[i]] = GradientImportanceScoreEvaluator(
                model=model,
                tokenizer=tokenizer,
                grad_type=grad_type[i]
            )
        
    DecomposedStateManager = decomposed_state_manager(model, tokenizer)

    results=[]
    for i, data in tqdm(enumerate(processed_data), 
                        total=len(processed_data), 
                        desc="Processing samples", 
                        ncols=100):
        prompt = data['prompt']
        target_token = data['target_token']
        input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
        target_token_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
        target_id = torch.tensor([target_token_id], device=device) 
        target_id = target_id.unsqueeze(0)
        
        attribute_state, states= DecomposedStateManager.get_last_layer_decomposed_state(prompt)
        data["DePass"] = DecomposedStateManager.compute_attribute_score(attribute_state, -1, target_token_id).unsqueeze(0).cpu().detach().numpy().tolist()
        results.append(data)
        
        for j in range(len(attn_type)):
            score = importance_score_evaluator_list[attn_type[j]].evaluate(input_ids, target_id)
            score = score.float().cpu().detach().numpy().tolist()
            data[attn_type[j]] = score
        
        for j in range(len(grad_type)):
            score =importance_score_evaluator_list[grad_type[j]].evaluate(input_ids, target_id)
            score= score.float().cpu().detach().numpy().tolist()
            data[grad_type[j]] = score
        

    file_name = f"{model_name}_{dataset}"
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, file_name)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()

