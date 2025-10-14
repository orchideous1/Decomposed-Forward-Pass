from classifier import ClassifierManager
import argparse
import torch
from tqdm import tqdm
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description='Train classifiers for probing subspace')
    parser.add_argument('--model_name', type=str, default='llama-2-7b-chat-hf',
                        help='Name of the model')
    parser.add_argument('--model_path', type=str, 
                        default='/root/models/transformers/llama-2/llama-2-7b-chat-hf',
                        help='Path to the model')
    parser.add_argument('--probing_type', type=str, default='truthful',
                        help='subspace probing type (e.g., truthful, safety)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'float32', 'bfloat16'],
                        help='Floating point precision type')
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = args.model_name
    model_path = args.model_path
    
    # Convert dtype string to torch dtype
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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if args.probing_type == "truthful":
        data_path = './truthful/train_data.json'
    elif args.probing_type == "safety":
        data_path = './safety/train_data.json'
    else:
        raise ValueError("Invalid probing type. Choose either 'truthful' or 'safety'.")
    
    classifier_manager = ClassifierManager(model, tokenizer, data_path)
    classifier_manager.train_classifiers()
    # from IPython import embed; embed()
    # assert False
    if args.output_dir is None:
        if args.probing_type not in ["truthful", "safety"]:
            raise ValueError("Invalid probing type. Choose either 'truthful' or 'safety'.")
        output_dir = f'./{args.probing_type}/classifiers/'
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{model_name}_classifiers.pt')
    classifier_manager.save_classifiers(output_path)
    print(f"Classifiers saved to {output_path}")
    print("Training complete.")    
    


if __name__ == "__main__":
    main()