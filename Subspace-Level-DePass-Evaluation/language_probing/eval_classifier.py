import argparse
import torch
import os
from transformers import AutoModel, AutoTokenizer
from classifier import ClassifierManager  # 确保你已定义好 classifier.py 中的逻辑

# python train_classifier.py \
#     --model_name "llama-2-7b-chat-hf" \
#     --model_path "/root/models/transformers/llama-2/llama-2-7b-chat-hf" \
#     --probing_type "truthful" \
#     --device "auto" 
def parse_args():
    parser = argparse.ArgumentParser(description='Train classifiers for probing subspace')
    parser.add_argument('--model_name', type=str, default='llama-3.1-8b-instruct',
                        help='Name of the model')
    parser.add_argument('--model_path', type=str, default="/root/models/llama_3_1_8b_instruct/",
                        help='Local path to the pretrained model (if any)')
    parser.add_argument('--dataset_path', type=str, default='./data/train_dataset.json',
                        help='Path to processed training dataset (JSON)')
    parser.add_argument('--input_dir', type=str, default='./output_classifiers',
                        help='Directory to save trained classifier')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: "auto", "cuda", or "cpu"')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'float32', 'bfloat16'],
                        help='Floating point precision')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    torch_dtype = dtype_map[args.dtype]
    model_path_or_name = args.model_path if args.model_path else args.model_name
    model = AutoModel.from_pretrained(model_path_or_name, torch_dtype=torch_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    model.eval()
    input_path = os.path.join(args.input_dir, f'{args.model_name}_classifier.pt')
    try:
        classifiers = torch.load(input_path)
        print(f"Successfully loaded classifiers from {input_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load classifiers from {input_path}: {str(e)}")
    manager = ClassifierManager(model, tokenizer, args.dataset_path)
    manager.classifiers = classifiers
    manager.evaluate_classifiers()
    
if __name__ == "__main__":
    main()