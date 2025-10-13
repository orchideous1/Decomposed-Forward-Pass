from sklearn.linear_model import LogisticRegression
import json
import torch.nn as nn
import torch
from tqdm import tqdm
import os
import numpy as np


from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn

class StateClassifier:
    def __init__(self, lr: float = 0.01, max_iter: int = 1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear = LogisticRegression(
            solver="saga", max_iter=max_iter, multi_class="multinomial"
        )

    def train(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor) -> list[float]:
        X_tensor = X_tensor.to(self.device, dtype=torch.float32)
        y_tensor = y_tensor.to(self.device, dtype=torch.long)
        perm = torch.randperm(X_tensor.size(0))
        X_tensor = X_tensor[perm]
        y_tensor = y_tensor[perm]

        self.linear.fit(X_tensor.cpu().numpy(), y_tensor.cpu().numpy())
        return []

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.linear.predict(tensor.cpu().numpy()))

    def predict_proba(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.device, dtype=torch.float32)
        probs = self.linear.predict_proba(tensor.cpu().numpy())
        return torch.tensor(probs)

    def evaluate_testacc(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor) -> float:
        X_tensor = X_tensor.to(self.device, dtype=torch.float32)
        y_tensor = y_tensor.to(self.device, dtype=torch.long)
        predictions = self.predict(X_tensor)
        correct_count = (predictions == y_tensor.cpu()).sum().item()
        return correct_count / len(y_tensor)

    def get_weights_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.linear.coef_).to(self.device), torch.tensor(self.linear.intercept_).to(self.device)



class ClassifierManager:
    def __init__(self, model, tokenizer, data_path):
        self.model = model
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.classifiers = []
        self.activations_by_class = {}
        self.labels = []

    def train_classifiers(self):
        print("Loading data...")
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        print("Getting activations...")
        for sample in tqdm(data, desc="Prompts"):
            label = sample["label"]
            hidden_state = get_last_token_state_list(sample["text"], self.model, self.tokenizer).cpu()
            if label not in self.activations_by_class:
                self.activations_by_class[label] = []
            self.activations_by_class[label].append(hidden_state)

        num_layers = self.model.config.num_hidden_layers
        for layer_idx in tqdm(range(num_layers), desc="Layers"):
            X_all = []
            y_all = []

            for label, tensor_list in self.activations_by_class.items():
                for hidden_state in tensor_list:
                    X_all.append(hidden_state[layer_idx])
                    y_all.append(label)

            X_tensor = torch.stack(X_all)
            y_tensor = torch.tensor(y_all)

            classifier = StateClassifier()
            classifier.train(X_tensor, y_tensor)
            self.classifiers.append(classifier)

        self.activations_by_class = {}

    def evaluate_classifiers(self):
        acc_list=[]
        print("Loading data...")
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        print("Getting activations...")
        activations_by_class = {}
        for sample in tqdm(data, desc="Prompts"):
            label = sample["label"]
            hidden_state = get_last_token_state_list(sample["text"], self.model, self.tokenizer, -2).cpu()
            hidden_state = get_last_token_state_list(sample["text"], self.model, self.tokenizer, -1).cpu()
            if label not in activations_by_class:
                activations_by_class[label] = []
            activations_by_class[label].append(hidden_state)

        num_layers = self.model.config.num_hidden_layers
        for layer_idx in tqdm(range(num_layers), desc="Layers"):
            X_all = []
            y_all = []
            for label, tensor_list in activations_by_class.items():
                for hidden_state in tensor_list:
                    X_all.append(hidden_state[layer_idx])
                    y_all.append(label)

            X_tensor = torch.stack(X_all)
            y_tensor = torch.tensor(y_all)

            acc = self.classifiers[layer_idx].evaluate_testacc(X_tensor, y_tensor)
            acc_list.append(acc)
            print(f"Layer {layer_idx} accuracy: {acc:.4f}")
        return acc_list

    def save_classifiers(self, output_path):
        # Clear references to large objects
        self.model = None
        self.tokenizer = None
        self.positive_activations = None
        self.negative_activations = None
        self.testacc = None
        self.data_path = None
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        torch.save(self.classifiers, output_path)
        

def get_last_token_state_list(prompt, model, tokenizer, token_idx=-1):
    device = model.device
    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**input, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    all_states = torch.stack(hidden_states)
    last_token_states = all_states[:, 0, token_idx, :]
    return last_token_states
