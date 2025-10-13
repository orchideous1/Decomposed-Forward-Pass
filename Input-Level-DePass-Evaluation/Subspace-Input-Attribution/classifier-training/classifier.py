from sklearn.linear_model import LogisticRegression
import json
import torch.nn as nn
import torch
from tqdm import tqdm
import os
import numpy as np

class StateClassifier:
    def __init__(self, lr: float=0.01, max_iter: int=1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear = LogisticRegression(solver="saga", max_iter=max_iter)
        self.losses = []

    def train(self, pos_tensor: torch.tensor, neg_tensor: torch.tensor, n_epoch: int=100, batch_size: int=64) -> list[float]:
        pos_tensor = pos_tensor.to(self.device, dtype=torch.float32)
        neg_tensor = neg_tensor.to(self.device, dtype=torch.float32)
        X = torch.vstack([pos_tensor, neg_tensor]).to(self.device, dtype=torch.float32)
        y = torch.cat((torch.ones(pos_tensor.size(0)), torch.zeros(neg_tensor.size(0)))).to(self.device, dtype=torch.float32)
        perm = torch.randperm(X.size(0))
        X = X[perm]
        y = y[perm]
        self.linear.fit(X.cpu().numpy(), y.cpu().numpy())
        del X, y

        return []
    
    def train_orthogonal(self, pos_tensor: torch.tensor, neg_tensor: torch.tensor, orthogonal_parameter, n_epoch: int=100, batch_size: int=64) -> list[float]:
        pos_tensor = pos_tensor.to(self.device, dtype=torch.float32)
        neg_tensor = neg_tensor.to(self.device, dtype=torch.float32)
        X = torch.vstack([pos_tensor, neg_tensor]).to(self.device, dtype=torch.float32)
        y = torch.cat((torch.ones(pos_tensor.size(0)), torch.zeros(neg_tensor.size(0)))).to(self.device, dtype=torch.float32)
        perm = torch.randperm(X.size(0))
        X = X[perm]
        y = y[perm]
        self.linear.fit(X.cpu().numpy(), y.cpu().numpy())
        del X, y
        return []
    
    def predict(self, tensor: torch.tensor) -> torch.tensor:
        return torch.tensor(self.linear.predict(tensor.cpu().numpy()))

    def predict_proba(self, tensor: torch.tensor) -> float:
        tensor = tensor.to(self.device, dtype=torch.float32)
        w, b = self.get_weights_bias()
        proba = torch.sigmoid(tensor @ w.T + b)
        return proba.item()
    
    def evaluate_testacc(self, pos_tensor: torch.tensor, neg_tensor: torch.tensor) -> float:
        pos_tensor = pos_tensor.to(self.device, dtype=torch.float32)
        neg_tensor = neg_tensor.to(self.device, dtype=torch.float32)
        test_data = torch.vstack([pos_tensor, neg_tensor]).to(self.device,dtype=torch.float32)
        predictions = self.predict(test_data)
        true_labels = torch.cat((torch.ones(pos_tensor.size(0)), torch.zeros(neg_tensor.size(0))))
        correct_count = torch.sum((predictions > 0.5) == true_labels).item()
        return correct_count / len(true_labels)   
    
    def get_weights_bias(self) -> tuple[torch.tensor, torch.tensor]:
        return torch.tensor(self.linear.coef_).to(self.device), torch.tensor(self.linear.intercept_).to(self.device)


class ClassifierManager:
    def __init__(self, model, tokenizer, data_path):
        self.model = model
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.classifiers = []
        self.positive_activations=[]
        self.negative_activations=[]
        self.testacc=[]
        
    def train_classifiers(
        self,
        lr: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
    ):  
        
        print("Loading data...")
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        print("Getting activations...")
        
        for data_idx in tqdm(range(len(data)), desc="Prompts"):
            if data[data_idx]['label'] == 1:
                self.positive_activations.append(get_last_token_state_list(data[data_idx]['text'], self.model, self.tokenizer, -2).cpu())
                self.positive_activations.append(get_last_token_state_list(data[data_idx]['text'], self.model, self.tokenizer, 2).cpu())
                self.positive_activations.append(get_last_token_state_list(data[data_idx]['text'], self.model, self.tokenizer, 3).cpu())
            else:
                self.negative_activations.append(get_last_token_state_list(data[data_idx]['text'], self.model, self.tokenizer, -2).cpu())
        self.positive_activations = torch.stack(self.positive_activations, dim=1)
        self.negative_activations = torch.stack(self.negative_activations, dim=1)
        print("Training classifiers...")
        for layer_idx in tqdm(range(self.model.config.num_hidden_layers), desc="Layers"):
            layer_positive_activations = self.positive_activations[layer_idx]
            layer_negative_activations = self.negative_activations[layer_idx]
            classifier = StateClassifier(lr)
            classifier.train(
                pos_tensor=layer_positive_activations,
                neg_tensor=layer_negative_activations,
                n_epoch=n_epochs,
                batch_size=batch_size,
            )
            self.classifiers.append(classifier)
            del layer_positive_activations
            del layer_negative_activations
        self.negative_activations = []
        self.positive_activations = []
    
    def evaluate_classifiers(
            self
        ):  
        print("Loading data...")
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        print("Getting activations...")
        print("Getting activations...")
        for data_idx in tqdm(range(len(data)), desc="Prompts"):
            if data[data_idx]['label'] == 1:
                self.positive_activations.append(get_last_token_state_list(data[data_idx]['text'], self.model, self.tokenizer, -2).cpu())
                self.positive_activations.append(get_last_token_state_list(data[data_idx]['text'], self.model, self.tokenizer, 3).cpu())
                
            else:
                self.negative_activations.append(get_last_token_state_list(data[data_idx]['text'], self.model, self.tokenizer, -2).cpu())
        
        self.positive_activations = torch.stack(self.positive_activations, dim=1) 
        self.negative_activations = torch.stack(self.negative_activations, dim=1)
        
        print("Testing classifiers...")
        for layer_idx in tqdm(range(self.model.config.num_hidden_layers), desc="Layers"):
            layer_positive_activations = self.positive_activations[layer_idx]
            layer_negative_activations = self.negative_activations[layer_idx]
            layer_test_acc = self.classifiers[layer_idx].evaluate_testacc(
                pos_tensor=layer_positive_activations,
                neg_tensor=layer_negative_activations
            )
            self.testacc.append(layer_test_acc)
            del layer_positive_activations
            del layer_negative_activations
        
        self.negative_activations = []
        self.positive_activations = []
        
    def get_prob(self,prompt):
        activations=self.activation_manager.get_activations(prompt)
        all_probs=[]
        for i in range(len(self.classifiers)):
            layer_probs=[]
            for j in range(len(self.classifiers[i])):
                head_activations = activations[i][j]
                prob=self.classifiers[i][j].predict_proba(head_activations)
                layer_probs.append(prob)
            all_probs.append(layer_probs)
        return all_probs


    def save_classifiers(self, output_path):
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