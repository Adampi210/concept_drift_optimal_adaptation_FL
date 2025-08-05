import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np
import random

# --- Prerequisite: fl_toolkit classes ---
# In a real scenario, these would be imported from your fl_toolkit library.
# For this script to be self-contained, we define minimal versions here.

class BaseModelArchitecture(nn.Module):
    """A base class for model architectures, inheriting from nn.Module."""
    def __init__(self):
        super(BaseModelArchitecture, self).__init__()

    def forward(self, *args, **kwargs):
        """
        Defines the forward pass of the model.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

class BaseNeuralNetwork:
    """A wrapper class to manage a PyTorch model, its device, and basic operations."""
    def __init__(self, model_class, device='cpu', *args, **kwargs):
        self.device = torch.device(device)
        self.model = model_class(*args, **kwargs).to(self.device)

    def evaluate(self, data_loader, metric_fn):
        """Evaluates the model on the given data loader using a specified metric."""
        self.model.eval()
        total_metric = 0.0
        with torch.no_grad():
            for inputs, labels in data_loader:
                # Move all tensor data in the inputs dictionary to the correct device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)

                outputs = self.model(**inputs, labels=labels)
                metric_value = metric_fn(outputs, labels)
                total_metric += metric_value.item()
        return total_metric / len(data_loader)

    def load_model(self, path):
        """Loads model weights from a file."""
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Model loaded successfully from {path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {path}")
            raise
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            raise

# --- Define TinyBert Model for Sentiment Analysis ---
# This must match the model architecture used for training.
class TinyBertForSentiment(BaseModelArchitecture):
    def __init__(self, num_classes=3): # POS, NEU, NEG
        super(TinyBertForSentiment, self).__init__()
        # It's good practice to handle the case where the model might not be available locally.
        try:
            self.bert = BertForSequenceClassification.from_pretrained(
                'prajjwal1/bert-tiny',
                num_labels=num_classes,
                local_files_only=True # Assumes model is cached locally
            )
        except OSError:
            print("Model 'prajjwal1/bert-tiny' not found locally. Attempting to download...")
            self.bert = BertForSequenceClassification.from_pretrained(
                'prajjwal1/bert-tiny',
                num_labels=num_classes,
                local_files_only=False
            )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

# --- Simplified Dataset Class (No changes from original) ---
class MEMDABSADataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=128):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label, domain = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length,
            return_token_type_ids=False, padding='max_length',
            return_attention_mask=True, return_tensors='pt', truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }, torch.tensor(label, dtype=torch.long)

# --- Helper function to load data (No changes from original) ---
def load_data_from_domain(root_dir, domain, split, tokenizer, max_len):
    sentiment_to_label = {'POS': 2, 'NEU': 1, 'NEG': 0}
    data_list = []
    file_path = os.path.join(root_dir, domain, f'{split}.json')
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping.")
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            content_list = json.load(f)
            for content in content_list:
                text = content.get('raw_words')
                quadruples = content.get('quadruples')
                if text and quadruples and len(quadruples) > 0:
                    sentiment_str = quadruples[0]['sentiment']
                    if sentiment_str in sentiment_to_label:
                        label = sentiment_to_label[sentiment_str]
                        data_list.append((text, label, domain))
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}")
            return None
    if not data_list:
        return None
    return MEMDABSADataset(data_list, tokenizer=tokenizer, max_length=max_len)

# --- Metric function ---
def accuracy_fn(outputs, labels):
    """Calculates accuracy from model output (logits) and labels."""
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean()

# --- Main Evaluation Function ---
def main():
    parser = argparse.ArgumentParser(description="Cross-Domain Evaluation of TinyBERT on MEMD-ABSA Dataset")
    parser.add_argument('--root_dir', type=str, default='/scratch/gautschi/apiasecz/data/MEMD-ABSA/MEMD_ABSA_Dataset/', help='Root directory of the dataset')
    parser.add_argument('--model_load_dir', type=str, default='/scratch/gautschi/apiasecz/models/concept_drift_models/', help='Directory where pre-trained models are saved')
    parser.add_argument('--model_name', type=str, default='TinyBertForSentiment', help='Name of the model architecture')
    parser.add_argument('--domains', type=str, nargs='+', default=['Books', 'Clothing', 'Hotel', 'Laptop', 'Restaurant'], help='List of domains to evaluate on')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='List of random seeds used for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--max_len', type=int, default=128, help='Max sequence length for tokenizer')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading TinyBert tokenizer...")
    try:
        tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny', local_files_only=True)
    except OSError:
        print("Tokenizer not found locally. Downloading...")
        tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny', local_files_only=False)


    # --- Data structure to hold all results ---
    # results[source_domain][target_domain] = [acc_seed_1, acc_seed_2, ...]
    cross_accuracies = {source: {target: [] for target in args.domains} for source in args.domains}

    # --- Evaluation Loop ---
    for seed in args.seeds:
        print(f"\n{'='*20} PROCESSING SEED: {seed} {'='*20}")
        for source_domain in args.domains:
            print(f"\n--- Loading model trained on '{source_domain}' (seed {seed}) ---")
            
            model_filename = f"{args.model_name}_{source_domain}_seed_{seed}.pth"
            model_path = os.path.join(args.model_load_dir, model_filename)

            if not os.path.exists(model_path):
                print(f"Warning: Model file not found at {model_path}. Skipping.")
                continue

            # Initialize and load the pre-trained model
            model_net = BaseNeuralNetwork(TinyBertForSentiment, device=device)
            try:
                model_net.load_model(model_path)
            except Exception:
                continue # Skip if model loading fails

            # Evaluate on all target domains
            for target_domain in args.domains:
                print(f"  -> Evaluating on '{target_domain}' test set...")
                test_dataset = load_data_from_domain(args.root_dir, target_domain, 'Test', tokenizer, args.max_len)
                
                if test_dataset is None:
                    print(f"  -> No test data for '{target_domain}'. Skipping.")
                    continue
                
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                
                # Calculate accuracy
                test_accuracy = model_net.evaluate(test_loader, accuracy_fn)
                cross_accuracies[source_domain][target_domain].append(test_accuracy)
                print(f"  -> Accuracy on '{target_domain}': {test_accuracy*100:.2f}%")

    # --- Calculate Average and Standard Deviation ---
    avg_accuracies = {source: {} for source in args.domains}
    std_devs = {source: {} for source in args.domains}

    for source_domain in args.domains:
        for target_domain in args.domains:
            acc_list = cross_accuracies[source_domain][target_domain]
            if acc_list:
                avg_accuracies[source_domain][target_domain] = np.mean(acc_list)
                std_devs[source_domain][target_domain] = np.std(acc_list)
            else:
                avg_accuracies[source_domain][target_domain] = 'N/A'
                std_devs[source_domain][target_domain] = 'N/A'

    # --- Print Results Tables ---
    print("\n" + "="*70)
    print("Cross-Domain Evaluation Results")
    print("="*70)

    # --- Average Accuracy Table ---
    print("\nAverage Accuracy Table (Rows: Pretrained on, Columns: Tested on)")
    header = f"{'Pretrained \\ Tested':<18} | " + " | ".join([f"{d:<10}" for d in args.domains])
    print(header)
    print("-" * len(header))
    for source in args.domains:
        row_data = []
        for target in args.domains:
            val = avg_accuracies[source].get(target)
            row_data.append(f"{val:.4f}" if isinstance(val, float) else f"{val: <10}")
        row_str = " | ".join(row_data)
        print(f"{source:<18} | {row_str}")

    # --- Standard Deviation Table ---
    print("\n\nStandard Deviation of Accuracy Table")
    header = f"{'Pretrained \\ Tested':<18} | " + " | ".join([f"{d:<10}" for d in args.domains])
    print(header)
    print("-" * len(header))
    for source in args.domains:
        row_data = []
        for target in args.domains:
            val = std_devs[source].get(target)
            row_data.append(f"{val:.4f}" if isinstance(val, float) else f"{val: <10}")
        row_str = " | ".join(row_data)
        print(f"{source:<18} | {row_str}")

    print("\n\nEvaluation complete.")

if __name__ == "__main__":
    main()
