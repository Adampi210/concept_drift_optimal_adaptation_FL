import os
import random
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np

# Import your custom modules from fl_toolkit
# Note: Ensure fl_toolkit is installed and accessible
from fl_toolkit import *


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define TinyBert Model for Sentiment Analysis ---
class TinyBertForSentiment(BaseModelArchitecture):
    def __init__(self, num_classes=3): # POS, NEU, NEG
        super(TinyBertForSentiment, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            'prajjwal1/bert-tiny',
            num_labels=num_classes,
            local_files_only=True
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

# --- Simplified Dataset Class (No changes needed) ---
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Helper function to load data (No changes needed) ---
def load_data_from_domain(root_dir, domain, split, tokenizer, max_len):
    sentiment_to_label = {'POS': 2, 'NEU': 1, 'NEG': 0}
    data_list = []
    file_path = os.path.join(root_dir, domain, f'{split}.json')
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping.")
        return None
    print(f"Loading {split} data from: {file_path}")
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
    return MEMDABSADataset(data_list, tokenizer=tokenizer, max_length=max_len)


# Training and Evaluation Function
def train_and_evaluate_model(model_class, model_path, seed, domain, num_epochs, batch_size, learning_rate, optimizer_choice, root_dir, max_len, tokenizer):
    set_seed(seed)
    train_dataset = load_data_from_domain(root_dir, domain, 'Train', tokenizer, max_len)
    val_dataset = load_data_from_domain(root_dir, domain, 'Dev', tokenizer, max_len)
    test_dataset = load_data_from_domain(root_dir, domain, 'Test', tokenizer, max_len)

    if train_dataset is None or val_dataset is None:
        print(f"Missing Train or Dev data for domain {domain}. Skipping training.")
        return None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    MEMD_ABSA_Model_Net = BaseNeuralNetwork(model_class, device=device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, MEMD_ABSA_Model_Net.model.parameters()), lr=learning_rate)
    
    patience = 2
    patience_counter = 0
    best_val_accuracy = 0.0
    best_model_state = None

    print(f"Starting training on domain: {domain}...")
    for epoch in range(num_epochs):
        # The fl_toolkit's update_steps calls the loss function we provide.
        avg_loss = MEMD_ABSA_Model_Net.update_steps(train_loader, optimizer, loss_fn, 60, verbose=True)
        
        val_accuracy = MEMD_ABSA_Model_Net.evaluate(val_loader, accuracy_fn)
        # We need to create a new loss_fn for evaluation that does the same as the training one.
        val_loss = MEMD_ABSA_Model_Net.evaluate(val_loader, loss_fn)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            best_model_state = MEMD_ABSA_Model_Net.get_params()
            print(f"New best model found at epoch {epoch + 1} with Val Accuracy: {best_val_accuracy:.4f}")

    if best_model_state:
        MEMD_ABSA_Model_Net.set_params(best_model_state)
        test_accuracy = MEMD_ABSA_Model_Net.evaluate(test_loader, accuracy_fn)
        print(f"\nBest model achieved Test Accuracy: {test_accuracy:.4f}")
        MEMD_ABSA_Model_Net.save_model(model_path)
        print(f"Best model for domain '{domain}' saved to {model_path}")
        return model_path
    else:
        print("Training did not result in a best model. No model saved.")
        return None

# --- Corrected helper functions ---
def accuracy_fn(outputs, labels):
    """Calculates accuracy from model output (logits) and labels."""
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean()

def loss_fn(outputs, labels):
    """
    Calculates the cross-entropy loss from the model's output logits.
    This is called by the toolkit's update_steps and evaluate methods.
    """
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    loss_fct = nn.CrossEntropyLoss()
    return loss_fct(logits, labels)

# Main Function
def main():
    parser = argparse.ArgumentParser(description="Pretrain TinyBERT on MEMD-ABSA Dataset Single Domains")
    parser.add_argument('--root_dir', type=str, default='/scratch/gautschi/apiasecz/data/MEMD-ABSA/MEMD_ABSA_Dataset/', help='Root directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model', type=str, default='TinyBertForSentiment', help='Model to train')
    parser.add_argument('--domains', type=str, nargs='+', default=['Books', 'Clothing', 'Hotel', 'Laptop', 'Restaurant'], help='Domains to train on')
    parser.add_argument('--num_epochs', type=int, default=10, help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
    parser.add_argument('--model_save_dir', type=str, default='/scratch/gautschi/apiasecz/models/concept_drift_models/', help='Model save directory')
    parser.add_argument('--max_len', type=int, default=128, help='Max sequence length')
    args = parser.parse_args()

    os.makedirs(args.model_save_dir, exist_ok=True)

    print("Loading TinyBert tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny', local_files_only=True)

    trained_model_paths = {}
    for domain in args.domains:
        print(f"\n===== Processing domain: {domain} =====")
        model_class = TinyBertForSentiment
        model_filename = f"{args.model}_{domain}_seed_{args.seed}.pth"
        model_save_path = os.path.join(args.model_save_dir, model_filename)

        model_path = train_and_evaluate_model(
            model_class=model_class, model_path=model_save_path, seed=args.seed, domain=domain,
            num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
            optimizer_choice=args.optimizer, root_dir=args.root_dir, max_len=args.max_len, tokenizer=tokenizer
        )
        if model_path:
            trained_model_paths[domain] = model_path

    print("\n===== Cross-Domain Evaluation =====")
    cross_accuracies = {source: {} for source in trained_model_paths}
    for source_domain, model_path in trained_model_paths.items():
        print(f"\n--- Evaluating model trained on: '{source_domain}' ---")
        model_net = BaseNeuralNetwork(TinyBertForSentiment, device=device)
        model_net.load_model(model_path)
        for target_domain in args.domains:
            test_dataset = load_data_from_domain(args.root_dir, target_domain, 'Test', tokenizer, args.max_len)
            if test_dataset is None: continue
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            test_accuracy = model_net.evaluate(test_loader, accuracy_fn)
            cross_accuracies[source_domain][target_domain] = test_accuracy
            print(f"  - Accuracy on '{target_domain}' test set: {test_accuracy*100:.2f}%")

    print("\nCross-Domain Accuracy Table (Rows: Pretrained on, Columns: Tested on):")
    header = f"{'Pretrained \\ Tested':<18} | " + " | ".join(args.domains)
    print(header)
    print("-" * len(header))
    for source in trained_model_paths:
        row = f"{source:<18} | " + " | ".join(f"{cross_accuracies[source].get(target, 'N/A'):.4f}" for target in args.domains)
        print(row)

    print("\nAll Trainings Completed")

if __name__ == "__main__":
    main()