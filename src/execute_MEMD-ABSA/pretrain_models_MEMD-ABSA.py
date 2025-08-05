import os
import random
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import numpy as np

# Import your custom modules from fl_toolkit
from fl_toolkit import BaseModelArchitecture, BaseNeuralNetwork

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Model and tokenizer cached successfully.")
# --- Define DistilBert Model for Sentiment Analysis ---
class DistilBertForSentiment(BaseModelArchitecture):
    def __init__(self, num_classes=3): # POS, NEU, NEG
        super(DistilBertForSentiment, self).__init__()
        self.distilbert = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', 
            num_labels=num_classes, 
            local_files_only=True
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits

# --- Simplified Dataset Class for this script ---
class MEMDABSADataset(Dataset):
    """A simplified PyTorch Dataset for MEMD-ABSA data with single labels."""
    def __init__(self, data_list, tokenizer, max_length=128):
        self.data = data_list # List of (text, label, domain)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label, domain = self.data[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        # The BaseNeuralNetwork expects (inputs, targets)
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

# --- Helper function to load data from a specific split (Train/Dev/Test) ---
def load_data_from_domain(root_dir, domain, split, tokenizer, max_len):
    """
    Loads and parses a specific split (Train.json, Dev.json, or Test.json) 
    for a domain, taking only the first sentiment label per sentence.
    """
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
    
    # Load Train, Validation (Dev), and Test sets
    train_dataset = load_data_from_domain(root_dir, domain, 'Train', tokenizer, max_len)
    val_dataset = load_data_from_domain(root_dir, domain, 'Dev', tokenizer, max_len)
    test_dataset = load_data_from_domain(root_dir, domain, 'Test', tokenizer, max_len)
    
    if train_dataset is None or val_dataset is None or len(train_dataset) == 0 or len(val_dataset) == 0:
        print(f"Missing Train or Dev data for domain {domain}. Skipping training.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    MEMD_ABSA_Model_Net = BaseNeuralNetwork(model_class, device=device)
    
    criterion = nn.CrossEntropyLoss()
    if optimizer_choice.lower() == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, MEMD_ABSA_Model_Net.model.parameters()), lr=learning_rate)
    else: # Default to SGD
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, MEMD_ABSA_Model_Net.model.parameters()), lr=learning_rate, momentum=0.9)

    # Early stopping parameters
    patience = 2
    patience_counter = 0
    best_val_accuracy = 0.0
    best_model_state = None
    val_accuracy = MEMD_ABSA_Model_Net.evaluate(val_loader, accuracy_fn)
    val_loss = MEMD_ABSA_Model_Net.evaluate(val_loader, loss_fn)
    print(f"Initial Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")
    print(f"Starting training on domain: {domain}...")
    for epoch in range(num_epochs):
        avg_loss = MEMD_ABSA_Model_Net.update_steps(train_loader, optimizer, criterion, 60, verbose=True)
        
        # Validation
        val_accuracy = MEMD_ABSA_Model_Net.evaluate(val_loader, accuracy_fn)
        val_loss = MEMD_ABSA_Model_Net.evaluate(val_loader, loss_fn)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            best_model_state = MEMD_ABSA_Model_Net.get_params()
            print(f"New best model found at epoch {epoch + 1} with Val Accuracy: {best_val_accuracy:.4f}")

        
    # Load the best model and evaluate on the test set
    if best_model_state:
        MEMD_ABSA_Model_Net.set_params(best_model_state)
        test_accuracy = MEMD_ABSA_Model_Net.evaluate(test_loader, accuracy_fn)
        print(f"\nBest model from epoch {epoch + 1 - patience} achieved Test Accuracy: {test_accuracy:.4f}")
        MEMD_ABSA_Model_Net.save_model(model_path)

        print(f"Best model for domain '{domain}' saved to {model_path}")
    else:
        print("Training did not result in a best model. No model saved.")


def accuracy_fn(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    return (preds == labels).float().mean()

def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

# Main Function
def main():
    parser = argparse.ArgumentParser(description="Pretrain DistilBERT on MEMD-ABSA Dataset Single Domains")
    parser.add_argument('--root_dir', type=str, default='/scratch/gautschi/apiasecz/data/MEMD-ABSA/MEMD_ABSA_Dataset/', help='Root directory of the MEMD-ABSA dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model', type=str, default='DistilBertForSentiment', choices=['DistilBertForSentiment'], help='Model to train')
    parser.add_argument('--domains', type=str, nargs='+', default=['Books', 'Clothing', 'Hotel', 'Laptop', 'Restaurant'], help='Domains to train on')
    parser.add_argument('--num_epochs', type=int, default=10, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--model_save_dir', type=str, default='/scratch/gautschi/apiasecz/models/concept_drift_models/', help='Model save directory')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length for tokenizer')
    args = parser.parse_args()

    os.makedirs(args.model_save_dir, exist_ok=True)

    # --- Caching: Load tokenizer once ---
    print("Loading DistilBert tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', local_files_only=True)
    
    for domain in args.domains:
        print(f"\n===== Processing domain: {domain} =====")
        
        # --- Caching: Instantiate a new model for each domain ---
        model_class = DistilBertForSentiment
        
        model_filename = f"{args.model}_{domain}_seed_{args.seed}.pth"
        model_save_path = os.path.join(args.model_save_dir, model_filename)

        train_and_evaluate_model(
            model_class=model_class,
            model_path=model_save_path,
            seed=args.seed,
            domain=domain,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optimizer_choice=args.optimizer,
            root_dir=args.root_dir,
            max_len=args.max_len,
            tokenizer=tokenizer # Pass the cached tokenizer
        )
        
        # Cross-domain evaluation
        print("\n===== Cross-Domain Evaluation =====")
        cross_accuracies = {source: {} for source in args.domains}
        
        for source_domain in args.domains:
            model_filename = f"{args.model}_{source_domain}_seed_{args.seed}.pth"
            model_path = os.path.join(args.model_save_dir, model_filename)
            
            if not os.path.exists(model_path):
                print(f"Model for {source_domain} not found at {model_path}. Skipping.")
                continue
            
            # Load the pretrained model
            model_net = BaseNeuralNetwork(DistilBertForSentiment, device=device)
            model_net.load_model(model_path)
            print(f"Loaded model pretrained on {source_domain}")
            
            for target_domain in args.domains:
                test_dataset = load_data_from_domain(args.root_dir, target_domain, 'Test', tokenizer, args.max_len)
                if test_dataset is None or len(test_dataset) == 0:
                    print(f"Missing Test data for domain {target_domain}. Skipping.")
                    continue
                
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                test_accuracy = model_net.evaluate(test_loader, accuracy_fn)
                cross_accuracies[source_domain][target_domain] = test_accuracy
                print(f"Model pretrained on {source_domain} evaluated on {target_domain} Test: Accuracy = {test_accuracy:.4f}")
        
    # Print cross-domain accuracy table
    print("\nCross-Domain Accuracy Table (Rows: Pretrained on, Columns: Tested on):")
    header = "Pretrained \\ Tested | " + " | ".join(args.domains)
    print(header)
    print("-" * len(header))
    for source in args.domains:
        row = f"{source:<18} | " + " | ".join(f"{cross_accuracies[source].get(target, 'N/A'):.4f}" for target in args.domains)
        print(row)
    print("\nAll Trainings Completed")

if __name__ == "__main__":
    main()
