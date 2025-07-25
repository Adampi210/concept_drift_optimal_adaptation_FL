import os
import random
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

# Import the base model architecture. We will not use the other toolkit
# components for this specific task, as Causal LM requires a custom loop.
from fl_toolkit import BaseModelArchitecture

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- New Data Handler Section for Multilingual MC4 Dataset ---

class C4Dataset(Dataset):
    """PyTorch Dataset for the C4/MC4 dataset, prepared for Causal LM."""
    def __init__(self, data, tokenizer, max_len=256):
        self.encodings = []
        print(f"Tokenizing data... This may take a moment.")
        for item in data:
            # Tokenize each text sample
            tokenized = tokenizer(item['text'], max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
            self.encodings.append(tokenized)
        print("Tokenization complete.")

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        # Return the dictionary of tokenized inputs
        item = {key: val.squeeze() for key, val in self.encodings[idx].items()}
        return item

class MC4DataHandler:
    """Data handler for the MC4 dataset for multilingual text prediction."""
    def __init__(self, domains=None, samples_per_domain=10000, model_name='bigscience/bloom-560m'):
        # Use the tokenizer corresponding to the multilingual model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set pad token if it's not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.domains = domains if domains else ['en', 'de', 'fr', 'es'] # Using 4 domains as requested
        self.samples_per_domain = samples_per_domain
        self.datasets = self._load_data()

    def _load_data(self):
        datasets = {}
        for domain in self.domains:
            print(f"Loading data for language domain: {domain}")
            try:
                # Use streaming to avoid downloading the entire massive MC4 dataset
                # The domain is the language code for the multilingual C4 dataset
                dataset = load_dataset('mc4', domain, split='train', streaming=True, trust_remote_code=True)
                # Take a sample from the stream
                domain_sample = list(dataset.take(self.samples_per_domain))
                datasets[domain] = domain_sample
            except Exception as e:
                print(f"Could not load data for domain {domain}. Error: {e}")
        return datasets

    def get_domain_dataset(self, domain, max_len=256):
        """Returns a tokenized PyTorch Dataset for a specific domain."""
        if domain not in self.datasets:
            raise ValueError(f"Domain '{domain}' not loaded.")
        return C4Dataset(self.datasets[domain], self.tokenizer, max_len)

# --- End of New Data Handler Section ---


# Define the new text prediction model
class BloomLM(BaseModelArchitecture):
    """
    A wrapper for the BLOOM-560m model for Causal Language Modeling.
    """
    def __init__(self, model_name='bigscience/bloom-560m'):
        super(BloomLM, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        # The model will return loss when labels are provided
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

# Utility Functions
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Main Training Function for Causal LM
def train_model(model_class, model_path, seed, domain, num_epochs, batch_size, learning_rate, optimizer_choice, results_save_path, max_len):
    """Main function to train and save a Causal LM on a single language domain."""
    set_seed(seed)
    
    # --- Data Loading ---
    data_handler = MC4DataHandler(samples_per_domain=20000) # Load more samples for better training
    full_dataset = data_handler.get_domain_dataset(domain, max_len=max_len)
    
    # Split data into 80% training, 20% holdout
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # --- Model and Optimizer ---
    model = model_class().to(device)
    if optimizer_choice.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    else: 
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # --- Custom Training and Evaluation Loop ---
    results = {'perplexity': [], 'loss': []}
    best_perplexity = float('inf')

    print(f"\n--- Starting training on language domain '{domain}' ---")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # The labels are the input_ids themselves for Causal LM
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            if i % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        
        results['loss'].append(avg_val_loss)
        results['perplexity'].append(perplexity)
        
        print(f"\n--- Epoch {epoch+1}/{num_epochs} Summary ---")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Perplexity: {perplexity:.4f}")
        print("---------------------------------\n")

        # Save the best model based on perplexity
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path} with perplexity: {best_perplexity:.4f}")

    # --- Save Final Results ---
    data = {
        "parameters": {
            "model_architecture": model_class.__name__, "domain": domain, "seed": seed,
            "num_epochs": num_epochs, "batch_size": batch_size, "learning_rate": learning_rate,
            "optimizer": optimizer_choice, "model_save_path": model_path, "max_len": max_len
        },
        "results": results
    }

    with open(results_save_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Final results saved to {results_save_path}")


# Main execution block
def main():
    parser = argparse.ArgumentParser(description="Pretrain Multilingual Text Prediction Models on MC4")
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--models', type=str, nargs='+', default=['BloomLM'], help='Models to train')
    parser.add_argument('--domains', type=str, nargs='+', default=['en', 'de', 'fr', 'es'], help='MC4 language domains to train on')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for text models (smaller for Causal LM)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for transformers')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'], help='Optimizer to use')
    parser.add_argument('--model_save_dir', type=str, default='./models/text_prediction_models/', help='Directory to save models')
    parser.add_argument('--results_save_dir', type=str, default='./results/text_prediction_results/', help='Directory to save results')
    parser.add_argument('--max_len', type=int, default=256, help='Max sequence length for tokenizer')
    parser.add_argument('--num_seeds', type=int, default=1, help='Number of different seeds to run')
    args = parser.parse_args()

    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.results_save_dir, exist_ok=True)

    for model_name in args.models:
        model_class = globals().get(model_name)
        if not model_class:
            print(f"Model {model_name} not defined. Skipping.")
            continue
        for domain in args.domains:
            print(f"\n--- Preparing to train {model_name} on language domain: {domain} ---")
            for seed_offset in range(args.num_seeds):
                seed = args.seed + seed_offset
                
                model_filename = f"{model_name}_{domain}_len_{args.max_len}_seed_{seed}.pth"
                model_save_path = os.path.join(args.model_save_dir, model_filename)

                results_filename = f"{model_name}_{domain}_len_{args.max_len}_seed_{seed}_results.json"
                results_save_path = os.path.join(args.results_save_dir, results_filename)

                train_model(
                    model_class=model_class, model_path=model_save_path, seed=seed,
                    domain=domain, num_epochs=args.num_epochs, batch_size=args.batch_size,
                    learning_rate=args.learning_rate, optimizer_choice=args.optimizer,
                    results_save_path=results_save_path, max_len=args.max_len,
                )
    print("\n--- All training sessions completed ---")

if __name__ == "__main__":
    main()
