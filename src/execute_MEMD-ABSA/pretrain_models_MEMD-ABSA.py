import os
import json
import random
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers import DistilBertTokenizer
import numpy as np
from fl_toolkit import *
from transformers import DistilBertForSequenceClassification

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MEMDABSANet model
class MEMDABSANet(BaseModelArchitecture):
    def __init__(self, num_classes=3):  # POS, NEG, NEU
        super(MEMDABSANet, self).__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=num_classes
        )
        # Freeze all but the classifier and last transformer layer
        for param in self.model.distilbert.embeddings.parameters():
            param.requires_grad = False
        for i in range(5):  # DistilBERT has 6 transformer layers; freeze first 5
            for param in self.model.distilbert.transformer.layer[i].parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def get_params(self):
        return self.state_dict()

# Utility Functions
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def print_debug_info(agent, num_epochs, loss, accuracy_fn):
    time_start = time.time()
    accuracy = agent.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False) * 100
    print(f"Time to evaluate model: {time.time() - time_start:.2f} seconds")
    return accuracy

# Training Function
def train_model(model_class, model_path, seed, domain, num_epochs, batch_size, learning_rate, optimizer_choice, results_save_path):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    root_dir = f'/scratch/gautschi/apiasecz/data/MEMD-ABSA/MEMD_ABSA_Dataset/'  # REPLACE WITH YOUR ACTUAL PATH
    full_dataset = MEMDABSADataHandler(root_dir=root_dir, transform=tokenizer).dataset
    dataset_size = len(full_dataset)
    print(f"Dataset size: {dataset_size}")
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)  # Randomize indices
    time_start = time.time()
    # Split into 80% training, 20% holdout
    train_size = int(0.8 * dataset_size)
    train_indices = indices[:train_size]
    holdout_indices = indices[train_size:]
    
    # Create subsets
    train_data_handler = MEMDABSADataHandler(root_dir=root_dir, transform=tokenizer)
    train_data_handler.set_subset(train_indices)
    holdout_data_handler = MEMDABSADataHandler(root_dir=root_dir, transform=tokenizer)
    holdout_data_handler.set_subset(holdout_indices)

    train_drift = DomainDrift(
        train_data_handler,
        source_domains=[domain],
        target_domains=[domain],
        drift_rate=0,  # No drift during pretraining
        desired_size=None  # Use all available data
    )
    
    holdout_drift = DomainDrift(
        holdout_data_handler,
        source_domains=[domain],
        target_domains=[domain],
        drift_rate=0,  # No drift during pretraining
        desired_size=None
    )
    
    agent_train = DriftAgent(
        client_id=0,
        model_architecture=model_class,
        domain_drift=train_drift,
        batch_size=batch_size,
        device=device
    )
    
    agent_holdout = DriftAgent(
        client_id=1,
        model_architecture=model_class,
        domain_drift=holdout_drift,
        batch_size=batch_size,
        device=device
    )

    model = agent_train.get_model().to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_choice.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_choice.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer. Choose 'adam' or 'sgd'.")
    
    accuracy_array = []
    avg_loss_array = []
    print(f"Time to load data: {time.time() - time_start:.2f} seconds")
    for i in range(num_epochs):
        print(f"Epoch {i+1}/{num_epochs}")
        time_start = time.time()
        avg_loss = agent_train.update_steps(num_updates=20, optimizer=optimizer, loss_fn=criterion, verbose=True)
        print(f"Time to update model: {time.time() - time_start:.2f} seconds")
        agent_holdout.set_model_params(agent_train.get_model_params())
        time_start = time.time()
        avg_loss_array.append(avg_loss)
        print("Holdout dataset:")
        accuracy = print_debug_info(agent_holdout, i, avg_loss, accuracy_fn)
        print(f"Holdout accuracy: {accuracy:.2f}%")
        accuracy_array.append(accuracy)
        if accuracy > 90:
            print(f"Early stopping at epoch {i+1} with accuracy {accuracy:.2f}%")
            break
    torch.save(agent_train.model.get_params(), model_path)
    print(f"Model saved to {model_path}")

    data = {
        "parameters": {
            "model_architecture": model_class.__name__,
            "domain": domain,
            "seed": seed,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer_choice,
            "model_save_path": model_path,
        },
        "accuracy": accuracy_array,
        "loss": avg_loss_array
    }

    with open(results_save_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to {results_save_path}")

# Main Function
def main():
    parser = argparse.ArgumentParser(description="Pretrain Models on MEMD-ABSA Dataset Single Domains")
    parser.add_argument('--seed', type=int, default=0, help='Base random seed')
    parser.add_argument('--model', type=str, default='MEMDABSANet', choices=['MEMDABSANet',],
                        help='Models to train')
    parser.add_argument('--domains', type=str, nargs='+', default=['Books', 'Clothing', 'Hotel', 'Restaurant', 'Laptop'],
                        help='Domains to train on')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of update steps')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')  # Smaller for text
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')  # Typical for transformers
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer')
    parser.add_argument('--model_save_dir', type=str, default='../../models/concept_drift_models/', help='Model save directory')
    parser.add_argument('--results_save_dir', type=str, default='../../data/results/', help='Results save directory')
    args = parser.parse_args()

    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.results_save_dir, exist_ok=True)

    model_name = args.model
    try:
        model_class = globals()[model_name]
    except KeyError:
        print(f"Model {model_name} not defined. Skipping.")
    for domain in args.domains:
        print(f"\nTraining {model_name} on {domain}")
        seed = args.seed
        print(f"Seed: {seed}")
        set_seed(seed)

        model_filename = f"{model_name}_{domain}_seed_{seed}.pth"
        model_save_path = os.path.join(args.model_save_dir, model_filename)
        
        results_filename = f"{model_name}_{domain}_seed_{seed}_results.json"
        results_save_path = os.path.join(args.results_save_dir, results_filename)

        train_model(
            model_class=model_class,
            model_path=model_save_path,
            seed=seed,
            domain=domain,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optimizer_choice=args.optimizer,
            results_save_path=results_save_path,
        )
    print("\nAll Trainings Completed")

if __name__ == "__main__":
    main()
