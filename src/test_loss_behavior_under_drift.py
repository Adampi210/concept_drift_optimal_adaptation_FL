import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import csv
import argparse

from fl_toolkit import *  # Ensure fl_toolkit is correctly installed and accessible

class PACSCNN(BaseModelArchitecture):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 7)  # 7 classes in PACS
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Decide if retrain or not
def policy_decision(decision_id=0, **kwargs):
    if decision_id == 0:
        return 0
    elif decision_id == 1:
        return 1
    else:
        raise ValueError("Invalid decision_id")

# Calculate cost of update/no update
def cost_function(decision, current_loss, alpha=1.0, beta=1.0):
    return alpha * decision + beta * current_loss

# Save results to CSV
def save_results(results, filepath):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'loss', 'decision', 'cost'])  # Header
        for result in results:
            writer.writerow([result['t'], result['loss'], result['decision'], result['cost']])

# Train the model
def train_model(model_path, seed, source_domains, epochs, batch_size, learning_rate, optimizer_choice):
    set_seed(seed)
    
    # Get train and test data
    data_handler = PACSDataHandler()
    data_handler.load_data()
    train_data, test_data = data_handler.train_dataset, data_handler.test_dataset
    
    available_domains = ['photo', 'cartoon', 'sketch', 'art_painting']
    for domain in source_domains:
        if domain not in available_domains:
            raise ValueError(f"Invalid source domain: {domain}. Available domains: {available_domains}")
    
    # Only have source domains avilable for training
    train_drift = PACSDomainDrift(
        source_domains=source_domains,
        target_domains=source_domains,
        drift_rate=1,
    )
    
    # Only have source domains available for testing
    test_drift = PACSDomainDrift(
        source_domains=source_domains,
        target_domains=source_domains,
        drift_rate=1,
    )
    
    # Initialize client
    client = FederatedDriftClient(
        client_id=0,
        model_architecture=PACSCNN, 
        train_domain_drift=train_drift,
        test_domain_drift=test_drift
    )
    
    # Set data
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    client.set_data(train_loader, test_loader)
    
    # Initialize model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = client.get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_choice.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_choice.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer. Choose 'adam' or 'sgd'.")
        
    # Training loop
    best_accuracy = 0.0
    for epoch in range(1, epochs + 1):
        client.train(
            epochs=2,
            optimizer=optimizer,
            loss_fn=criterion,
            verbose=True)
            
        # Evaluate
        accuracy = client.evaluate(metric_fn=accuracy_fn, verbose=False) * 100
        print(f'Epoch {epoch}: Accuracy = {accuracy:.2f}%')
        
        # Save best acc
        if accuracy > best_accuracy:
            bast_accuracy = accuracy
        # Check if accuracy threshold met
        if best_accuracy >= 0.85:
            print(f"Desired accuracy reached: {best_accuracy:.4f}% at epoch {epoch}. Stopping training.")
            break
        
    torch.save(client.model.get_params(), model_path)
            
    print(f"Training completed. Best Accuracy: {best_accuracy:.4f}%")

# Evaluate model under drift
def evaluate_under_drift(**kwargs):
    pass

# Apply retraining according to policy
def retrain_with_policy_under_drift():
    pass

# Main Function
def main():
    parser = argparse.ArgumentParser(description="PACS CNN Training and Evaluation with Drift Handling")
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation: train or evaluate', required=True)
    
    # Training subparser
    train_parser = subparsers.add_parser('train', help='Train the PACSCNN model')
    train_parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    train_parser.add_argument('--src_domains', type=str, nargs='+', default=['photo'], 
                              help='List of source domains to train on (choose from photo, cartoon, sketch, art_painting)')
    train_parser.add_argument('--epochs', type=int, default=200, help='Maximum number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    train_parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer choice')
    train_parser.add_argument('--model_save_dir', type=str, default='../data/models/', help='Directory to save the trained model')
    
    # Evaluation subparser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the PACSCNN model with drift')
    eval_parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    eval_parser.add_argument('--src_domains', type=str, nargs='+', default=['photo'], 
                            help='List of source domains to train on (choose from photo, cartoon, sketch, art_painting)')
    eval_parser.add_argument('--tgt_domains', type=str, nargs='+', default=['sketch'], 
                             help='List of target domains to evaluate on (choose from photo, cartoon, sketch, art_painting)')
    eval_parser.add_argument('--drift_rate', type=float, default=0.1, help='Drift rate to apply (e.g., 0.1)')
    eval_parser.add_argument('--model_path', type=str, default='../data/models/model_domains_photo.pth', help='Path to the trained model')
    eval_parser.add_argument('--n_rounds', type=int, default=200, help='Number of drift evaluation rounds')
    eval_parser.add_argument('--results_dir', type=str, default='../data/results', help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Extract training arguments
        seed = args.seed
        source_domains = args.src_domains
        epochs = args.epochs
        batch_size = args.batch_size
        lr = args.lr
        optimizer_choice = args.optimizer
        model_save_dir = args.model_save_dir
        
        # Create model save directory if it doesn't exist
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Define model save path with domains in filename
        domains_str = "_".join(source_domains)
        model_filename = f"model_domains_{domains_str}.pth"
        model_path = os.path.join(model_save_dir, model_filename)
        
        # Train the model
        train_model(
            model_path=model_path,
            seed=seed,
            source_domains=source_domains,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            optimizer_choice=optimizer_choice,
        )
        
    elif args.mode == 'evaluate':
        # Extract evaluation arguments
        seed = args.seed
        target_domains = args.target_domains
        drift_rate = args.drift_rate
        model_path = args.model_path
        n_rounds = args.n_rounds
        results_dir = args.results_dir
        
        # Validate target domains
        available_domains = ['photo', 'cartoon', 'sketch', 'art_painting']
        for domain in target_domains:
            if domain not in available_domains:
                raise ValueError(f"Invalid target domain: {domain}. Available domains: {available_domains}")
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Evaluate the model with drift
        evaluate_under_drift(
            model_path=model_path,
            seed=seed,
            target_domains=target_domains,
            drift_rate=drift_rate,
            n_rounds=n_rounds,
            results_dir=results_dir
        )
        
    else:
        print("Invalid mode selected. Choose 'train' or 'evaluate'.")


# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()