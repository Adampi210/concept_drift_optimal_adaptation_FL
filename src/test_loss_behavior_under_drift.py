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


# Apply retraining according to policy
def retrain_with_policy_under_drift(
    source_domains,
    target_domains,
    model_path,
    seed,
    drift_rate,
    n_rounds,
    learning_rate=0.01,
    policy_id=0,
    setting_id=0,
    batch_size=128,
    alpha=1.0,
    beta=1.0
):
    set_seed(seed)
    
    # Initialize data handler
    data_handler = PACSDataHandler()
    data_handler.load_data()
    train_data, test_data = data_handler.train_dataset, data_handler.test_dataset
    
    # Set up domain drift for training and testing
    train_drift = PACSDomainDrift(
        source_domains=source_domains,
        target_domains=target_domains,
        drift_rate=drift_rate
    )
    
    test_drift = PACSDomainDrift(
        source_domains=source_domains,
        target_domains=target_domains,
        drift_rate=drift_rate
    )
    
    # Initialize client
    client = FederatedDriftClient(
        client_id=0,
        model_architecture=PACSCNN,
        train_domain_drift=train_drift,
        test_domain_drift=test_drift
    )
    
    # Set up data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    client.set_data(train_loader, test_loader)
    
    # Load pre-trained model
    model = client.get_model()
    model.load_state_dict(torch.load(model_path))
    
    # Initialize optimizer and criterion
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize results list
    results = []
    
    # Training loop with policy decisions
    for t in range(n_rounds):
        # Apply drift
        client.apply_test_drift()
        client.apply_train_drift()
        
        # Evaluate current performance
        current_accuracy = client.evaluate(metric_fn=accuracy_fn, verbose=False)
        current_loss = client.evaluate(metric_fn=criterion, verbose=True)
        
        # Get policy decision
        decision = policy_decision(decision_id=policy_id)
        
        # Calculate cost
        cost = cost_function(decision, current_loss, alpha=alpha, beta=beta)
        
        # Retrain if policy decides to
        if decision == 1:
            client.train(
                epochs=1,
                optimizer=optimizer,
                loss_fn=criterion,
                verbose=False
            )
        
        # Save results
        results.append({
            't': t,
            'accuracy': current_accuracy,
            'loss': current_loss,
            'decision': decision,
            'cost': cost
        })
        
        # Optional progress printing
        print(f"Round {t}: Accuracy = {current_accuracy:.4f}, Decision = {decision}, Cost = {cost:.4f}")
    
    # Save results to CSV with hyperparameters
    src_domains_str = "_".join(source_domains)
    tgt_domains_str = "_".join(target_domains) 
    results_filename = f"../data/results/policy_{policy_id}_setting_{setting_id}_src_domains_{src_domains_str}_tgt_domains_{tgt_domains_str}_seed_{seed}.csv"
    
    with open(results_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # First write all hyperparameters
        writer.writerow(['Parameter', 'Value'])
        writer.writerow(['source_domains', ','.join(source_domains)])
        writer.writerow(['target_domains', ','.join(target_domains)])
        writer.writerow(['model_path', model_path])
        writer.writerow(['seed', seed])
        writer.writerow(['drift_rate', drift_rate])
        writer.writerow(['n_rounds', n_rounds])
        writer.writerow(['learning_rate', learning_rate])
        writer.writerow(['policy_id', policy_id])
        writer.writerow(['setting_id', setting_id])
        writer.writerow(['batch_size', batch_size])
        writer.writerow(['alpha', alpha])
        writer.writerow(['beta', beta])
        
        # Add a blank row for separation
        writer.writerow([])
        
        # Write the experimental results
        writer.writerow(['t', 'accuracy', 'loss', 'decision', 'cost'])
        for result in results:
            writer.writerow([result['t'], result['accuracy'], result['loss'], 
                           result['decision'], result['cost']])
    
    return results

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
                            help='List of source domains (1-3 domains)')
    eval_parser.add_argument('--tgt_domains', type=str, nargs='+', default=['art_painting'],
                            help='List of target domains for evaluation')
    eval_parser.add_argument('--drift_rate', type=float, default=0.1, help='Drift rate')
    eval_parser.add_argument('--n_rounds', type=int, default=300, help='Number of evaluation rounds')
    eval_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for SGD')
    eval_parser.add_argument('--policy_id', type=int, default=0, help='Policy ID for retraining decisions')
    eval_parser.add_argument('--setting_id', type=int, default=0, help='Setting ID for hyperparameter configuration')
    eval_parser.add_argument('--alpha', type=float, default=1.0, help='Cost function alpha parameter')
    eval_parser.add_argument('--beta', type=float, default=1.0, help='Cost function beta parameter')
    
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
        model_filename = f"../../../models/concept_drift_models/model_domains_{domains_str}.pth"
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
        
    if args.mode == 'evaluate':
        # Construct model path based on source domains
        domains_str = "_".join(args.src_domains)
        model_path = f"../../../models/concept_drift_models/model_domains_{domains_str}.pth"
        
        # Run evaluation with drift
        results = retrain_with_policy_under_drift(
            source_domains=args.src_domains,
            target_domains=args.tgt_domains,
            model_path=model_path,
            seed=args.seed,
            drift_rate=args.drift_rate,
            n_rounds=args.n_rounds,
            learning_rate=args.lr,
            policy_id=args.policy_id,
            setting_id=args.setting_id,
            alpha=args.alpha,
            beta=args.beta
        )
    else:
        print("Invalid mode selected. Choose 'train' or 'evaluate'.")


# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()