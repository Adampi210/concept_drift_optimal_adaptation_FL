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
    def __init__(self, num_classes=7):
        super(PACSCNN, self).__init__()
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
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
   \

class Policy:
    def __init__(self):
        self.virtual_queue = 0
        self.update_history = []
        self.previous_loss = None
        
    def estimate_ci(self, current_loss):
        if not self.update_history or self.previous_loss is None:
            return 1.0  # Default value for initialization
            
        loss_diff = abs(current_loss - self.previous_loss)
        time_diff = self.update_history[-1][0] - (len(self.update_history) > 1 and self.update_history[-2][0] or 0)
        
        if time_diff == 0:
            return 1.0
            
        return loss_diff / time_diff
        
    def policy_decision(self, decision_id: int, **kwargs):
        pi_bar = kwargs.get('pi_bar', 0.1)
        current_time = kwargs.get('current_time', 0)
        current_loss = kwargs.get('current_loss', None)
        V = kwargs.get('V', 1.0)
        delta = kwargs.get('delta', 0.1)
        lambda_cost = kwargs.get('lambda_cost', 1.0)
        
        if decision_id == 0:
            return 0
            
        elif decision_id == 1:
            # Uniform random policy with expectation pi_bar
            return np.random.random() < pi_bar
            
        elif decision_id == 2:
            # Fixed interval policy
            interval = int(1 / pi_bar)
            return current_time % interval == 0
            
        elif decision_id == 3:
            # Lyapunov optimization based policy
            if current_loss is None:
                return 0
                
            # Estimate c_i
            ci = self.estimate_ci(current_loss)
            
            # Update virtual queue
            threshold = (self.virtual_queue + 0.5 - pi_bar/lambda_cost) / (V * delta)
            
            # Make decision
            should_update = ci > threshold
            
            # Update queue
            if should_update:
                self.virtual_queue = max(0, self.virtual_queue + 1 - pi_bar/lambda_cost)
                self.update_history.append((current_time, current_loss))
                self.previous_loss = current_loss
                
            return int(should_update)
            
        else:
            raise ValueError("Invalid decision_id")

    def reset(self):
        """Reset internal state"""
        self.virtual_queue = 0
        self.update_history = []
        self.previous_loss = None

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    beta=1.0, 
    pi_bar=0.1,
    V=1
):
    set_seed(seed)
    # Initialize the policy
    policy = Policy()
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
        decision = policy.policy_decision(decision_id=policy_id, pi_bar=pi_bar, V=V, current_time=t)
        if decision:
            print('Retraining')
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
    parser = argparse.ArgumentParser(description="PACS CNN Evaluation with Drift Handling")

    # Evaluation arguments
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--src_domains', type=str, nargs='+', default=['photo'],
                        help='List of source domains (1-3 domains)')
    parser.add_argument('--tgt_domains', type=str, nargs='+', default=['sketch'],
                        help='List of target domains for evaluation')
    parser.add_argument('--drift_rate', type=float, default=0.01, help='Drift rate')
    parser.add_argument('--n_rounds', type=int, default=300, help='Number of evaluation rounds')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for SGD')
    parser.add_argument('--policy_id', type=int, default=2, help='Policy ID for retraining decisions')
    parser.add_argument('--setting_id', type=int, default=1, help='Setting ID for hyperparameter configuration')
    parser.add_argument('--alpha', type=float, default=1.0, help='Cost function alpha parameter')
    parser.add_argument('--beta', type=float, default=1.0, help='Cost function beta parameter')
    
    args = parser.parse_args()

    # Construct model path based on source domains
    domains_str = "_".join(args.src_domains)
    model_path = f"../../../models/concept_drift_models/PACSCNN_{domains_str}_seed_{args.seed}.pth"
    
    # SETTINGS
    # setting_id = 0: pi_bar = 0.1, drift_rate = 0.01, V = 1
    # setting_id = 1: pi_bar = 0.3, drift_rate = 0.01, V = 1
    # setting_id = 2: pi_bar = 0.5, drift_rate = 0.01, V = 1
    # setting_id = 3: pi_bar = 0.7, drift_rate = 0.01, V = 1
    # setting_id = 4: pi_bar = 0.1, drift_rate = 0.01, V = 0.01
    # setting_id = 5: pi_bar = 0.3, drift_rate = 0.01, V = 0.01
    # setting_id = 6: pi_bar = 0.5, drift_rate = 0.01, V = 0.01
    # setting_id = 7: pi_bar = 0.7, drift_rate = 0.01, V = 0.01
    # setting_id = 8: pi_bar = 0.1, drift_rate = 0.01, V = 100
    # setting_id = 9: pi_bar = 0.3, drift_rate = 0.01, V = 100
    # setting_id = 10: pi_bar = 0.5, drift_rate = 0.01, V = 100
    # setting_id = 11: pi_bar = 0.7, drift_rate = 0.01, V = 100
    # Hyperparams
    # Decrease the drift rate 
    pi_bar = 0.3
    V = 1
    print(f'policy_id: {args.policy_id}')
    print(f'setting_id: {args.setting_id}')
    print(f'drift_rate: {args.drift_rate}')
    print(f'pi_bar: {pi_bar}')
    print(f'V: {V}')
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
        beta=args.beta, 
        pi_bar=pi_bar,
        V=V
    )


# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()