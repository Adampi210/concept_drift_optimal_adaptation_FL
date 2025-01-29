import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import csv
import argparse
import torchvision
import torchvision.transforms as transforms

from fl_toolkit import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
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
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Policy:
    def __init__(self):
        self.virtual_queue = 0
        self.update_history = []        

    def policy_decision(self, decision_id: int, **kwargs):
        pi_bar = kwargs.get('pi_bar', 0.1)
        loss_diff = kwargs.get('loss_diff', 0)
        current_time = kwargs.get('current_time', 0)
        V = kwargs.get('V', 1.0)
        
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
            threshold = (self.virtual_queue + 0.5 - pi_bar) / V
            should_update = loss_diff > threshold
            
            self.virtual_queue = max(0, self.virtual_queue + should_update - pi_bar)
            
            if should_update:
                self.update_history.append(current_time)
                
            return int(should_update)
            
        else:
            raise ValueError("Invalid decision_id")

    def reset(self):
        self.virtual_queue = 0
        self.update_history = []

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def retrain_with_policy_under_drift(
    model_path,
    seed,
    drift_type,
    drift_rate,
    n_rounds,
    learning_rate=0.01,
    policy_id=0,
    setting_id=0,
    batch_size=128, 
    pi_bar=0.1,
    V=1
):
    set_seed(seed)
    policy = Policy()
    
    # Load CIFAR10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='/scratch/gilbreth/apiasecz/data/', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='/scratch/gilbreth/apiasecz/data/', train=False, download=True, transform=transform)
    
    # Get appropriate drift transform
    drift_transforms = {
        'gaussian': CIFAR10DriftTypes.gaussian_noise(),
        'rotation': CIFAR10DriftTypes.rotation_drift(),
        'blur': CIFAR10DriftTypes.blur_drift(),
        'color': CIFAR10DriftTypes.color_shift(),
        'intensity': CIFAR10DriftTypes.intensity_drift()
    }
    print(drift_transforms[drift_type])
    
    train_drift = CIFAR10DomainDrift(
        drift_rate=drift_rate,
        desired_size=50000,
        transform=drift_transforms[drift_type]
    )
    test_drift = CIFAR10DomainDrift(
        drift_rate=drift_rate,
        desired_size=10000,
        transform=drift_transforms[drift_type]
    )
    
    # Set up client
    client = FederatedDriftClientCIFAR10(
        client_id=0,
        model_architecture=CIFAR10CNN,
        train_domain_drift=train_drift,
        test_domain_drift=test_drift, 
        device=device
    )
    
    # Set up initial data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    client.set_data(train_loader, test_loader)

    model = client.get_model()
    if model_path:
        model.load_state_dict(torch.load(model_path))
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    loss_array = []
    loss_array.append(client.get_train_metric(metric_fn=criterion, verbose=False))
    
    results = []
    
    for t in range(n_rounds):
        # Apply drifts
        client.apply_test_drift()
        client.apply_train_drift()
        
        # Evaluate current performance
        current_accuracy = client.evaluate(metric_fn=accuracy_fn, verbose=False)
        current_loss = client.get_train_metric(metric_fn=criterion, verbose=False)
        loss_array.append(current_loss)
        # Get policy decision
        decision = policy.policy_decision(decision_id=policy_id, pi_bar=pi_bar, V=V, current_time=t, loss_diff=loss_array[-1] - loss_array[-2]) 

        # Get policy decision
        decision = policy.policy_decision(
            decision_id=policy_id,
            pi_bar=pi_bar,
            V=V,
            current_time=t,
            loss_diff=loss_array[-1] - loss_array[-2]
        )
        
        # Retrain if policy decides to
        if decision:
            current_loss = client.train(
                epochs=5,
                optimizer=optimizer,
                loss_fn=criterion,
                verbose=True
            )
        else:
            current_loss = client.get_train_metric(metric_fn=criterion, verbose=False)
        
        # Save results
        results.append({
            't': t,
            'accuracy': current_accuracy,
            'loss': current_loss,
            'decision': decision
        })
        
        print(f"Round {t}: Accuracy = {current_accuracy:.4f}, Loss = {current_loss:.4f}, Decision = {decision}")
    
    # Save results
    save_results(results, drift_type, policy_id, setting_id, seed)
    
    return results


def save_results(results, drift_type, policy_id, setting_id, seed):
    filename = f"results/cifar10_drift_{drift_type}_policy_{policy_id}_setting_{setting_id}_seed_{seed}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'accuracy', 'loss', 'decision'])
        for result in results:
            writer.writerow([result['t'], result['accuracy'], result['loss'], result['decision']])

def main():
    parser = argparse.ArgumentParser(description="CIFAR10 CNN Evaluation with Drift Handling")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--drift_type', type=str, default='gaussian',
                      choices=['gaussian', 'rotation', 'blur', 'color', 'intensity'])
    parser.add_argument('--n_rounds', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--policy_id', type=int, default=0)
    parser.add_argument('--setting_id', type=int, default=0)
    parser.add_argument('--model_path', type=str, default=None)
    
    args = parser.parse_args()
    model_path = f"../../../models/concept_drift_models/CIFAR10CNN_seed_{args.seed}.pth"
    # Settings dictionary (can be expanded based on your needs)
    settings = {
        0: {'pi_bar': 0.1, 'drift_rate': 0.1, 'V': 65},
        1: {'pi_bar': 0.15, 'drift_rate': 0.01, 'V': 65},
        2: {'pi_bar': 0.20, 'drift_rate': 0.01, 'V': 65},
        3: {'pi_bar': 0.05, 'drift_rate': 0.01, 'V': 65},
        4: {'pi_bar': 0.03, 'drift_rate': 0.01, 'V': 65},
        5: {'pi_bar': 0.1, 'drift_rate': 0.03, 'V': 65},
        6: {'pi_bar': 0.15, 'drift_rate': 0.03, 'V': 65},
        7: {'pi_bar': 0.20, 'drift_rate': 0.03, 'V': 65},
        8: {'pi_bar': 0.05, 'drift_rate': 0.03, 'V': 65},
        9: {'pi_bar': 0.03, 'drift_rate': 0.03, 'V': 65},
        10: {'pi_bar': 0.1, 'drift_rate': 0.003, 'V': 65},
        11: {'pi_bar': 0.15, 'drift_rate': 0.003, 'V': 65},
        12: {'pi_bar': 0.20, 'drift_rate': 0.003, 'V': 65},
        13: {'pi_bar': 0.05, 'drift_rate': 0.003, 'V': 65},
        14: {'pi_bar': 0.03, 'drift_rate': 0.003, 'V': 65},
        15: {'pi_bar': 0.1, 'drift_rate': 0.01, 'V': 115},
        16: {'pi_bar': 0.15, 'drift_rate': 0.01, 'V': 115},
        17: {'pi_bar': 0.20, 'drift_rate': 0.01, 'V': 115},
        18: {'pi_bar': 0.05, 'drift_rate': 0.01, 'V': 115},
        19: {'pi_bar': 0.03, 'drift_rate': 0.01, 'V': 115},
        20: {'pi_bar': 0.1, 'drift_rate': 0.03, 'V': 115},
        21: {'pi_bar': 0.15, 'drift_rate': 0.03, 'V': 115},
        22: {'pi_bar': 0.20, 'drift_rate': 0.03, 'V': 115},
        23: {'pi_bar': 0.05, 'drift_rate': 0.03, 'V': 115},
        24: {'pi_bar': 0.03, 'drift_rate': 0.03, 'V': 115},
        25: {'pi_bar': 0.1, 'drift_rate': 0.003, 'V': 115},
        26: {'pi_bar': 0.15, 'drift_rate': 0.003, 'V': 115},
        27: {'pi_bar': 0.20, 'drift_rate': 0.003, 'V': 115},
        28: {'pi_bar': 0.05, 'drift_rate': 0.003, 'V': 115},
        # ... other settings as needed
    }
    
    results = retrain_with_policy_under_drift(
        model_path=model_path,
        seed=args.seed,
        drift_type=args.drift_type,
        drift_rate=settings[args.setting_id]['drift_rate'],
        n_rounds=args.n_rounds,
        learning_rate=args.lr,
        policy_id=args.policy_id,
        setting_id=args.setting_id,
        pi_bar=settings[args.setting_id]['pi_bar'],
        V=settings[args.setting_id]['V']
    )

if __name__ == "__main__":
    main()