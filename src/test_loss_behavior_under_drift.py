# examples/drift_evaluation_loss.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from fl_toolkit import *
import os
from enum import Enum
import random
import csv
import argparse

class DriftConfig(Enum):
    SLOW_ROTATION = "slow_rotation"
    MEDIUM_ROTATION = "medium_rotation"
    FAST_ROTATION = "fast_rotation"
    SLOW_SCALING = "slow_scaling"
    MEDIUM_SCALING = "medium_scaling"
    FAST_SCALING = "fast_scaling"
    MILD_NOISE = "mild_noise"
    MEDIUM_NOISE = "medium_noise"
    STRONG_NOISE = "strong_noise"
    COMPOSITE = "composite"

class CIFAR10Net(BaseModelArchitecture):
    """ResNet-like architecture for CIFAR10."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)

def get_drift_config(config_name):
    """Get drift configuration based on name."""
    if isinstance(config_name, str):
        config_name = DriftConfig(config_name.lower())
        
    if config_name == DriftConfig.SLOW_ROTATION:
        return ConceptDrift('rotation', angle_range=(-2, 2))
        
    elif config_name == DriftConfig.MEDIUM_ROTATION:
        return ConceptDrift('rotation', angle_range=(-5, 5))
        
    elif config_name == DriftConfig.FAST_ROTATION:
        return ConceptDrift('rotation', angle_range=(-10, 10))
        
    elif config_name == DriftConfig.SLOW_SCALING:
        return ConceptDrift('scaling', scale_range=(0.95, 1.05))
        
    elif config_name == DriftConfig.MEDIUM_SCALING:
        return ConceptDrift('scaling', scale_range=(0.9, 1.1))
        
    elif config_name == DriftConfig.FAST_SCALING:
        return ConceptDrift('scaling', scale_range=(0.8, 1.2))
        
    elif config_name == DriftConfig.MILD_NOISE:
        return ConceptDrift('noise', noise_std=0.05)
        
    elif config_name == DriftConfig.MEDIUM_NOISE:
        return ConceptDrift('noise', noise_std=0.1)
        
    elif config_name == DriftConfig.STRONG_NOISE:
        return ConceptDrift('noise', noise_std=0.2)
        
    elif config_name == DriftConfig.COMPOSITE:
        return ConceptDrift('composite', transforms_list=[
            ConceptDrift('rotation', angle_range=(-5, 5)).transform,
            ConceptDrift('noise', noise_std=0.05).transform
        ])
    
    else:
        raise ValueError(f"Unknown drift configuration: {config_name}")

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def save_results(results, filepath):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'loss'])  # header
        for result in results:
            writer.writerow([result['t'], result['loss']])

def train_and_save_model(model_path='../data/models/cifar10_model.pth', train_epochs=50):
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return
    
    print("Training new model...")
    
    # Setup
    data_handler = CIFAR10DataHandler()
    data_handler.load_data('~/data/CIFAR10/')
    
    client = FederatedDriftClient(
        client_id=0,
        model_architecture=CIFAR10Net()
    )

    # Get dataloaders
    train_loader = data_handler.get_train_loader(batch_size=128)
    test_loader = data_handler.get_test_loader(batch_size=128)
    client.set_data(train_loader, test_loader)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(client.model.model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(train_epochs):
        # Train
        client.train(
            epochs=2,
            optimizer=optimizer,
            loss_fn=criterion,
            verbose=True)
            
        # Evaluate
        accuracy = client.evaluate(metric_fn=accuracy_fn, verbose=False) * 100
        
        print(f'Epoch {epoch}: Accuracy = {accuracy:.2f}%')
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': client.model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, model_path)
            print(f'Saved model with accuracy: {accuracy:.2f}%')
            
        scheduler.step(accuracy)

def evaluate_with_drift(model_path, seed, config_name, n_rounds=50, results_dir='results'):
    # Setup
    data_handler = CIFAR10DataHandler()
    data_handler.load_data('~/data/CIFAR10/')
    
    client = FederatedDriftClient(
        client_id=0,
        model_architecture=CIFAR10Net()
    )
    
    # Load trained model
    checkpoint = torch.load(model_path)
    client.model.model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model with accuracy: {checkpoint['accuracy']:.2f}%")
    
    # Get test loader
    test_loader = data_handler.get_test_loader(batch_size=128)
    client.set_data(None, test_loader)
    
    # Setup drift conditions
    data_drift = None  # For now just test IID, no data drift
    concept_drift = get_drift_config(config_name)
    
    client.update_drift(data_drift, concept_drift)
    
    # Evaluation loop
    criterion = nn.CrossEntropyLoss()
    results = []
    
    # First evaluation without drift
    loss = client.evaluate(metric_fn=criterion, verbose=True)
    results.append({'t': 0, 'loss': loss})
    print(f"Round 0 (No Drift) - Loss: {loss:.4f}")
    
    # Evaluation rounds with drift
    for t in range(1, n_rounds + 1):
        # Apply drift for this round
        client.apply_drift()
        
        # Evaluate
        loss = client.evaluate(metric_fn=criterion, verbose=True)
        results.append({'t': t, 'loss': loss})
        print(f"Round {t} - Loss: {loss:.4f}")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'loss_eval_drift_{config_name}_seed_{seed}.csv')
    save_results(results, csv_path)
    print(f"Results saved to: {csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--config', type=str, required=True, 
                       choices=[c.value for c in DriftConfig])
    args = parser.parse_args()
    
    SEED = args.seed
    CONFIG = args.config
    MODEL_PATH = '../data/models/cifar10_model.pth'
    RESULTS_DIR = '../data/results/'
    N_ROUNDS = 50
    
    set_seed(SEED)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print('evaluate')
    evaluate_with_drift(MODEL_PATH, SEED, CONFIG, N_ROUNDS, RESULTS_DIR)
    
if __name__ == '__main__':
    main()