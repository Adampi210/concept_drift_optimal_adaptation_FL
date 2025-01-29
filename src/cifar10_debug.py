import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import csv
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import PIL
from PIL import Image
import argparse
from fl_toolkit import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_batch(dataloader, round_num, prefix="train", classes=None):
    """Visualize a batch of images in a 2x5 grid"""
    # Get a batch
    batch_size = 10
    vis_loader = DataLoader(dataloader.dataset, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(vis_loader))
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'Round {round_num} - {prefix.capitalize()} Data Samples')
    
    # Plot each image
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // 5
        col = idx % 5
        
        # Convert image for display
        if isinstance(img, torch.Tensor):
            img = img.numpy().transpose(1, 2, 0)
            # Unnormalize
            img = img * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
            img = np.clip(img, 0, 1)
            
        axes[row, col].imshow(img)
        if classes:
            title = classes[label]
        else:
            title = f"Class {label}"
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'batch_vis/{prefix}_round_{round_num}.png')
    plt.close()

class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def retrain_with_policy_under_drift(model_path, seed, drift_type, drift_rate, n_rounds):
    set_seed(seed)
    os.makedirs('batch_vis', exist_ok=True)
    
    # Setup data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='/scratch/gilbreth/apiasecz/data/', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='/scratch/gilbreth/apiasecz/data/', train=False, download=True, transform=transform)
    
    # Setup drift
    drift_transforms = {
        'color_balance': CIFAR10DriftTypes.color_balance_drift(factor=0.4),
        'darkness': CIFAR10DriftTypes.bounded_darkness_drift(factor=0.2),
        'soft_gamma': CIFAR10DriftTypes.soft_gamma_drift(gamma_factor=0.3),
        'color_wash': CIFAR10DriftTypes.color_wash_drift(factor=0.3)
    }
    
    train_drift = CIFAR10DomainDrift(drift_rate=drift_rate, transform=drift_transforms[drift_type], seed=seed)
    test_drift = CIFAR10DomainDrift(drift_rate=drift_rate, transform=drift_transforms[drift_type], seed=seed+1)
    
    # Setup client
    client = FederatedDriftClientCIFAR10(
        client_id=0,
        model_architecture=CIFAR10CNN,
        train_domain_drift=train_drift,
        test_domain_drift=test_drift,
        device=device
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    client.set_data(train_loader, test_loader)
    
    if model_path:
        client.model.load_state_dict(torch.load(model_path))
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(client.model.parameters(), lr=0.001)
    
    # CIFAR10 classes for visualization
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    results = []
    test_results = []
    
    for t in range(n_rounds):
        # Apply drift
        client.apply_test_drift()
        client.apply_train_drift()
        
        # Visualize current batches
        visualize_batch(client.train_loader, t, "train", classes)
        visualize_batch(client.test_loader, t, "test", classes)
        
        # Evaluate
        train_accuracy = client.get_train_metric(metric_fn=accuracy_fn)
        test_accuracy = client.evaluate(metric_fn=accuracy_fn)
        train_loss = client.get_train_metric(metric_fn=criterion)
        test_loss = client.evaluate(metric_fn=criterion)
        
        # Save results
        results.append({
            't': t,
            'accuracy': train_accuracy,
            'loss': train_loss
        })
        
        test_results.append({
            't': t,
            'accuracy': test_accuracy,
            'loss': test_loss
        })
        
        print(f"Round {t}:")
        print(f"Train - Accuracy = {train_accuracy:.4f}, Loss = {train_loss:.4f}")
        print(f"Test  - Accuracy = {test_accuracy:.4f}, Loss = {test_loss:.4f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    with open(f'results/train_results_{drift_type}_rate_{drift_rate}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'accuracy', 'loss'])
        for i, result in enumerate(results):
            writer.writerow([i, result['accuracy'], result['loss']])
            
    with open(f'results/test_results_{drift_type}_rate_{drift_rate}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'accuracy', 'loss'])
        for i, result in enumerate(test_results):
            writer.writerow([i, result['accuracy'], result['loss']])
    
    return results, test_results

def main():
    drift_transforms = {
        'color_balance': CIFAR10DriftTypes.color_balance_drift(factor=0.4),
        'darkness': CIFAR10DriftTypes.bounded_darkness_drift(factor=0.2),
        'soft_gamma': CIFAR10DriftTypes.soft_gamma_drift(gamma_factor=0.3),
        'color_wash': CIFAR10DriftTypes.color_wash_drift(factor=0.3)
    }
    
    parser = argparse.ArgumentParser(description="CIFAR10 CNN Evaluation with Drift Handling")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--drift_type', type=str, default='color_wash',
                      choices=drift_transforms.keys())
    parser.add_argument('--drift_rate', type=float, default=0.5)
    parser.add_argument('--n_rounds', type=int, default=10)
    parser.add_argument('--model_path', type=str, default='../../../models/concept_drift_models/CIFAR10CNN_seed_1.pth')
    
    args = parser.parse_args()
    
    results, test_results = retrain_with_policy_under_drift(
        model_path=args.model_path,
        seed=args.seed,
        drift_type=args.drift_type,
        drift_rate=args.drift_rate,
        n_rounds=args.n_rounds
    )

if __name__ == '__main__':
    main()