import os
import random
import argparse
import csv
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset

import numpy as np
import pandas as pd

from fl_toolkit import *  # Ensure fl_toolkit is correctly installed and accessible
from torchvision import models

# Set device
torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Define Multiple Models
# =========================

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

class ResidualBlock(nn.Module):
    """A residual block with skip connections for PACSCNN_4."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class PACSCNN_1(BaseModelArchitecture):
    """A simple CNN with 2 blocks."""
    def __init__(self, num_classes=7):
        super(PACSCNN_1, self).__init__()
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=2),
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class PACSCNN_2(BaseModelArchitecture):
    """A moderate CNN with 3 blocks."""
    def __init__(self, num_classes=7):
        super(PACSCNN_2, self).__init__()
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=2),
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=2),
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=2),
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

class PACSCNN_3(BaseModelArchitecture):
    """A CNN with 4 blocks, similar to the original PACSCNN."""
    def __init__(self, num_classes=7):
        super(PACSCNN_3, self).__init__()
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=2),
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=2),
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=2),
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=2),
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

class PACSCNN_4(BaseModelArchitecture):
    """A deeper CNN with residual blocks and skip connections."""
    def __init__(self, num_classes=7):
        super(PACSCNN_4, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 channels
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
            # Downsample to 128 channels
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
            # Downsample to 256 channels
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
            # Downsample to 512 channels
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1),
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

# =========================
# Utility Functions
# =========================

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name, num_classes=7):
        return PACSCNN(num_classes=num_classes)

def save_accuracy(results, save_path):
    """Save the accuracy results to a CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)


def save_model(model, save_path):
    """Save the model state_dict."""
    torch.save(model.state_dict(), save_path)

def load_model(model, model_path):
    """Load the model state_dict."""
    model.load_state_dict(torch.load(model_path))
    return model

def filter_dataset_by_domain(dataset, domain):
    """
    Filter the dataset to include only samples from the specified domain.

    Assumes that the dataset has a 'domains' attribute that lists the domain for each sample.
    Modify this function based on the actual structure of PACSDataHandler.
    """
    if not hasattr(dataset, 'domains'):
        raise AttributeError("Dataset does not have a 'domains' attribute. Please modify the filter_dataset_by_domain function accordingly.")

    # Get indices where the domain matches
    indices = [i for i, d in enumerate(dataset.domains) if d == domain]
    return Subset(dataset, indices)

# =========================
# Training and Evaluation
# =========================

def train_model(model_architecture, model_path, seed, domains, epochs, batch_size, learning_rate, optimizer_choice, results_save_path):
    set_seed(seed)

    # Get train and test data
    data_handler = PACSDataHandler()
    data_handler.load_data()
    train_data, test_data = data_handler.train_dataset, data_handler.test_dataset

    available_domains = ['photo', 'cartoon', 'sketch', 'art_painting']
    for domain in domains:
        if domain not in available_domains:
            raise ValueError(f"Invalid domain: {domain}. Available domains: {available_domains}")

    # Only have the specific domain available for training and testing
    train_drift = PACSDomainDrift(
        source_domains=domains,
        target_domains=domains,
        drift_rate=1,
    )

    test_drift = PACSDomainDrift(
        source_domains=domains,
        target_domains=domains,
        drift_rate=1,
    )

    # Initialize client with the specified model architecture
    client = FederatedDriftClient(
        client_id=0,
        model_architecture=model_architecture,
        train_domain_drift=train_drift,
        test_domain_drift=test_drift
    )

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    client.set_data(train_loader, test_loader)

    # Initialize model, loss, optimizer
    model = client.get_model().to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_choice.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_choice.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer. Choose 'adam' or 'sgd'.")

    # Learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)

    # Training loop
    best_accuracy = 0.0
    best_epoch = 0  # Added to track the epoch of best accuracy
    results = []

    for epoch in range(1, epochs + 1):
        # Train for one epoch
        client.train(
            epochs=1,
            optimizer=optimizer,
            loss_fn=criterion,
            verbose=True
        )

        # Evaluate
        accuracy = client.evaluate(metric_fn=accuracy_fn, verbose=False) * 100
        loss = client.evaluate(metric_fn=criterion, verbose=False)
        print(f'Epoch {epoch}: Accuracy = {accuracy:.2f}%, Loss = {loss}')

        # Record results
        results.append({
            'epoch': epoch,
            'accuracy': accuracy,
            'loss': loss
        })

        # Update best accuracy and epoch
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch  # Update best_epoch when best_accuracy improves

        # Step the scheduler
        scheduler.step(accuracy)

        # Check if accuracy threshold met
        if best_accuracy >= 85.0:
            print(f"Desired accuracy reached: {best_accuracy:.2f}% at epoch {epoch}. Stopping training.")
            break

    # Save the best model
    torch.save(client.model.get_params(), model_path)
    print(f"Training completed for {model_architecture.__name__} on {'_'.join(domains)}. Best Accuracy: {best_accuracy:.2f}%")

    # Save results to JSON
    data = {
        "parameters": {
            "model_architecture": model_architecture.__name__,
            "domains": domains,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer_choice,
            "drift_rate": 1.0,  # Fixed drift rate used in training
            "model_save_path": model_path
        },
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "results": results
    }

    with open(results_save_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Results saved to {results_save_path}")
    

# =========================
# Main Function
# =========================

def main():
    parser = argparse.ArgumentParser(description="Train Multiple Models on PACS Dataset Across Domains")
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--models', type=str, nargs='+', default=['PACSCNN_1', 'PACSCNN_2', 'PACSCNN_3', 'PACSCNN_4'],
                        help='List of model architectures to train')
    parser.add_argument('--domains', type=str, nargs='+', default=['photo', 'cartoon', 'sketch', 'art_painting'],
                        help='List of domains to train on')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs per model-domain')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer choice')
    parser.add_argument('--model_save_dir', type=str, default='../../../../models/concept_drift_models/', help='Directory to save trained models')
    parser.add_argument('--results_save_dir', type=str, default='../../data/results/', help='Directory to save training results')
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Ensure save directories exist
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.results_save_dir, exist_ok=True)

    # Initialize data handler
    data_handler = PACSDataHandler()
    data_handler.load_data()

    available_domains = ['photo', 'cartoon', 'sketch', 'art_painting']
    for domain in args.domains:
        if domain not in available_domains:
            raise ValueError(f"Invalid domain: {domain}. Available domains: {available_domains}")

    for model_name in args.models:
        # Instantiate the model class
        try:
            model_class = globals()[model_name]
        except KeyError:
            print(f"Model {model_name} is not defined. Skipping.")
            continue
        domain_str = '_'.join(args.domains)
        print(f"\n========== Training {model_name} on Domain: {domain_str} ==========")

        # Define model save path
        model_filename = f"{model_name}_{domain_str}_seed_{args.seed}.pth"
        model_save_path = os.path.join(args.model_save_dir, model_filename)

        # Define results save path
        results_filename = f"{model_name}_{domain_str}_seed_{args.seed}_results.json"
        results_save_path = os.path.join(args.results_save_dir, results_filename)

        # Train the model
        try:
            train_model(
                model_architecture=model_class,
                model_path=model_save_path,
                seed=args.seed,
                domains=args.domains,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                optimizer_choice=args.optimizer,
                results_save_path=results_save_path
            )
        except Exception as e:
            print(f"An error occurred while training {model_name} on {domain}: {e}")
            continue

    print("\n========== All Trainings Completed ==========")


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    main()
