import os
import random
import argparse
import csv
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Define Multiple Models
# =========================

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


def get_model(model_name, num_classes=10):
        return CIFAR10CNN(num_classes=num_classes)

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

# =========================
# Training and Evaluation
# =========================

def train_model(model_architecture, model_path, seed, epochs, batch_size, learning_rate, optimizer_choice, results_save_path):
    set_seed(seed)

    # Get train and test data
    data_handler = CIFAR10DataHandler()
    data_handler.load_data(data_dir='/scratch/gilbreth/apiasecz/data/')
    train_data, test_data = data_handler.train_dataset, data_handler.test_dataset

    # Initialize client with the specified model architecture
    client = FederatedClient(
        client_id=0,
        model_architecture=model_architecture,
        device=device
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
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)

    # Training loop
    best_accuracy = 0.0
    results = []

    for epoch in range(1, epochs + 1):
        # Train for one epoch
        client.train(
            epochs=2,
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

        # Update best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        # Step the scheduler
        scheduler.step(accuracy)

        # Check if accuracy threshold met
        if best_accuracy >= 85.0:
            print(f"Desired accuracy reached: {best_accuracy:.2f}% at epoch {epoch}. Stopping training.")
            break

    # Save the best model
    torch.save(client.model.get_params(), model_path)
    print(f"Training completed for {model_architecture.__name__}. Best Accuracy: {best_accuracy:.2f}%")

    # Save the results
    save_accuracy(results, results_save_path)
    print(f"Accuracy results saved to {results_save_path}")


# =========================
# Main Function
# =========================

def main():
    parser = argparse.ArgumentParser(description="Train CIFAR10 model")
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--models', type=str, nargs='+', default=['CIFAR10CNN',],
                        help='List of model architectures to train')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer choice')
    parser.add_argument('--model_save_dir', type=str, default='../../../models/concept_drift_models/', help='Directory to save trained models')
    parser.add_argument('--results_save_dir', type=str, default='../data/results/', help='Directory to save training results')
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Ensure save directories exist
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.results_save_dir, exist_ok=True)

    # Initialize data handler
    data_handler = CIFAR10DataHandler()
    data_handler.load_data(data_dir='/scratch/gilbreth/apiasecz/data/')

    for model_name in args.models:
        # Instantiate the model class
        try:
            model_class = globals()[model_name]
        except KeyError:
            print(f"Model {model_name} is not defined. Skipping.")
            continue
        print(f"\n========== Training {model_name} ==========")

        # Define model save path
        model_filename = f"{model_name}_seed_{args.seed}.pth"
        model_save_path = os.path.join(args.model_save_dir, model_filename)

        # Define results save path
        results_filename = f"{model_name}_seed_{args.seed}_results.csv"
        results_save_path = os.path.join(args.results_save_dir, results_filename)

        # Train the model
        try:
            train_model(
                model_architecture=model_class,
                model_path=model_save_path,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                optimizer_choice=args.optimizer,
                results_save_path=results_save_path
            )
        except Exception as e:
            print(f"An error occurred while training {model_name}: {e}")
            continue

    print("\n========== All Trainings Completed ==========")


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    main()
