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

from fl_toolkit import *
from torchvision import models, transforms

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

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(SEResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction)
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
        out = self.se(out)
        out += identity
        out = self.relu(out)
        return out

class PACSCNN_1(BaseModelArchitecture):
    def __init__(self, num_classes=7):
        super(PACSCNN_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(64, num_classes)
        self.apply(self.init_weights)

    @torch.no_grad()
    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class PACSCNN_2(BaseModelArchitecture):
    def __init__(self, num_classes=7):
        super(PACSCNN_2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=2),
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
    def __init__(self, num_classes=7):
        super(PACSCNN_3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=2),
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
    def __init__(self, num_classes=7):
        super(PACSCNN_4, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
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

class PACSCNN_5(BaseModelArchitecture):
    def __init__(self, num_classes=7):
        super(PACSCNN_5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 128, stride=1),  # Added
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1),  # Added
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1),
            ResidualBlock(512, 512, stride=1),  # Added
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

class PACSCNN_6(BaseModelArchitecture):
    def __init__(self, num_classes=7):
        super(PACSCNN_6, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEResidualBlock(64, 64, stride=1),
            SEResidualBlock(64, 64, stride=1),
            SEResidualBlock(64, 128, stride=2),
            SEResidualBlock(128, 128, stride=1),
            SEResidualBlock(128, 256, stride=2),
            SEResidualBlock(256, 256, stride=1),
            SEResidualBlock(256, 512, stride=2),
            SEResidualBlock(512, 512, stride=1),
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(model_name, num_classes=7):
    return PACSCNN(num_classes=num_classes)

def save_accuracy(results, save_path):
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model

def filter_dataset_by_domain(dataset, domain):
    if not hasattr(dataset, 'domains'):
        raise AttributeError("Dataset does not have a 'domains' attribute.")
    indices = [i for i, d in enumerate(dataset.domains) if d == domain]
    return Subset(dataset, indices)

# =========================
# Training and Evaluation
# =========================

def train_model(model_architecture, model_path, seed, domains, epochs, batch_size, learning_rate, optimizer_choice, results_save_path, img_size):
    set_seed(seed)

    data_handler = PACSDataHandler()
    data_handler.load_data()
    train_data, test_data = data_handler.train_dataset, data_handler.test_dataset

    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data.transform = transform
    test_data.transform = transform

    available_domains = ['photo', 'cartoon', 'sketch', 'art_painting']
    for domain in domains:
        if domain not in available_domains:
            raise ValueError(f"Invalid domain: {domain}. Available domains: {available_domains}")

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

    client = FederatedDriftClient(
        client_id=0,
        model_architecture=model_architecture,
        train_domain_drift=train_drift,
        test_domain_drift=test_drift
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    client.set_data(train_loader, test_loader)

    model = client.get_model().to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_choice.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_choice.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer. Choose 'adam' or 'sgd'.")

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)

    best_accuracy = 0.0
    best_epoch = 0
    results = []

    for epoch in range(1, epochs + 1):
        client.train(
            epochs=1,
            optimizer=optimizer,
            loss_fn=criterion,
            verbose=True
        )
        accuracy = client.evaluate(metric_fn=accuracy_fn, verbose=False) * 100
        loss = client.evaluate(metric_fn=criterion, verbose=False)
        print(f'Epoch {epoch}: Accuracy = {accuracy:.2f}%, Loss = {loss}')

        results.append({
            'epoch': epoch,
            'accuracy': accuracy,
            'loss': loss
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch

        scheduler.step(accuracy)

        if best_accuracy >= 85.0:
            print(f"Desired accuracy reached: {best_accuracy:.2f}% at epoch {epoch}. Stopping training.")
            break

    torch.save(client.model.get_params(), model_path)
    print(f"Training completed for {model_architecture.__name__} on {'_'.join(domains)}. Best Accuracy: {best_accuracy:.2f}%")

    data = {
        "parameters": {
            "model_architecture": model_architecture.__name__,
            "domains": domains,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer_choice,
            "drift_rate": 1.0,
            "model_save_path": model_path,
            "img_size": img_size
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
    parser.add_argument('--models', type=str, nargs='+', default=['PACSCNN_1', 'PACSCNN_2', 'PACSCNN_3','PACSCNN_4',],
                        help='List of model architectures to train')
    parser.add_argument('--domains', type=str, nargs='+', default=['photo', 'cartoon', 'sketch', 'art_painting'],
                        help='List of domains to train on')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs per model-domain')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'], help='Optimizer choice')
    parser.add_argument('--model_save_dir', type=str, default='../../../../models/concept_drift_models/', help='Directory to save trained models')
    parser.add_argument('--results_save_dir', type=str, default='../../data/results/', help='Directory to save training results')
    parser.add_argument('--img_size', type=int, default=64, help='Size to resize images to (img_size x img_size)')
    parser.add_argument('--num_seeds', type=int, default=20, help='Number of seeds to train for each model-domain')
    args = parser.parse_args()

    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.results_save_dir, exist_ok=True)

    data_handler = PACSDataHandler()
    data_handler.load_data()

    available_domains = ['photo', 'cartoon', 'sketch', 'art_painting']
    for domain in args.domains:
        if domain not in available_domains:
            raise ValueError(f"Invalid domain: {domain}. Available domains: {available_domains}")

    for model_name in args.models:
        try:
            model_class = globals()[model_name]
        except KeyError:
            print(f"Model {model_name} is not defined. Skipping.")
            continue
        domain_str = '_'.join(args.domains)
        print(f"\n========== Training {model_name} on Domain: {domain_str} ==========")

        try:
            for seed in range(args.num_seeds):
                print(f'Using seed {seed}')
                set_seed(seed)
                
                model_filename = f"{model_name}_{domain_str}_img_size_{args.img_size}_seed_{seed}.pth"
                model_save_path = os.path.join(args.model_save_dir, model_filename)

                results_filename = f"{model_name}_{domain_str}_img_size_{args.img_size}_seed_{seed}_results.json"
                results_save_path = os.path.join(args.results_save_dir, results_filename)

                train_model(
                    model_architecture=model_class,
                    model_path=model_save_path,
                    seed=seed,
                    domains=args.domains,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    optimizer_choice=args.optimizer,
                    results_save_path=results_save_path,
                    img_size=args.img_size
                )
        except Exception as e:
            print(f"An error occurred while training {model_name} on {domain_str}: {e}")
            continue

    print("\n========== All Trainings Completed ==========")

# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    main()