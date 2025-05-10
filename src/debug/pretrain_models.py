import os
import random
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import numpy as np

from fl_toolkit import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Models
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
    accuracy = agent.evaluate(metric_fn=accuracy_fn, verbose=False) * 100
    print(f"After {num_epochs} epochs: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")
    domain_counts = {domain: agent.len_domain_samples(domain) for domain in ['photo', 'sketch', 'art_painting', 'cartoon']}
    print(f"Domain counts in current dataset: {domain_counts}")
    return accuracy

# Training Function
def train_model(model_class, model_path, seed, domain, num_epochs, batch_size, learning_rate, optimizer_choice, results_save_path, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = PACSDataHandler(transform=transform).dataset
    dataset_size = len(full_dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)  # Randomize indices

    # Split into 80% training, 20% holdout
    train_size = int(0.8 * dataset_size)
    train_indices = indices[:train_size]
    holdout_indices = indices[train_size:]

    # Create subsets
    train_subset = Subset(full_dataset, train_indices)
    holdout_subset = Subset(full_dataset, holdout_indices)
    
    train_data_handler = PACSDataHandler()
    train_data_handler.dataset = train_subset
    holdout_data_handler = PACSDataHandler()
    holdout_data_handler.dataset = holdout_subset
    
    train_drift = PACSDomainDrift(
        train_data_handler,
        source_domains=[domain],
        target_domains=[domain],
        drift_rate=0,  # No drift during pretraining
        desired_size=None  # Use all available data
    )
    
    holdout_drift = PACSDomainDrift(
        holdout_data_handler,
        source_domains=[domain],
        target_domains=[domain],
        drift_rate=0,  # No drift during pretraining
        desired_size=None  # Use all available data
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
    
    for i in range(num_epochs):
        avg_loss = agent_train.update_steps(num_updates=10, optimizer=optimizer, loss_fn=criterion, verbose=True)
        agent_holdout.set_model_params(agent_train.get_model_params())
        avg_loss_array.append(avg_loss)
        print(f"Epoch {i+1}/{num_epochs}")
        print("Training dataset:")
        print_debug_info(agent_train, i, avg_loss, accuracy_fn)
        print("Holdout dataset:")
        accuracy = print_debug_info(agent_holdout, i, avg_loss, accuracy_fn)
        accuracy_array.append(accuracy)
        if accuracy > 75:
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
            "img_size": img_size
        },
        "accuracy": accuracy_array,
        "loss": avg_loss_array
    }

    with open(results_save_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to {results_save_path}")

# Main Function
def main():
    parser = argparse.ArgumentParser(description="Pretrain Models on PACS Dataset Single Domains")
    parser.add_argument('--seed', type=int, default=0, help='Base random seed')
    parser.add_argument('--models', type=str, nargs='+', default=['PACSCNN_1', 'PACSCNN_2', 'PACSCNN_3', 'PACSCNN_4'],
                        help='Models to train')
    parser.add_argument('--domains', type=str, nargs='+', default=['photo', 'sketch', 'art_painting', 'cartoon'],
                        help='Domains to train on')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of update steps')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'], help='Optimizer')
    parser.add_argument('--model_save_dir', type=str, default='../../../../models/concept_drift_models/', help='Model save directory')
    parser.add_argument('--results_save_dir', type=str, default='../../data/results/', help='Results save directory')
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    parser.add_argument('--num_seeds', type=int, default=20, help='Number of seeds')
    args = parser.parse_args()

    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.results_save_dir, exist_ok=True)

    for model_name in args.models:
        try:
            model_class = globals()[model_name]
        except KeyError:
            print(f"Model {model_name} not defined. Skipping.")
            continue
        for domain in args.domains:
            print(f"\nTraining {model_name} on {domain}")
            for seed_offset in range(args.num_seeds):
                seed = args.seed + seed_offset
                print(f"Seed: {seed}")
                set_seed(seed)

                model_filename = f"{model_name}_{domain}_img_{args.img_size}_seed_{seed}.pth"
                model_save_path = os.path.join(args.model_save_dir, model_filename)

                results_filename = f"{model_name}_{domain}_img_{args.img_size}_seed_{seed}_results.json"
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
                    img_size=args.img_size,
                )
    print("\nAll Trainings Completed")

if __name__ == "__main__":
    main()