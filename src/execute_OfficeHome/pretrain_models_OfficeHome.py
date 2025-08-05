import os
import random
import argparse
import json
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from torchvision.models import resnet18 as _resnet18
from torchvision.models import resnet34 as _resnet34
from torchvision.models import resnet50 as _resnet50

import numpy as np

from fl_toolkit import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OfficeHomeNet(BaseModelArchitecture):
    def __init__(self, num_classes=65):
        super(OfficeHomeNet, self).__init__()
        self.model = _resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        # Freeze conv1, bn1, layer1, layer2, layer3
        for param in self.model.conv1.parameters():
            param.requires_grad = False
        for param in self.model.bn1.parameters():
            param.requires_grad = False
        for param in self.model.layer1.parameters():
            param.requires_grad = False
        for param in self.model.layer2.parameters():
            param.requires_grad = False
        for param in self.model.layer3.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)
    
    def get_params(self):
        return self.state_dict() 
    

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
    time_start = time.time()
    accuracy = agent.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False) * 100
    print(f"Time to evaluate model: {time.time() - time_start:.2f} seconds")
    return accuracy

def count_parameters(model):
    """Counts the number of total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# Training Function
def train_model(model_class, model_path, seed, domain, num_epochs, batch_size, learning_rate, optimizer_choice, results_save_path, img_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Assuming img_size=224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    holdout_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    root_dir = f'/scratch/gautschi/apiasecz/data/OfficeHome/OfficeHomeDataset_10072016/' ### directory with the OfficeHome dataset
    full_dataset = OfficeHomeDataHandler(root_dir=root_dir, transform=train_transform).dataset
    dataset_size = len(full_dataset)
    print(f"Dataset size: {dataset_size}")
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)  # Randomize indices
    time_start = time.time()
    # Split into 80% training, 20% holdout
    train_size = int(0.8 * dataset_size)
    train_indices = indices[:train_size]
    holdout_indices = indices[train_size:]
    
    # Create subsets
    train_data_handler = OfficeHomeDataHandler(root_dir=root_dir, transform=train_transform)
    train_data_handler.set_subset(train_indices)
    holdout_data_handler = OfficeHomeDataHandler(root_dir=root_dir, transform=holdout_transform)
    holdout_data_handler.set_subset(holdout_indices)

    train_drift = DomainDrift(
        train_data_handler,
        source_domains=[domain],
        target_domains=[domain],
        drift_rate=0,  # No drift during pretraining
        desired_size=None  # Use all available data
    )
    
    holdout_drift = DomainDrift(
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
    print(f"Time to load data: {time.time() - time_start:.2f} seconds")
    for i in range(num_epochs):
        print(f"Epoch {i+1}/{num_epochs}")
        time_start = time.time()
        avg_loss = agent_train.update_steps(num_updates=20, optimizer=optimizer, loss_fn=criterion, verbose=True)
        print(f"Time to update model: {time.time() - time_start:.2f} seconds")
        agent_holdout.set_model_params(agent_train.get_model_params())
        time_start = time.time()
        avg_loss_array.append(avg_loss)
        print("Holdout dataset:")
        accuracy = print_debug_info(agent_holdout, i, avg_loss, accuracy_fn)
        print(f"Holdout accuracy: {accuracy:.2f}%")
        accuracy_array.append(accuracy)
        if accuracy > 90:
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
    model = OfficeHomeNet()
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    exit()
    
    parser = argparse.ArgumentParser(description="Pretrain Models on OfficeHome Dataset Single Domains")
    parser.add_argument('--seed', type=int, default=0, help='Base random seed')
    parser.add_argument('--model', type=str, default='OfficeHomeNet', choices=['OfficeHomeNet',],
                        help='Models to train')
    parser.add_argument('--domains', type=str, nargs='+', default=['RealWorld', 'Clipart', 'Product', 'Art'],
                        help='Domains to train on')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of update steps')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'], help='Optimizer')
    parser.add_argument('--model_save_dir', type=str, default='../../models/concept_drift_models/', help='Model save directory')
    parser.add_argument('--results_save_dir', type=str, default='../../data/results/', help='Results save directory')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    args = parser.parse_args()

    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.results_save_dir, exist_ok=True)

    model_name = args.model
    try:
        model_class = globals()[model_name]
    except KeyError:
        print(f"Model {model_name} not defined. Skipping.")
    for domain in args.domains:
        print(f"\nTraining {model_name} on {domain}")
        seed = args.seed
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