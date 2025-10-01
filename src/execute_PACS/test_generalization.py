import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import random
import csv
import argparse
import time
from torchvision.models import resnet18
from torchvision import transforms
from fl_toolkit import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"cuDNN Version: {torch.backends.cudnn.version()}")
print(f"PyTorch Version: {torch.__version__}\n")

# Define Model
class PACSCNN(BaseModelArchitecture):
    def __init__(self, num_classes=7):
        super(PACSCNN, self).__init__()
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
  
# Evaluate Policy Function
def evaluate_policy_under_drift(
    seed=0,
    learning_rate=0.01,
    n_steps=50
):
    set_seed(seed)
    # Set the seed and define transform
    
    model_path_photo = f"../../../../models/concept_drift_models/PACSCNN_photo_img_128_seed_{seed}.pth"
    model_path_art_painting = f"../../../../models/concept_drift_models/PACSCNN_art_painting_img_128_seed_{seed}.pth"
    model_path_cartoon = f"../../../../models/concept_drift_models/PACSCNN_cartoon_img_128_seed_{seed}.pth"
    model_path_sketch = f"../../../../models/concept_drift_models/PACSCNN_sketch_img_128_seed_{seed}.pth"
    
    domains = ['photo', 'art_painting', 'cartoon', 'sketch']
    
    # Define datasets for training+infrence and holdout for testing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
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
    train_data_handler = PACSDataHandler(transform=transform)
    train_data_handler.set_subset(train_indices)
    holdout_data_handler = PACSDataHandler(transform=transform)
    holdout_data_handler.set_subset(holdout_indices)
    
      
    # Define drift objects
    train_drift = DomainDrift(
        train_data_handler,
        source_domains=['photo',],
        target_domains=['photo',],
        drift_rate=0,  # Initially no drift
        desired_size=DSET_SIZE
    )
    holdout_drift_0 = DomainDrift(
        train_data_handler,
        source_domains=['photo',],
        target_domains=['photo',],
        drift_rate=0,  # Initially no drift
        desired_size=int(DSET_SIZE * 0.25)
    )
    holdout_drift_1 = DomainDrift(
        train_data_handler,
        source_domains=['art_painting',],
        target_domains=['art_painting',],
        drift_rate=0,  # Initially no drift
        desired_size=int(DSET_SIZE * 0.25)
    )
    holdout_drift_2 = DomainDrift(
        train_data_handler,
        source_domains=['cartoon',],
        target_domains=['cartoon',],
        drift_rate=0,  # Initially no drift
        desired_size=int(DSET_SIZE * 0.25)
    )
    holdout_drift_3 = DomainDrift(
        train_data_handler,
        source_domains=['sketch',],
        target_domains=['sketch',],
        drift_rate=0,  # Initially no drift
        desired_size=int(DSET_SIZE * 0.25)
    )
    
    # Agents: training and inference and holdout for performance
    agent_train = DriftAgent(
        client_id=0,
        model_architecture=PACSCNN,
        domain_drift=train_drift,
        batch_size=128,
        device=device
    )
    agent_train.apply_drift()
    
    agent_holdout_0 = DriftAgent(
        client_id=1,
        model_architecture=PACSCNN,
        domain_drift=holdout_drift_0,
        batch_size=128,
        device=device
    )
    agent_holdout_0.apply_drift()
    
    
    agent_holdout_1 = DriftAgent(
        client_id=2,
        model_architecture=PACSCNN,
        domain_drift=holdout_drift_1,
        batch_size=128,
        device=device
    )
    agent_holdout_1.apply_drift()
    
    agent_holdout_2 = DriftAgent(
        client_id=3,
        model_architecture=PACSCNN,
        domain_drift=holdout_drift_2,
        batch_size=128,
        device=device
    )
    agent_holdout_2.apply_drift()

    agent_holdout_3 = DriftAgent(
        client_id=4,
        model_architecture=PACSCNN,
        domain_drift=holdout_drift_3,
        batch_size=128,
        device=device
    )
    agent_holdout_3.apply_drift()

    for model_path in [
        model_path_photo, 
    ]:
        print(f'Loading model from {model_path}')
        model = agent_train.get_model()
        model.load_state_dict(torch.load(model_path, weights_only=False))
        agent_holdout_0.set_model_params(agent_train.get_model_params())
        agent_holdout_1.set_model_params(agent_train.get_model_params())
        agent_holdout_2.set_model_params(agent_train.get_model_params())
        agent_holdout_3.set_model_params(agent_train.get_model_params())
        
        # Set initial drift
        agent_train.set_drift_rate(1.0)
        agent_train.set_target_domains(['photo',])
        agent_train.apply_drift()
        
        # Optimizer and loss function
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
        print(f'Initial accuracies on holdout datasets:')
        print(f"Holdout 0 (photo): {agent_holdout_0.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 1 (art_painting): {agent_holdout_1.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 2 (cartoon): {agent_holdout_2.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 3 (sketch): {agent_holdout_3.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")

        print(f'Retraining model on art_painting dataset...')
        agent_train.set_drift_rate(1.0)
        agent_train.set_target_domains(['art_painting',])
        agent_train.apply_drift()

        for _ in range(5):
            agent_train.update_steps(
                num_updates=n_steps, 
                optimizer=optimizer, 
                loss_fn=criterion, 
                verbose=True
            )
            
        agent_holdout_0.set_model_params(agent_train.get_model_params())
        agent_holdout_1.set_model_params(agent_train.get_model_params())
        agent_holdout_2.set_model_params(agent_train.get_model_params())
        agent_holdout_3.set_model_params(agent_train.get_model_params())
            
        print(f'Updated accuracies on holdout datasets:')
        print(f"Holdout 0 (photo): {agent_holdout_0.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 1 (art_painting): {agent_holdout_1.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 2 (cartoon): {agent_holdout_2.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 3 (sketch): {agent_holdout_3.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        
        print(f'Retraining model on cartoon dataset...')
        agent_train.set_drift_rate(1.0)
        agent_train.set_target_domains(['cartoon',])
        agent_train.apply_drift()
        for _ in range(5):
            agent_train.update_steps(
                num_updates=n_steps, 
                optimizer=optimizer, 
                loss_fn=criterion, 
                verbose=True
            )
            
        agent_holdout_0.set_model_params(agent_train.get_model_params())
        agent_holdout_1.set_model_params(agent_train.get_model_params())
        agent_holdout_2.set_model_params(agent_train.get_model_params())
        agent_holdout_3.set_model_params(agent_train.get_model_params())
        
        print(f'Updated accuracies on holdout datasets:')
        print(f"Holdout 0 (photo): {agent_holdout_0.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 1 (art_painting): {agent_holdout_1.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 2 (cartoon): {agent_holdout_2.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 3 (sketch): {agent_holdout_3.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        
        
        print(f'Retraining model on sketch dataset...')
        agent_train.set_drift_rate(1.0)
        agent_train.set_target_domains(['sketch',])
        agent_train.apply_drift()
        
        for _ in range(5):
            agent_train.update_steps(
                num_updates=n_steps, 
                optimizer=optimizer, 
                loss_fn=criterion, 
                verbose=True
            )
            
        agent_holdout_0.set_model_params(agent_train.get_model_params())
        agent_holdout_1.set_model_params(agent_train.get_model_params())
        agent_holdout_2.set_model_params(agent_train.get_model_params())
        agent_holdout_3.set_model_params(agent_train.get_model_params())
        
        print(f'Updated accuracies on holdout datasets:')
        print(f"Holdout 0 (photo): {agent_holdout_0.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 1 (art_painting): {agent_holdout_1.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 2 (cartoon): {agent_holdout_2.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 3 (sketch): {agent_holdout_3.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        
        print(f'Retraining model on photo dataset...')
        agent_train.set_drift_rate(1.0)
        agent_train.set_target_domains(['photo',])
        agent_train.apply_drift()
        
        for _ in range(5):
            agent_train.update_steps(
                num_updates=n_steps, 
                optimizer=optimizer, 
                loss_fn=criterion, 
                verbose=True
            )
            
        agent_holdout_0.set_model_params(agent_train.get_model_params())
        agent_holdout_1.set_model_params(agent_train.get_model_params())
        agent_holdout_2.set_model_params(agent_train.get_model_params())
        agent_holdout_3.set_model_params(agent_train.get_model_params())
        
        
        print(f'Final accuracies on holdout datasets:')
        print(f"Holdout 0 (photo): {agent_holdout_0.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 1 (art_painting): {agent_holdout_1.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 2 (cartoon): {agent_holdout_2.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 3 (sketch): {agent_holdout_3.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
    

        print(f'Retraining model on art_painting dataset...')
        agent_train.set_drift_rate(1.0)
        agent_train.set_target_domains(['art_painting',])
        agent_train.apply_drift()

        for _ in range(5):
            agent_train.update_steps(
                num_updates=n_steps, 
                optimizer=optimizer, 
                loss_fn=criterion, 
                verbose=True
            )
            
        agent_holdout_0.set_model_params(agent_train.get_model_params())
        agent_holdout_1.set_model_params(agent_train.get_model_params())
        agent_holdout_2.set_model_params(agent_train.get_model_params())
        agent_holdout_3.set_model_params(agent_train.get_model_params())
            
        print(f'Updated accuracies on holdout datasets:')
        print(f"Holdout 0 (photo): {agent_holdout_0.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 1 (art_painting): {agent_holdout_1.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 2 (cartoon): {agent_holdout_2.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 3 (sketch): {agent_holdout_3.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        
        print(f'Retraining model on cartoon dataset...')
        agent_train.set_drift_rate(1.0)
        agent_train.set_target_domains(['cartoon',])
        agent_train.apply_drift()
        for _ in range(5):
            agent_train.update_steps(
                num_updates=n_steps, 
                optimizer=optimizer, 
                loss_fn=criterion, 
                verbose=True
            )
            
        agent_holdout_0.set_model_params(agent_train.get_model_params())
        agent_holdout_1.set_model_params(agent_train.get_model_params())
        agent_holdout_2.set_model_params(agent_train.get_model_params())
        agent_holdout_3.set_model_params(agent_train.get_model_params())
        
        print(f'Updated accuracies on holdout datasets:')
        print(f"Holdout 0 (photo): {agent_holdout_0.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 1 (art_painting): {agent_holdout_1.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 2 (cartoon): {agent_holdout_2.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 3 (sketch): {agent_holdout_3.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        
        
        print(f'Retraining model on sketch dataset...')
        agent_train.set_drift_rate(1.0)
        agent_train.set_target_domains(['sketch',])
        agent_train.apply_drift()
        
        for _ in range(5):
            agent_train.update_steps(
                num_updates=n_steps, 
                optimizer=optimizer, 
                loss_fn=criterion, 
                verbose=True
            )
            
        agent_holdout_0.set_model_params(agent_train.get_model_params())
        agent_holdout_1.set_model_params(agent_train.get_model_params())
        agent_holdout_2.set_model_params(agent_train.get_model_params())
        agent_holdout_3.set_model_params(agent_train.get_model_params())
        
        print(f'Updated accuracies on holdout datasets:')
        print(f"Holdout 0 (photo): {agent_holdout_0.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 1 (art_painting): {agent_holdout_1.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 2 (cartoon): {agent_holdout_2.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 3 (sketch): {agent_holdout_3.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        
        
        print(f'----------------------------------------------------------------------------')
        print(f'Holdout buffer experiments')
        print(f'----------------------------------------------------------------------------')
    
        print(f'Retraining model on photo and art dataset...')
        agent_train.set_drift_rate(1.0)
        agent_train.set_target_domains(['photo', 'art_painting'])
        agent_train.apply_drift()
        
        for _ in range(5):
            agent_train.update_steps(
                num_updates=n_steps, 
                optimizer=optimizer, 
                loss_fn=criterion, 
                verbose=True
            )
            
        agent_holdout_0.set_model_params(agent_train.get_model_params())
        agent_holdout_1.set_model_params(agent_train.get_model_params())
        agent_holdout_2.set_model_params(agent_train.get_model_params())
        agent_holdout_3.set_model_params(agent_train.get_model_params())
        
        print(f'Updated accuracies on holdout datasets:')
        print(f"Holdout 0 (photo): {agent_holdout_0.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 1 (art_painting): {agent_holdout_1.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 2 (cartoon): {agent_holdout_2.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 3 (sketch): {agent_holdout_3.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        
        print(f'Retraining model on photo and cartoon dataset...')
        agent_train.set_drift_rate(1.0)
        agent_train.set_target_domains(['photo',])
        agent_train.apply_drift()
        
        for _ in range(5):
            agent_train.update_steps(
                num_updates=n_steps, 
                optimizer=optimizer, 
                loss_fn=criterion, 
                verbose=True
            )
            
        agent_holdout_0.set_model_params(agent_train.get_model_params())
        agent_holdout_1.set_model_params(agent_train.get_model_params())
        agent_holdout_2.set_model_params(agent_train.get_model_params())
        agent_holdout_3.set_model_params(agent_train.get_model_params())
        
        print(f'Updated accuracies on holdout datasets:')
        print(f"Holdout 0 (photo): {agent_holdout_0.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 1 (art_painting): {agent_holdout_1.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 2 (cartoon): {agent_holdout_2.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 3 (sketch): {agent_holdout_3.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        
        print(f'Retraining model on photo and sketch dataset...')
        agent_train.set_drift_rate(0.8)
        agent_train.set_target_domains(['art_painting',])
        agent_train.apply_drift()
        
        for _ in range(5):
            agent_train.update_steps(
                num_updates=n_steps, 
                optimizer=optimizer, 
                loss_fn=criterion, 
                verbose=True
            )
            
        agent_holdout_0.set_model_params(agent_train.get_model_params())
        agent_holdout_1.set_model_params(agent_train.get_model_params())
        agent_holdout_2.set_model_params(agent_train.get_model_params())
        agent_holdout_3.set_model_params(agent_train.get_model_params())
        
        print(f'Updated accuracies on holdout datasets:')
        print(f"Holdout 0 (photo): {agent_holdout_0.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 1 (art_painting): {agent_holdout_1.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 2 (cartoon): {agent_holdout_2.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 3 (sketch): {agent_holdout_3.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        
        
        print(f'Retraining model on photo, art, cartoon dataset...')
        agent_train.set_drift_rate(0.8)
        agent_train.set_target_domains(['cartoon'])
        agent_train.apply_drift()
        
        for _ in range(5):
            agent_train.update_steps(
                num_updates=n_steps, 
                optimizer=optimizer, 
                loss_fn=criterion, 
                verbose=True
            )
            
        agent_holdout_0.set_model_params(agent_train.get_model_params())
        agent_holdout_1.set_model_params(agent_train.get_model_params())
        agent_holdout_2.set_model_params(agent_train.get_model_params())
        agent_holdout_3.set_model_params(agent_train.get_model_params())
        
        print(f'Updated accuracies on holdout datasets:')
        print(f"Holdout 0 (photo): {agent_holdout_0.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 1 (art_painting): {agent_holdout_1.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 2 (cartoon): {agent_holdout_2.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        print(f"Holdout 3 (sketch): {agent_holdout_3.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)}")
        
        
    return
    
    
# Hyperparameters and Constants
DSET_SIZE = 1024

# Main function 
def main():
    
    # Evaluate the policy
    evaluate_policy_under_drift(
        seed=0,
        learning_rate=0.01,
        n_steps=50,
    )
    
if __name__ == "__main__":
    main()
