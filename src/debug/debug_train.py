import os
import time
start_time = time.time()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.utils import save_image, make_grid
import numpy as np
import random
from torchvision import transforms
from fl_toolkit import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

img_size = 128
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=img_size, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_batch_img(num_img, domain, directory='./test_imgs'):
    os.makedirs(directory, exist_ok=True)
    domain_data = data_handler.get_domain_data(domain)
    for i in range(num_img):
        sample = domain_data[i]
        image_tensor = sample[0]
        label = sample[1]
        filename = f'test_img_{domain}_{i}_{data_handler.categories[label]}.png'
        filepath = os.path.join(directory, filename)
        save_image(image_tensor, filepath)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_dataset_grid(dataset, filename, nrow=4):
    images = [dataset[i][0] for i in range(min(16, len(dataset)))]
    images = torch.stack(images)
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    grid = make_grid(images, nrow=nrow, padding=2)
    save_image(grid, filename)

def print_domain_counts(dataset):
    domains = [sample[2] for sample in dataset]
    unique, counts = np.unique(domains, return_counts=True)
    print(dict(zip(unique, counts)))

# Initialize data handler and domain drift
data_handler = PACSDataHandler(transform=transform_train)
print(f"Domains: {data_handler.domains}")

# Domain drift object
pacs_drift = PACSDomainDrift(
    data_handler,
    source_domains=['photo'],
    target_domains=['sketch'],
    drift_rate=0.25,
    desired_size=160
)

# DriftAgent object
agent = DriftAgent(
    client_id=0,
    model_architecture=None,  # Replace with actual model class if needed
    domain_drift=pacs_drift,
    batch_size=32,
    device=device
)

# Initial dataset setup
# Set initial dataset (source domains)
print(f"Initial dataset size: {len(agent.current_dataset)}")
train_dataset = agent.current_dataset
test_loader = agent.get_test_loader(test_size=0.1, test_batch_size=32)
test_subset = agent.test_subset
save_dataset_grid(train_dataset, os.path.join('./test_imgs', 'train_grid_0.png'), nrow=4)
save_dataset_grid(test_subset, os.path.join('./test_imgs', 'test_grid_0.png'), nrow=4)
print("Initial training dataset domains:")
print_domain_counts(train_dataset)
print("Initial testing dataset domains:")
print_domain_counts(test_subset)

# Drift application loop
target_sequence = ['sketch', 'art_painting', 'cartoon', 'photo']
counter = 1
for target in target_sequence:
    agent.set_target_domains([target])
    for j in range(4):
        agent.apply_drift()
        train_dataset = agent.current_dataset
        test_loader = agent.get_test_loader(test_size=0.1, test_batch_size=32)
        test_subset = agent.test_subset
        save_dataset_grid(train_dataset, os.path.join('./test_imgs', f'train_grid_{counter}.png'), nrow=4)
        save_dataset_grid(test_subset, os.path.join('./test_imgs', f'test_grid_{counter}.png'), nrow=4)
        print(f"After drift {counter} with target domain: {target}")
        print("Training dataset domains:")
        print_domain_counts(train_dataset)
        print("Testing dataset domains:")
        print_domain_counts(test_subset)
        counter += 1

print(f'Took {time.time() - start_time:.2f} seconds to run the script.')