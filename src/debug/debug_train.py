import os
import time
start_time = time.time()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.utils import save_image
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

seed = 42
set_seed(seed)

# Test PACS data hanlder
data_handler = PACSDataHandler(transform=transform_test)
print(f"Domains: {data_handler.domains}")

photo_domain = data_handler.get_domain_data('photo')
art_painting_domain = data_handler.get_domain_data('art_painting')
cartoon_domain = data_handler.get_domain_data('cartoon')
sketch_domain = data_handler.get_domain_data('sketch')

# Test different domains
# get_batch_img(4, 'photo')
# get_batch_img(4, 'art_painting')
# get_batch_img(4, 'cartoon')
# get_batch_img(4, 'sketch')

pacs_drift = PACSDomainDrift(
    data_handler,
    source_domains=['photo',],
    target_domains=['sketch',],
    drift_rate=0.1,
    desired_size=1600
)

print(f'Took {time.time() - start_time:.2f} seconds to run the script.')