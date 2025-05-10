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
print(f"PyTorch Version: {torch.__version__}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)

data_handler = PACSDataHandler()
print(data_handler.get_dataset_info())

domains = data_handler.domains
print(data_handler.dataset[0][0].shape, type(data_handler.dataset[0][0]))
print(data_handler.dataset[0][1])
print(data_handler.dataset[0][2])

photo_domain_data = data_handler.get_domain_data('photo')
sketch_domain_data = data_handler.get_domain_data('sketch')
art_painting_domain_data = data_handler.get_domain_data('art_painting')
cartoon_domain_data = data_handler.get_domain_data('cartoon')

print(f"Photo domain data: {len(photo_domain_data)} samples")
print(f"Sketch domain data: {len(sketch_domain_data)} samples")
print(f"Art painting domain data: {len(art_painting_domain_data)} samples")
print(f"Cartoon domain data: {len(cartoon_domain_data)} samples")

