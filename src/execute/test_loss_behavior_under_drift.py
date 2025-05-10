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

# Model definitions remain unchanged
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
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1),
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

# Policy class (unchanged)
class Policy:
    def __init__(self, alpha=0.01, L_i=1.0, K_p=1.0, K_d=1.0):
        self.virtual_queue = 0
        self.update_history = []
        self.last_gradient_magnitude = None
        self.alpha = alpha
        self.L_i = L_i
        self.K_p = K_p
        self.K_d = K_d
        self.loss_initial = None
        
    def update_gradient(self, model, data_loader, criterion, device):
        model.eval()
        with torch.enable_grad():
            for batch in data_loader:
                if len(batch) > 2:
                    inputs, targets, _ = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                model.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += torch.norm(param.grad).item() ** 2
                grad_norm = grad_norm ** 0.5
                self.last_gradient_magnitude = grad_norm
                break
        model.train()

    def set_initial_loss(self, loss_initial):
        self.loss_initial = loss_initial

    def policy_decision(self, decision_id, loss_curr, loss_prev, current_time, V, pi_bar, loss_best=float('inf')):
        if decision_id == 0:
            return 1
        elif decision_id == 1:
            return int(np.random.random() < pi_bar)
        elif decision_id == 2:
            interval = int(1 / pi_bar)
            return int(current_time % interval == 0)
        elif decision_id == 3:
            if current_time >= 55 and current_time <= 61 \
                or current_time >= 105 and current_time <= 112 \
                or current_time >= 155 and current_time <= 162:
                return 1
            else:
                return 0
        elif decision_id == 4:
            delta_L = loss_curr - loss_prev
            grad_magnitude = self.last_gradient_magnitude if self.last_gradient_magnitude is not None else 0.0
            left_side = V * (delta_L + self.L_i * self.alpha * grad_magnitude)
            right_side = self.virtual_queue + (0.5 - pi_bar)
            should_update = left_side >= right_side
            self.virtual_queue = max(0, self.virtual_queue + should_update - pi_bar)
            if should_update:
                self.update_history.append(current_time)
            return int(should_update)
        elif decision_id == 5:
            delta_L = loss_curr - loss_best
            grad_magnitude = self.last_gradient_magnitude if self.last_gradient_magnitude is not None else 0.0
            left_side = V * (delta_L + self.L_i * self.alpha * grad_magnitude)
            right_side = self.virtual_queue + (0.5 - pi_bar)
            should_update = left_side >= right_side
            self.virtual_queue = max(0, self.virtual_queue + should_update - pi_bar)
            if should_update:
                self.update_history.append(current_time)
            return int(should_update)
        elif decision_id == 6:
            if self.loss_initial is None:
                raise ValueError("Initial loss not set for Policy 6")
            e_t = loss_curr - loss_best
            delta_e = loss_curr - loss_prev
            pd_term = self.K_p * e_t + self.K_d * delta_e
            threshold = self.virtual_queue + 0.5 - pi_bar
            should_update = V * pd_term > threshold
            self.virtual_queue = max(0, self.virtual_queue + should_update - pi_bar)
            if should_update:
                self.update_history.append(current_time)
            return int(should_update)
        else:
            raise ValueError(f"Invalid decision_id: {decision_id}")

# DriftScheduler class (unchanged)
class DriftScheduler:
    SCHEDULE_CONFIGS = {
        "domain_change_burst_0": lambda: {
            'burst_interval': 50,
            'burst_duration': 5,
            'base_rate': 0.0,
            'burst_rate': 0.4,
            'target_domains': ['sketch', 'cartoon', 'art_painting'],
            'initial_delay': 55,
            'strategy': 'replace'
        },
        "domain_change_burst_1": lambda: {
            'burst_interval': 60,
            'burst_duration': 3,
            'base_rate': 0.0,
            'burst_rate': 0.4,
            'target_domains': ['photo', 'cartoon', 'sketch'],
            'initial_delay': 45,
            'strategy': 'replace'
        },
        "domain_change_burst_2": lambda: {
            'burst_interval': 100,
            'burst_duration': 2,
            'base_rate': 0.0,
            'burst_rate': 0.5,
            'target_domains': ['sketch', 'cartoon'],
            'initial_delay': 25,
            'strategy': 'replace'
        },
        "domain_change_burst_3": lambda: {
            'burst_interval': 50,
            'burst_duration': 5,
            'base_rate': 0.0,
            'burst_rate': 1.0,
            'target_domains': ['cartoon', 'photo', 'sketch'],
            'initial_delay': 55,
            'strategy': 'replace'
        },
        "domain_change_burst_4": lambda: {
            'burst_interval': 20,
            'burst_duration': 5,
            'base_rate': 0.0,
            'burst_rate': 1.0,
            'target_domains': ['cartoon'],
            'initial_delay': 205,
            'strategy': 'replace'
        },
        "RV_domain_change_burst_0": lambda: {
            'burst_interval': np.random.uniform(30, 70),
            'burst_duration': np.random.uniform(2, 8),
            'base_rate': 0.0,
            'burst_rate': np.random.uniform(0.1, 0.3),
            'target_domains': ['sketch', 'art_painting', 'cartoon'],
            'initial_delay': np.random.uniform(30, 70),
            'strategy': 'replace'
        },
        "RV_domain_change_burst_1": lambda: {
            'burst_interval': np.random.uniform(40, 80),
            'burst_duration': np.random.uniform(3, 6),
            'base_rate': 0.0,
            'burst_rate': np.random.uniform(0.2, 0.4),
            'target_domains': ['sketch', 'cartoon', 'art_painting'],
            'initial_delay': np.random.uniform(40, 80),
            'strategy': 'replace'
        },
        "RV_domain_change_burst_2": lambda: {
            'burst_interval': np.random.uniform(50, 90),
            'burst_duration': np.random.uniform(2, 6),
            'base_rate': 0.0,
            'burst_rate': np.random.uniform(0.3, 0.5),
            'target_domains': ['sketch', 'art_painting', 'cartoon'],
            'initial_delay': np.random.uniform(50, 90),
            'strategy': 'replace'
        },
        
        "step_0": lambda: {
            'step_points': [50, 100, 150],
            'step_rates': [0.01, 0.03, 0.05, 0.07],
            'step_domains': ['sketch', 'art_painting', 'cartoon', 'photo'],
            'strategy': 'replace'
        },
        "step_1": lambda: {
            'step_points': [40, 80, 120, 160],
            'step_rates': [0.02, 0.04, 0.06, 0.08, 0.1],
            'step_domains': ['sketch', 'cartoon', 'art_painting', 'photo', 'sketch'],
            'strategy': 'replace'
        },
        "step_2": lambda: {
            'step_points': [30, 60, 90, 120, 150],
            'step_rates': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
            'step_domains': ['art_painting', 'photo', 'sketch', 'cartoon', 'art_painting', 'photo'],
            'strategy': 'replace'
        },
        "constant_drift_domain_change": lambda: {
            'drift_rate': 0.05,
            'domain_change_interval': 50,
            'target_domains': ['sketch', 'art_painting', 'cartoon', 'photo'],
            'strategy': 'replace'
        },
        "sine_wave_domain_change": lambda: {
            'amplitude': 0.1,
            'period': 50,
            'target_domains': ['sketch', 'art_painting', 'cartoon', 'photo'],
            'strategy': 'replace'
        },
    }

    def __init__(self, schedule_type, **kwargs):
        if schedule_type not in self.SCHEDULE_CONFIGS:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        config = self.SCHEDULE_CONFIGS[schedule_type]()
        config.update(kwargs)
        for key, value in config.items():
            setattr(self, key, value)
        self.schedule_type = schedule_type
        self.current_step = 0
        self.last_burst_time = -1
        if 'target_domains' in config:
            self.current_target_domain = self.target_domains[0]
            self.domain_index = 0
            self.first_burst_completed = False

    def get_drift_rate(self, t):
        self.current_step = t
        if 'burst_interval' in self.__dict__:
            if t < self.initial_delay:
                return self.base_rate
            adjusted_t = t - self.initial_delay
            cycle_position = adjusted_t % self.burst_interval
            if cycle_position == 0 and adjusted_t >= 0:
                self.select_new_target_domain()
            return self.burst_rate if cycle_position < self.burst_duration else self.base_rate
        elif 'step_points' in self.__dict__:
            if t < self.step_points[0]:
                return 0.0
            for i, point in enumerate(self.step_points):
                if t < point:
                    return self.step_rates[i]
            return self.step_rates[-1]
        elif 'drift_rate' in self.__dict__ and 'domain_change_interval' in self.__dict__:
            return self.drift_rate
        elif 'amplitude' in self.__dict__ and 'period' in self.__dict__:
            return self.amplitude * (1 + np.sin(2 * np.pi * t / self.period)) / 2
        else:
            raise ValueError(f"Unsupported schedule type: {self.schedule_type}")

    def get_schedule_params(self):
        params = {'schedule_type': self.schedule_type, 'strategy': self.strategy}
        if 'burst_interval' in self.__dict__:
            params.update({
                'burst_interval': self.burst_interval,
                'burst_duration': self.burst_duration,
                'base_rate': self.base_rate,
                'burst_rate': self.burst_rate,
                'initial_delay': self.initial_delay,
                'target_domains': self.target_domains
            })
        elif 'step_points' in self.__dict__:
            params.update({
                'step_points': self.step_points,
                'step_rates': self.step_rates,
                'step_domains': self.step_domains
            })
        elif 'domain_change_interval' in self.__dict__:
            params.update({
                'drift_rate': self.drift_rate,
                'domain_change_interval': self.domain_change_interval,
                'target_domains': self.target_domains
            })
        elif 'amplitude' in self.__dict__:
            params.update({
                'amplitude': self.amplitude,
                'period': self.period,
                'target_domains': self.target_domains
            })
        return params

    def select_new_target_domain(self):
        if 'target_domains' not in self.__dict__:
            return None
        if not self.first_burst_completed:
            self.first_burst_completed = True
            return self.current_target_domain
        self.domain_index = (self.domain_index + 1) % len(self.target_domains)
        self.current_target_domain = self.target_domains[self.domain_index]
        return self.current_target_domain

    def get_current_target_domain(self):
        t = self.current_step
        if 'step_points' in self.__dict__:
            if t < self.step_points[0]:
                return self.step_domains[0]
            for i in range(len(self.step_points)):
                if i == len(self.step_points) - 1:
                    if t >= self.step_points[i]:
                        return self.step_domains[i + 1]
                elif t >= self.step_points[i] and t < self.step_points[i + 1]:
                    return self.step_domains[i + 1]
            return self.step_domains[-1]
        elif 'domain_change_interval' in self.__dict__:
            domain_index = (t // self.domain_change_interval) % len(self.target_domains)
            return self.target_domains[domain_index]
        elif 'period' in self.__dict__:
            domain_index = (t // self.period) % len(self.target_domains)
            return self.target_domains[domain_index]
        elif 'burst_interval' in self.__dict__:
            return self.current_target_domain
        else:
            return self.target_domains[0] if 'target_domains' in self.__dict__ else None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Modified test_policy_under_drift function
def test_policy_under_drift(
    source_domains,
    target_domains,
    model_path,
    model_architecture,
    img_size,
    seed,
    drift_scheduler,
    n_rounds,
    learning_rate=0.01,
    policy_id=0,
    setting_id=0,
    batch_size=128,
    pi_bar=0.1,
    V=1,
    L_i=1.0,
    K_p=1.0,
    K_d=1.0,
    n_steps=1
):
    drift_seed = 0
    set_seed(seed)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Should I normalize?

    policy = Policy(alpha=learning_rate, L_i=L_i, K_p=K_p, K_d=K_d)
    set_seed(drift_seed)
    data_handler = PACSDataHandler()
    data_handler.load_data()
    train_data = data_handler.train_dataset
    # test_data = data_handler.train_dataset
    train_data.transform = transform
    # test_data.transform = transform

    # Create domain-specific test loaders
    domains = ['photo', 'art_painting', 'cartoon', 'sketch']
    test_loaders = {}
    # for domain in domains:
    #     indices = [i for i, (_, _, d) in enumerate(test_data) if d == domain]
    #     subset = Subset(test_data, indices)
    #     test_loaders[domain] = DataLoader(subset, batch_size=batch_size, num_workers=4, persistent_workers=True, pin_memory=True)

    train_drift = PACSDomainDrift(
        source_domains=source_domains,
        target_domains=target_domains,
        drift_rate=0.0
    )
    test_drift = PACSDomainDrift(
        source_domains=source_domains,
        target_domains=target_domains,
        drift_rate=0.0
    )
    set_seed(seed)

    client = FederatedDriftClient(
        client_id=0,
        model_architecture=model_architecture,
        train_domain_drift=train_drift,
        test_domain_drift=test_drift
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    test_loader = None
    # Will be overwritten
    client.set_data(train_loader, train_loader)

    # Apply initial drift
    client.apply_train_drift()
    initial_train_subset_size = len(client.train_loader.dataset)
    schedule_params = drift_scheduler.get_schedule_params()
    strategy = schedule_params.get('strategy', 'add')
    if strategy == 'replace':
        client.train_domain_drift.desired_size = initial_train_subset_size
    else:
        client.train_domain_drift.desired_size = None
        
    print(f"Initial training subset size: {initial_train_subset_size}")
    
    model = client.get_model()
    if 'ResNet18' not in model_architecture.__name__:
        model.load_state_dict(torch.load(model_path))
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Initial split
    set_seed(drift_seed)
    initial_train_dataset = client.train_loader.dataset
    train_size = int(0.9 * len(initial_train_dataset))
    val_size = len(initial_train_dataset) - train_size
    train_subset, val_subset = random_split(initial_train_dataset, [train_size, val_size])
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)
    client.test_loader = val_loader
    set_seed(seed)
    # Initial loss
    loss_array = [client.get_train_metric(metric_fn=criterion, verbose=False)]
    loss_best = loss_array[0]
    policy.set_initial_loss(loss_array[0])
    
    results = []
    time_arr = []

    for t in range(n_rounds):
        time_start_round = time.time()
        current_drift_rate = drift_scheduler.get_drift_rate(t)

        if hasattr(drift_scheduler, 'target_domains'):
            current_target = drift_scheduler.get_current_target_domain()
            client.train_domain_drift.target_domains = [current_target]
            client.test_domain_drift.target_domains = [current_target]

        client.train_domain_drift.drift_rate = current_drift_rate
        if current_drift_rate > 0:
            client.apply_test_drift()
            client.apply_train_drift()

        # Split dynamically
        set_seed(drift_seed)
        current_train_dataset = client.train_loader.dataset
        train_size = int(0.9 * len(current_train_dataset))
        val_size = len(current_train_dataset) - train_size
        train_subset, val_subset = random_split(current_train_dataset, [train_size, val_size])
        set_seed(seed)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        client.test_loader = val_loader  # Use validation loader for evaluation
        current_loss = client.get_train_metric(metric_fn=criterion, verbose=False)

        # Evaluate each domain separately
        # domain_accuracies = {}
        # domain_losses = {}
        # original_test_loader = client.test_loader
        # for domain in domains:
        #     client.test_loader = test_loaders[domain]
        #     domain_accuracies[domain] = client.evaluate(metric_fn=accuracy_fn, verbose=False)
        #     domain_losses[domain] = client.get_train_metric(metric_fn=criterion, verbose=False)
        #     client
        # client.test_loader = original_test_loader  # Restore original test loader
        loss_prev = loss_array[-1]

        decision = policy.policy_decision(
            decision_id=policy_id,
            loss_curr=current_loss,
            loss_prev=loss_prev,
            current_time=t,
            V=V,
            pi_bar=pi_bar,
            loss_best=loss_best
        )
        # if policy_id == 3 and (t in [55, 105, 155]):
        #     print(f"Reset Weights")
        #     client.randomize_weights()

        if decision:
            update_loss = client.update_steps(
                num_updates=n_steps,
                optimizer=optimizer,
                loss_fn=criterion,
                verbose=True
            )
            policy.update_gradient(model, train_loader, criterion, device)
                
        loss_array.append(current_loss)
        current_accuracy = client.evaluate(metric_fn=accuracy_fn, verbose=False)

        if current_loss < loss_best:
            loss_best = current_loss
        result_dict = {
            't': t,
            'current_accuracy': current_accuracy,
            'current_loss': current_loss,
            # 'domain_accuracies': domain_accuracies,
            # 'domain_losses': domain_losses,
            'decision': decision,
            'drift_rate': current_drift_rate
        }
        if hasattr(drift_scheduler, 'target_domains'):
            result_dict['target_domain'] = drift_scheduler.get_current_target_domain()

        results.append(result_dict)
        time_arr.append(time.time() - time_start_round)
        print(f'Round {t} took {time_arr[-1]} seconds')
        print(f"    Round {t}: Decision = {decision}, Drift = {current_drift_rate:.4f}")
        print(f"    Current accuracy: {current_accuracy:.4f}, Loss: {current_loss:.4f}")
        # for domain in domains:
        #     print(f"  Domain {domain}: Accuracy = {domain_accuracies[domain]:.4f}, Loss = {domain_losses[domain]:.4f}")
        if hasattr(drift_scheduler, 'target_domains'):
            print(f"Current target domain: {drift_scheduler.get_current_target_domain()}")

    schedule_params = drift_scheduler.get_schedule_params()
    schedule_type = schedule_params['schedule_type']
    src_domains_str = "_".join(source_domains)
    tgt_domains_str = "_".join(target_domains)
    results_filename = (
        f"../../data/results/policy_{policy_id}_setting_{setting_id}_schedule_{schedule_type}"
        f"_src_{src_domains_str}_model_{model_architecture.__name__}_img_size_{img_size}_seed_{seed}.json"
    )

    data = {
        'parameters': {
            'source_domains': source_domains,
            'target_domains': target_domains,
            'model_path': model_path,
            "model_architecture": model_architecture.__name__,
            'seed': seed,
            'n_rounds': n_rounds,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'policy_id': policy_id,
            'setting_id': setting_id,
            'pi_bar': pi_bar,
            'V': V,
            'K_p': K_p,
            'K_d': K_d
        },
        'schedule_params': schedule_params,
        'results': results
    }

    with open(results_filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Average time per round: {np.mean(time_arr)}")
    print(f"Total time: {np.sum(time_arr)}")
    return results

def main():
    parser = argparse.ArgumentParser(description="PACS CNN Evaluation with Dynamic Drift")
    parser.add_argument('--schedule_type', type=str, default='domain_change_burst_0',
                        choices=list(DriftScheduler.SCHEDULE_CONFIGS.keys()),
                        help='Type of drift rate schedule')
    settings = {
        0: {'pi_bar': 0.03, 'V': 65, 'L_i': 1.0},
        1: {'pi_bar': 0.05, 'V': 65, 'L_i': 1.0},
        2: {'pi_bar': 0.1, 'V': 65, 'L_i': 1.0},
        3: {'pi_bar': 0.15, 'V': 65, 'L_i': 1.0},
        4: {'pi_bar': 0.2, 'V': 65, 'L_i': 1.0},
        5: {'pi_bar': 0.25, 'V': 65, 'L_i': 1.0},
        6: {'pi_bar': 0.3, 'V': 65, 'L_i': 1.0},
        7: {'pi_bar': 0.5, 'V': 1, 'L_i': 1.0},
        8: {'pi_bar': 0.7, 'V': 10, 'L_i': 1.0},
        9: {'pi_bar': 0.9, 'V': 100, 'L_i': 1.0},
        10: {'pi_bar': 0.1, 'V': 1, 'L_i': 10.0},
        11: {'pi_bar': 0.1, 'V': 10, 'L_i': 10.0},
        12: {'pi_bar': 0.1, 'V': 100, 'L_i': 10.0},
        13: {'pi_bar': 0.1, 'V': 1, 'L_i': 100.0},
        14: {'pi_bar': 0.1, 'V': 10, 'L_i': 100.0},
        15: {'pi_bar': 0.1, 'V': 100, 'L_i': 100.0}, 
        16: {'pi_bar': 0.1, 'V': 0.1, 'L_i': 1}, 
        17: {'pi_bar': 0.1, 'V': 0.1, 'L_i': 0.1},
        18: {'pi_bar': 0.1, 'V': 1, 'L_i': 0.1},
        19: {'pi_bar': 0.1, 'V': 1, 'L_i': 1000.0},
        20: {'pi_bar': 0.1, 'V': 1000, 'L_i': 1},
        21: {'pi_bar': 0.1, 'V': 1000, 'L_i': 1000},
        22: {'pi_bar': 0.1, 'V': 10000, 'L_i': 0.1},
        23: {'pi_bar': 0.1, 'V': 1000, 'L_i': 0.1},
        24: {'pi_bar': 0.1, 'V': 100, 'L_i': 0.1},
        25: {'pi_bar': 0.1, 'V': 10, 'L_i': 0.1},
        26: {'pi_bar': 0.1, 'V': 10000, 'L_i': 0.01},
        27: {'pi_bar': 0.1, 'V': 1000, 'L_i': 0.01},
        28: {'pi_bar': 0.1, 'V': 100, 'L_i': 0.01},
        29: {'pi_bar': 0.1, 'V': 10, 'L_i': 0.01},
        30: {'pi_bar': 0.1, 'V': 100, 'L_i': 0.0001},
        31: {'pi_bar': 0.1, 'V': 50, 'L_i': 0.0001},
        32: {'pi_bar': 0.1, 'V': 10, 'L_i': 0.0001},
        33: {'pi_bar': 0.1, 'V': 1, 'L_i': 0.0001},
        34: {'pi_bar': 0.1, 'V': 0.5, 'L_i': 0.0001},
        35: {'pi_bar': 0.1, 'V': 100, 'L_i': 0},
        36: {'pi_bar': 0.1, 'V': 50, 'L_i': 0},
        37: {'pi_bar': 0.1, 'V': 10, 'L_i': 0},
        38: {'pi_bar': 0.1, 'V': 1, 'L_i': 0},
        39: {'pi_bar': 0.1, 'V': 0.5, 'L_i': 0},
        40: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 1.0, 'K_d': 1.0},
        41: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 2.0, 'K_d': 2.0},
        42: {'pi_bar': 0.1, 'V': 1, 'L_i': 0, 'K_p': 1.0, 'K_d': 1.0},
        43: {'pi_bar': 0.1, 'V': 1, 'L_i': 0, 'K_p': 2.0, 'K_d': 2.0},
        44: {'pi_bar': 0.1, 'V': 0.1, 'L_i': 0, 'K_p': 1.0, 'K_d': 1.0},
        45: {'pi_bar': 0.1, 'V': 0.1, 'L_i': 0, 'K_p': 2.0, 'K_d': 2.0},
        46: {'pi_bar': 0.1, 'V': 0.1, 'L_i': 0, 'K_p': 5.0, 'K_d': 5.0},
        47: {'pi_bar': 0.1, 'V': 0.1, 'L_i': 0, 'K_p': 10.0, 'K_d': 10.0},
        48: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 2.0, 'K_d': 0.5},
        49: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0},
        50: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.25, 'K_d': 2.0},
        51: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.10, 'K_d': 2.0},
        52: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 1.0},
        53: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.25, 'K_d': 1.0},
        54: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.10, 'K_d': 1.0},
        55: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 0.5},
        56: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.25, 'K_d': 0.5},
        57: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.10, 'K_d': 0.5},
        60: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':1},
        61: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':2},
        62: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':3},
        63: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.05, 'n_steps':1},
        64: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.05, 'n_steps':2},
        65: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.05, 'n_steps':3},
        66: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.1, 'n_steps':1},
        67: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.1, 'n_steps':2},
        68: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.1, 'n_steps':3},
    }
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--src_domains', type=str, nargs='+', default=['photo'])
    parser.add_argument('--tgt_domains', type=str, nargs='+', default=['sketch'])
    parser.add_argument('--n_rounds', type=int, default=200)
    # parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--policy_id', type=int, default=4)
    parser.add_argument('--setting_id', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='PACSCNN_4', choices=['PACSCNN_1', 'PACSCNN_2', 'PACSCNN_3', 'PACSCNN_4'], help='Model architecture to use')
    parser.add_argument('--img_size', type=int, default=128, help='Size to resize images to (img_size x img_size)')

    model_architectures = {
        'PACSCNN_1': PACSCNN_1,
        'PACSCNN_2': PACSCNN_2,
        'PACSCNN_3': PACSCNN_3,
        'PACSCNN_4': PACSCNN_4,
    }
    
    args = parser.parse_args()
    print(f'Model used: {args.model_name}')
    drift_scheduler = DriftScheduler(args.schedule_type)
    cluster_used = 'gautschi'
    if args.seed == 42:
        model_path = f"/scratch/{cluster_used}/apiasecz/models/concept_drift_models/{args.model_name}_{'_'.join(args.src_domains)}_img_size_{args.img_size}_seed_0.pth"
    else:
        model_path = f"../../../../models/concept_drift_models/{args.model_name}_{'_'.join(args.src_domains)}_img_size_{args.img_size}_seed_{args.seed}.pth"
    current_settings = settings[args.setting_id]
    K_p = current_settings.get('K_p', 1.0)
    K_d = current_settings.get('K_d', 1.0)
    lr = current_settings.get('lr', 0.01)
    num_steps = current_settings.get('n_steps', 1)
    results = test_policy_under_drift(
        source_domains=args.src_domains,
        target_domains=args.tgt_domains,
        model_path=model_path,
        model_architecture=model_architectures[args.model_name],
        img_size=args.img_size,
        seed=args.seed,
        drift_scheduler=drift_scheduler,
        n_rounds=args.n_rounds,
        learning_rate=lr,
        policy_id=args.policy_id,
        setting_id=args.setting_id,
        pi_bar=current_settings['pi_bar'],
        V=current_settings['V'],
        L_i=current_settings['L_i'],
        K_p=K_p,
        K_d=K_d, 
        n_steps=num_steps,
    )

if __name__ == "__main__":
    main()
    
    # For the test set - fix the partitioning/or partition the data with the same seed.
     