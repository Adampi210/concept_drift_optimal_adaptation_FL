import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import csv
import argparse
import time


from fl_toolkit import *  # Ensure fl_toolkit is correctly installed and accessible

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"cuDNN Version: {torch.backends.cudnn.version()}")
print(f"PyTorch Version: {torch.__version__}")

# =========================
# Models
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
# Policies
# =========================
class Policy:
    def __init__(self, alpha=0.01, L_i = 1):
        """
        Initialize the Policy class.

        Args:
            alpha (float): Learning rate or step size.
        """
        self.virtual_queue = 0  # Virtual queue Q(t)
        self.update_history = []  # History of update times
        self.last_gradient_magnitude = None  # Store the last gradient magnitude
        self.alpha = alpha  # Learning rate
        self.L_i = L_i # Lipschitz constant defined internally (default value)

    def update_gradient(self, model, data_loader, criterion, device):
        """
        Compute and store the gradient magnitude after a model update.

        Args:
            model: The trained model.
            data_loader: DataLoader for the training data.
            criterion: Loss function.
            device: Device to use (e.g., 'cpu' or 'cuda').
        """
        model.eval()
        with torch.enable_grad():
            for batch in data_loader:
                if len(batch) > 2:
                    inputs, targets, _ = batch  # Ignore rest
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
                break  # Use one batch for efficiency
        model.train()

    def policy_decision(self, decision_id, loss_curr, loss_prev, current_time, V, pi_bar):
        """
        Decide whether to update the model based on the selected policy.

        Args:
            decision_id (int): ID of the policy to use (0-4).
            loss_curr (float): Current loss L(θ_t; D_{t+1}).
            loss_prev (float): Previous loss L(θ_{t-1}; D_t).
            current_time (int): Current time step.
            V (float): Trade-off parameter.
            pi_bar (float): Average update rate (λ̄).

        Returns:
            int: 1 if update, 0 if not.
        """
        if decision_id == 0:
            # Policy 0: Never update
            return 0
        elif decision_id == 1:
            # Policy 1: Random update with probability pi_bar
            return int(np.random.random() < pi_bar)
        elif decision_id == 2:
            # Policy 2: Update at fixed intervals (1/pi_bar)
            interval = int(1 / pi_bar)
            return int(current_time % interval == 0)
        elif decision_id == 3:
            # Policy 3: Virtual queue-based update
            delta_L = loss_curr - loss_prev
            threshold = self.virtual_queue + 0.5 - pi_bar
            should_update = V * delta_L > threshold
            self.virtual_queue = max(0, self.virtual_queue + should_update - pi_bar)
            if should_update:
                self.update_history.append(current_time)
            return int(should_update)
        elif decision_id == 4:
            # Policy 4: New policy with gradient magnitude and internal L_i
            delta_L = loss_curr - loss_prev
            grad_magnitude = self.last_gradient_magnitude if self.last_gradient_magnitude is not None else 0.0
            left_side = V * (delta_L + self.L_i * self.alpha * grad_magnitude)
            right_side = self.virtual_queue + (pi_bar - 0.5)
            should_update = left_side >= right_side
            self.virtual_queue = max(0, self.virtual_queue + should_update - pi_bar)
            if should_update:
                self.update_history.append(current_time)
            return int(should_update)
        else:
            raise ValueError("Invalid decision_id")

# =========================
# Drift Scheduling
# =========================
class DriftScheduler:
    # Class-level configuration dictionary for all schedule types
    SCHEDULE_CONFIGS = {
        "burst_0": lambda: {
            'burst_interval': 50,
            'burst_duration': 5,
            'base_rate': 0.0,
            'burst_rate': 0.2,
            'initial_delay': 50  # Add initial delay
        },
        "burst_1": lambda: {
            'burst_interval': 80,
            'burst_duration': 5,
            'base_rate': 0.0,
            'burst_rate': 0.4,
            'initial_delay': 80  # Add initial delay
        },
        "RV_burst_0": lambda: {
            'burst_interval': np.random.uniform(30, 70),
            'burst_duration': np.random.uniform(1, 9),
            'base_rate': 0.0,
            'burst_rate': np.random.uniform(0.1, 0.3),
            'initial_delay': np.random.uniform(30, 70)  # Random initial delay
        },
        "RV_burst_1": lambda: {
            'burst_interval': np.random.uniform(80, 120),
            'burst_duration': np.random.uniform(1, 9),
            'base_rate': 0.0,
            'burst_rate': np.random.uniform(0.4, 0.7),
            'initial_delay': np.random.uniform(80, 120)  # Random initial delay
        },
        "domain_change_burst_0": lambda: {
            'burst_interval': 50,
            'burst_duration': 5,
            'base_rate': 0.0,
            'burst_rate': 0.2,
            'target_domains': ['sketch', 'art_painting', 'cartoon'],
            'initial_delay': 50  # Add initial delay
        },
        "domain_change_burst_1": lambda: {
            'burst_interval': 50,
            'burst_duration': 5,
            'base_rate': 0.0,
            'burst_rate': 0.2,
            'target_domains': ['sketch', 'cartoon', 'art_painting'],
            'initial_delay': 50  # Add initial delay
        },
        "domain_change_burst_2": lambda: {
            'burst_interval': 50,
            'burst_duration': 5,
            'base_rate': 0.0,
            'burst_rate': 0.2,
            'target_domains': ['sketch', 'cartoon', 'photo'],
            'initial_delay': 50  # Add initial delay
        },
        "oscillating_0": lambda: {
            'high_duration': 20,
            'low_duration': 20,
            'high_rate': 0.05,
            'low_rate': 0.01
        },
        "oscillating_1": lambda: {
            'high_duration': 40,
            'low_duration': 40,
            'high_rate': 0.04,
            'low_rate': 0.01
        },
        "oscillating_increasing_0": lambda: {
            'alpha': 0.01,
            'period': 20
        },
        "step_0": lambda: {
            'step_points': [50, 100, 150],
            'step_rates': [0.01, 0.03, 0.05, 0.07]
        },
        "custom": lambda: {
            'custom_schedule': {i: 0.1 for i in range(0, 200, 50)}
        }
    }

    def __init__(self, schedule_type, **kwargs):
        """
        Initialize the DriftScheduler with a specified schedule type.

        Args:
            schedule_type (str): The type of schedule to use (key in SCHEDULE_CONFIGS).
            **kwargs: Optional overrides for configuration parameters.
        """
        if schedule_type not in self.SCHEDULE_CONFIGS:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Get the configuration and allow overrides via kwargs
        config = self.SCHEDULE_CONFIGS[schedule_type]()
        config.update(kwargs)

        # Set all configuration parameters as instance attributes
        for key, value in config.items():
            setattr(self, key, value)

        self.schedule_type = schedule_type
        self.current_step = 0
        self.last_burst_time = -1  # Track last burst for domain changes

        # Initialize domain-related attributes if applicable
        if 'target_domains' in config:
            self.current_target_domain = self.target_domains[0]
            self.domain_index = 0
            self.first_burst_completed = False

    def get_drift_rate(self, t):
        """
        Calculate the drift rate at time t based on the schedule type.

        Args:
            t (int): Current timestep.

        Returns:
            float: The drift rate at time t.
        """
        self.current_step = t

        # Burst schedules (including RV and domain-changing bursts)
        if 'burst_interval' in self.__dict__:
            # If within the initial delay period, return base_rate
            if t < self.initial_delay:
                return self.base_rate
            # Adjust time to account for the initial delay
            adjusted_t = t - self.initial_delay
            cycle_position = adjusted_t % self.burst_interval
            # Handle domain changes at the end of a burst
            if 'target_domains' in self.__dict__ and cycle_position == self.burst_duration and adjusted_t > 0 and t != self.last_burst_time:
                self.select_new_target_domain()
                self.last_burst_time = t
            return self.burst_rate if cycle_position < self.burst_duration else self.base_rate

        # Oscillating schedules (fixed high/low periods)
        elif 'high_duration' in self.__dict__:
            cycle_length = self.high_duration + self.low_duration
            cycle_position = t % cycle_length
            return self.high_rate if cycle_position < self.high_duration else self.low_rate

        # Oscillating increasing schedules (sinusoidal with increasing amplitude)
        elif 'alpha' in self.__dict__:
            oscillation = 0.5 * np.sin(2 * np.pi * t / self.period) + 0.5
            rate = self.alpha * t * oscillation
            return min(max(rate, 0.0), 1.0)

        # Step schedules
        elif 'step_points' in self.__dict__:
            for point, rate in zip(self.step_points + [float('inf')], self.step_rates):
                if t < point:
                    return rate
            return self.step_rates[-1]

        # Custom schedules (dictionary-based)
        elif 'custom_schedule' in self.__dict__:
            return self.custom_schedule.get(t, 0.0)

        else:
            raise ValueError(f"Unsupported schedule type: {self.schedule_type}")

    def get_schedule_params(self):
        """
        Return the current schedule parameters for logging or saving.

        Returns:
            dict: Parameters defining the current schedule.
        """
        params = {'schedule_type': self.schedule_type}

        if 'burst_interval' in self.__dict__:
            params.update({
                'burst_interval': self.burst_interval,
                'burst_duration': self.burst_duration,
                'base_rate': self.base_rate,
                'burst_rate': self.burst_rate,
                'initial_delay': self.initial_delay  # Include initial_delay in params
            })
            if 'target_domains' in self.__dict__:
                params['target_domains'] = self.target_domains
        elif 'high_duration' in self.__dict__:
            params.update({
                'high_duration': self.high_duration,
                'low_duration': self.low_duration,
                'high_rate': self.high_rate,
                'low_rate': self.low_rate
            })
        elif 'alpha' in self.__dict__:
            params.update({
                'alpha': self.alpha,
                'period': self.period
            })
        elif 'step_points' in self.__dict__:
            params.update({
                'step_points': self.step_points,
                'step_rates': self.step_rates
            })
        elif 'custom_schedule' in self.__dict__:
            params['custom_schedule'] = self.custom_schedule

        return params

    def select_new_target_domain(self):
        """
        Select the next target domain for domain-changing schedules.

        Returns:
            str: The new current target domain, or None if not applicable.
        """
        if 'target_domains' not in self.__dict__:
            return None

        if not self.first_burst_completed:
            self.first_burst_completed = True
            return self.current_target_domain

        self.domain_index = (self.domain_index + 1) % len(self.target_domains)
        self.current_target_domain = self.target_domains[self.domain_index]
        return self.current_target_domain

    def get_current_target_domain(self):
        """
        Get the current target domain.

        Returns:
            str or None: The current target domain, or None if not applicable.
        """
        return self.current_target_domain if 'target_domains' in self.__dict__ else None

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
  
def test_policy_under_drift(
    source_domains,
    target_domains,
    model_path,
    model_architecture,
    seed,
    drift_scheduler,
    n_rounds,
    learning_rate=0.01,
    policy_id=0,
    setting_id=0,
    batch_size=128,
    pi_bar=0.1,
    V=1, 
    L_i=1.0
    ):
    """
    Modify retraining with policy under drift using the new DriftScheduler.

    Args:
        source_domains (list): List of source domain names.
        target_domains (list): List of target domain names.
        model_path (str): Path to the pretrained model.
        seed (int): Random seed for reproducibility.
        drift_scheduler (DriftScheduler): Instance of DriftScheduler managing drift rates and domains.
        n_rounds (int): Number of training rounds.
        learning_rate (float): Learning rate for the optimizer.
        policy_id (int): Identifier for the policy.
        setting_id (int): Identifier for the setting.
        batch_size (int): Batch size for data loaders.
        pi_bar (float): Policy threshold parameter.
        V (int): Policy parameter for decision-making.

    Returns:
        list: List of result dictionaries for each round.
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Initialize policy and data handler
    policy = Policy(alpha=learning_rate, L_i = L_i)
    data_handler = PACSDataHandler()
    data_handler.load_data()
    train_data, test_data = data_handler.train_dataset, data_handler.test_dataset

    # Initialize domain drifts
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

    # Initialize client
    client = FederatedDriftClient(
        client_id=0,
        model_architecture=model_architecture,
        train_domain_drift=train_drift,
        test_domain_drift=test_drift
    )
    print(f'Client device: {client.device}')

    # Set up data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    client.set_data(train_loader, test_loader)

    # Load pretrained model
    model = client.get_model()
    model.load_state_dict(torch.load(model_path))

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Track loss and results
    loss_array = [client.get_train_metric(metric_fn=criterion, verbose=False)]
    results = []
    time_arr = []
    # Training loop
    for t in range(n_rounds):
        time_start_round = time.time()
        # Get current drift rate from scheduler
        current_drift_rate = drift_scheduler.get_drift_rate(t)

        # Update target domains if the schedule involves domain changes
        if hasattr(drift_scheduler, 'target_domains'):
            current_target = drift_scheduler.get_current_target_domain()
            client.train_domain_drift.target_domains = [current_target]
            client.test_domain_drift.target_domains = [current_target]

        # Apply drift rates
        client.train_domain_drift.drift_rate = current_drift_rate
        client.test_domain_drift.drift_rate = current_drift_rate
        if current_drift_rate > 0:
            client.apply_test_drift()
            client.apply_train_drift()

        # Evaluate and compute loss
        current_accuracy = client.evaluate(metric_fn=accuracy_fn, verbose=False)
        current_loss = client.get_train_metric(metric_fn=criterion, verbose=False)
        loss_array.append(current_loss)

        # Make retraining decision
        decision = policy.policy_decision(
            decision_id=policy_id,  # Or 0, 1, 2, 3 as needed
            loss_curr=loss_array[-1],
            loss_prev=loss_array[-2],
            current_time=t,
            V=1.0,
            pi_bar=0.1
        )

        # Train or evaluate based on decision
        if decision:
            current_loss = client.train(
                epochs=2,
                optimizer=optimizer,
                loss_fn=criterion,
                verbose=True
            )
            policy.update_gradient(model, train_loader, criterion, device)
        else:
            current_loss = client.get_train_metric(metric_fn=criterion, verbose=False)

        # Collect results
        result_dict = {
            't': t,
            'accuracy': current_accuracy,
            'loss': current_loss,
            'decision': decision,
            'drift_rate': current_drift_rate
        }
        if hasattr(drift_scheduler, 'target_domains'):
            result_dict['target_domain'] = drift_scheduler.get_current_target_domain()

        results.append(result_dict)
        time_arr.append(time.time() - time_start_round)
        print(f'Round {t} took {time_arr[-1]} seconds')
        # Print progress
        print(f"Round {t}: Accuracy = {current_accuracy:.4f}, Decision = {decision}, Drift = {current_drift_rate:.4f}")
        if hasattr(drift_scheduler, 'target_domains'):
            print(f"Current target domain: {drift_scheduler.get_current_target_domain()}")

    # Save results as JSON
    schedule_params = drift_scheduler.get_schedule_params()
    schedule_type = schedule_params['schedule_type']
    src_domains_str = "_".join(source_domains)
    tgt_domains_str = "_".join(target_domains)
    results_filename = (
        f"../../data/results/policy_{policy_id}_setting_{setting_id}_schedule_{schedule_type}"
        f"_src_{src_domains_str}_tgt_{tgt_domains_str}_seed_{seed}.json"
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
            'V': V
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
    
    # Update schedule type choices to reflect DriftScheduler's capabilities
    parser.add_argument('--schedule_type', type=str, default='domain_change_burst_0',
                        choices=list(DriftScheduler.SCHEDULE_CONFIGS.keys()),
                        help='Type of drift rate schedule')

    # Settings dictionary (unchanged)
    settings = {
        0: {'pi_bar': 0.03, 'V': 65, 'L_i': 1.0},
        1: {'pi_bar': 0.05, 'V': 65, 'L_i': 1.0},
        2: {'pi_bar': 0.1, 'V': 65, 'L_i': 1.0},
        3: {'pi_bar': 0.15, 'V': 65, 'L_i': 1.0},
        4: {'pi_bar': 0.2, 'V': 65, 'L_i': 1.0},
        5: {'pi_bar': 0.25, 'V': 65, 'L_i': 1.0},
        6: {'pi_bar': 0.3, 'V': 65, 'L_i': 1.0},
        7: {'pi_bar': 0.1, 'V': 1, 'L_i': 1.0},
        8: {'pi_bar': 0.1, 'V': 10, 'L_i': 1.0},
        9: {'pi_bar': 0.1, 'V': 100, 'L_i': 1.0},
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
        22: {{'pi_bar': 0.1, 'V': 10000, 'L_i': 0.1}},
        23: {{'pi_bar': 0.1, 'V': 1000, 'L_i': 0.1}},
        24: {{'pi_bar': 0.1, 'V': 100, 'L_i': 0.1}},
        25: {{'pi_bar': 0.1, 'V': 10, 'L_i': 0.1}},
        26: {{'pi_bar': 0.1, 'V': 10000, 'L_i': 0.01}},
        27: {{'pi_bar': 0.1, 'V': 1000, 'L_i': 0.01}},
        28: {{'pi_bar': 0.1, 'V': 100, 'L_i': 0.01}},
        29: {{'pi_bar': 0.1, 'V': 10, 'L_i': 0.01}},
    }
    
    # Existing arguments
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--src_domains', type=str, nargs='+', default=['photo'])
    parser.add_argument('--tgt_domains', type=str, nargs='+', default=['sketch'])
    parser.add_argument('--n_rounds', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--policy_id', type=int, default=4)
    parser.add_argument('--setting_id', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='PACSCNN_3', choices=['PACSCNN', 'PACSCNN_1', 'PACSCNN_2', 'PACSCNN_3', 'PACSCNN_4'], help='Model architecture to use')
    
    model_architectures = {
        'PACSCNN': PACSCNN,
        'PACSCNN_1': PACSCNN_1,
        'PACSCNN_2': PACSCNN_2,
        'PACSCNN_3': PACSCNN_3,
        'PACSCNN_4': PACSCNN_4
    }
    
    args = parser.parse_args()
    
    print(f'Model used: {args.model_name}')
    
    # Initialize DriftScheduler
    drift_scheduler = DriftScheduler(args.schedule_type)

    # Construct model path
    domains_str = "_".join(args.src_domains)
    ###### DELETE THIS LINE AFTER TESTING
    if args.seed == 42:
        model_path = f"/scratch/gilbreth/apiasecz/models/concept_drift_models/{args.model_name}_{domains_str}_seed_0.pth"
    else:
        model_path = f"/scratch/gilbreth/apiasecz/models/concept_drift_models/{args.model_name}_{domains_str}_seed_{args.seed}.pth"

    # Get settings
    current_settings = settings[args.setting_id]

    # Run evaluation
    results = test_policy_under_drift(
        source_domains=args.src_domains,
        target_domains=args.tgt_domains,
        model_path=model_path,
        model_architecture=model_architectures[args.model_name],
        seed=args.seed,
        drift_scheduler=drift_scheduler,
        n_rounds=args.n_rounds,
        learning_rate=args.lr,
        policy_id=args.policy_id,
        setting_id=args.setting_id,
        pi_bar=current_settings['pi_bar'],
        V=current_settings['V'], 
        L_i=current_settings['L_i']
    )

if __name__ == "__main__":
    main()