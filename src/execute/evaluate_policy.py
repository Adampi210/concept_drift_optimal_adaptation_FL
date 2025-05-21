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

# Policy
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
        self.consecutive_increases = 0
        self.loss_window = []  # For local consecutive increases
        self.tokens = 0.0      # Budget term
        self.loss_best = float('inf')
        
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
            w = 40  # Window size
            m = 3   # Consecutive increases
            self.loss_window.append(loss_curr)
            if len(self.loss_window) > w:
                self.loss_window.pop(0)
            # Check for m consecutive increases ending at the current time
            increases = 0
            for i in range(len(self.loss_window) - 1, 0, -1):  # Start from end, move backward
                if self.loss_window[i] > self.loss_window[i - 1]:
                    increases += 1
                else:
                    break  # Stop at the first non-increase
            if increases >= m and self.tokens >= 1.0:
                self.tokens -= 1.0
                return 1
            self.tokens += pi_bar
            return 0
        elif decision_id == 4:
            delta = 0.1  # 10% threshold
            w = 40
            max_loss = max(self.loss_window) if self.loss_window else loss_curr
            if loss_curr > max_loss * (1 + delta) and self.tokens >= 1.0:
                self.tokens -= 1.0
                return 1
            self.tokens += pi_bar
            self.loss_window.append(loss_curr)
            if len(self.loss_window) > w:
                self.loss_window.pop(0)
            return 0
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

# Drift Scheduler
class DriftScheduler:
    SCHEDULE_CONFIGS = {
        "domain_change_burst_0": lambda: {
            'burst_interval': 100,
            'burst_duration': 5,
            'base_rate': 0.0,
            'burst_rate': 0.4,
            'target_domains': ['sketch', 'cartoon', 'art_painting'],
            'initial_delay': 55,
            'strategy': 'replace'
        },
        "domain_change_burst_1": lambda: {
            'burst_interval': 120,
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
            'burst_interval_limits': (30, 70),
            'burst_duration_limits': (4, 7),
            'base_rate': 0.0,
            'burst_rate': (0.2, 0.5),
            'target_domains': ['sketch', 'photo', 'cartoon'],
            'initial_delay_limits': (30, 60),
            'strategy': 'replace'
        },
        "RV_domain_change_burst_1": lambda: {
            'burst_interval_limits': (90, 130),
            'burst_duration_limits': (3, 6),
            'base_rate': 0.0,
            'burst_rate': (0.3, 0.6),
            'target_domains':  ['photo', 'cartoon', 'sketch'],
            'initial_delay_limits': (30, 60),
            'strategy': 'replace'
        },
        "RV_domain_change_burst_2": lambda: {
            'burst_interval_limits': (50, 90),
            'burst_duration_limits': (2, 5),
            'base_rate': 0.0,
            'burst_rate': (0.4, 0.6),
            'target_domains': ['cartoon', 'photo', 'sketch'],
            'initial_delay_limits': (50, 80),
            'strategy': 'replace'
        },
        "step_0": lambda: {
            'step_points': [50, 100, 150],
            'step_rates': [0.004, 0.008, 0.016],
            'step_domains': ['sketch', 'photo', 'cartoon', 'sketch'],
            'strategy': 'replace'
        },
        "step_1": lambda: {
            'step_points': [60, 120, 180],
            'step_rates': [0.004, 0.006, 0.008, 0.01],
            'step_domains': ['sketch', 'cartoon', 'art_painting', 'photo', 'sketch'],
            'strategy': 'replace'
        },
        "constant_drift_domain_change_0": lambda: {
            'drift_rate': 0.016,
            'domain_change_interval': 50,
            'target_domains': ['sketch', 'photo', 'cartoon', 'art_painting'],
            'strategy': 'replace'
        },
        "constant_drift_domain_change_1": lambda: {
            'drift_rate': 0.008,
            'domain_change_interval': 50,
            'target_domains': ['sketch', 'photo', 'cartoon', 'art_painting'],
            'strategy': 'replace'
        },
        "constant_drift_domain_change_2": lambda: {
            'drift_rate': 0.004,
            'domain_change_interval': 50,
            'target_domains': ['sketch', 'photo', 'cartoon', 'art_painting'],
            'strategy': 'replace'
        },
        "sine_wave_domain_change_0": lambda: {
            'amplitude': 0.008,
            'period': 50,
            'target_domains': ['sketch', 'photo', 'cartoon', 'art_painting'],
            'strategy': 'replace'
        },
        "sine_wave_domain_change_1": lambda: {
            'amplitude': 0.016,
            'period': 50,
            'target_domains': ['sketch', 'photo', 'cartoon', 'art_painting'],
            'strategy': 'replace'
        },
        "sine_wave_domain_change_2": lambda: {
            'amplitude': 0.032,
            'period': 50,
            'target_domains': ['sketch', 'photo', 'cartoon', 'art_painting'],
            'strategy': 'replace'
        },
        "intermittent_shifts": lambda: {
            'min_interval': 2,
            'max_interval': 10,
            'burst_duration_range': (5, 15),
            'drift_range': (0.3, 0.6),
            'low_drift_start': 30,
            'low_drift_end': 40,
            'low_drift_rate': 0.2,
            'target_domains': ['sketch', 'cartoon', 'art_painting', 'photo'],
            'strategy': 'replace'
        },
        "quiet_then_low_0": lambda: {
            'burst_interval': 60,
            'burst_duration': 20,
            'base_rate': 0.0,
            'burst_rate': 0.024,
            'target_domains':  ['photo', 'cartoon', 'sketch'],
            'initial_delay': 60,
            'strategy': 'replace'
        },
        "quiet_then_low_1": lambda: {
            'burst_interval': 70,
            'burst_duration': 30,
            'base_rate': 0.0,
            'burst_rate': 0.032,
            'target_domains':  ['photo', 'cartoon', 'sketch'],
            'initial_delay': 50,
            'strategy': 'replace'
        },
        "decaying_spikes": lambda: {
            'initial_burst_interval': 30,   # Initial time between starts of spikes
            'interval_increment_per_spike': 10, # How much the interval increases after each spike
            'max_burst_interval': 120,      # Maximum interval between spikes
            'burst_duration': 3,            # Duration of each spike
            'base_rate': 0.0,               # Drift rate between spikes
            'burst_rate': 0.35,             # Drift rate during a spike
            'target_domains': ['sketch', 'photo', 'cartoon'], # Domains to cycle through for spikes
            'initial_delay': 20,            # Delay before the first spike can occur
            'strategy': 'replace'
        },

        "seasonal_flux": lambda: {
            'cycle_period': 150,            # Duration of a full seasonal cycle (e.g., 150 rounds)
            'max_amplitude_drift_rate': 0.016, # Max drift rate during peak transition between domains
            'base_drift_rate': 0.001,       # A very slow underlying constant drift (can be 0)
            'domain_A': 'photo',            # First primary domain in the cycle
            'domain_B': 'sketch',           # Second primary domain in the cycle
            'initial_phase_offset_t': 0,    # Optional: shifts the start of the sine wave
            'initial_delay': 10,            # Overall initial delay for the schedule
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
        
        if self.schedule_type.startswith("RV_domain_change_burst"):
            self.modify_burst = False
            self.initial_delay = np.random.randint(self.initial_delay_limits[0], self.initial_delay_limits[1] + 1)
            self.burst_interval = np.random.randint(self.burst_interval_limits[0], self.burst_interval_limits[1] + 1)
            self.burst_duration = np.random.randint(self.burst_duration_limits[0], self.burst_duration_limits[1] + 1)
        
        if self.schedule_type == "decaying_spikes":
            # State for decaying_spikes
            self.next_spike_start_time_ds = self.initial_delay
            self.current_spike_end_time_ds = -1 # Tracks if currently in a spike, and when it ends
            self.current_interval_ds = self.initial_burst_interval

        
        # Generate burst periods for intermittent_shifts
        if self.schedule_type == "intermittent_shifts":
            max_t = 1000  # Maximum time horizon, adjust if needed
            self.burst_periods = []
            current_t = 0
            while current_t < max_t:
                interval = np.random.randint(self.min_interval, self.max_interval + 1)
                current_t += interval
                if current_t >= max_t:
                    break
                duration = np.random.randint(self.burst_duration_range[0], self.burst_duration_range[1] + 1)
                burst_start = current_t
                burst_end = min(current_t + duration, max_t)
                drift_rate = np.random.uniform(self.drift_range[0], self.drift_range[1])
                domain_index = len(self.burst_periods) % len(self.target_domains)
                target_domain = self.target_domains[domain_index]
                self.burst_periods.append({
                    'start': burst_start,
                    'end': burst_end,
                    'drift_rate': drift_rate,
                    'target_domain': target_domain
                })
                current_t = burst_end

    def get_drift_params(self, t):
        """Returns (drift_rate, target_domains) for the given time t."""
        self.current_step = t
        if self.schedule_type.startswith("domain_change_burst") or self.schedule_type.startswith("quiet_then_low"):
            if t < self.initial_delay:
                drift_rate = self.base_rate
                target_domains = []
            else:
                adjusted_t = t - self.initial_delay
                cycle_position = adjusted_t % self.burst_interval
                if cycle_position == 0 and adjusted_t >= 0:
                    self.select_new_target_domain()
                if cycle_position < self.burst_duration:
                    drift_rate = self.burst_rate
                    target_domains = [self.current_target_domain]
                else:
                    drift_rate = self.base_rate
                    target_domains = []
        elif self.schedule_type == "intermittent_shifts":
            for burst in self.burst_periods:
                if burst['start'] <= t < burst['end']:
                    drift_rate = burst['drift_rate']
                    target_domains = [burst['target_domain']]
                    break
            else:
                if self.low_drift_start <= t < self.low_drift_end:
                    drift_rate = self.low_drift_rate
                    target_domains = [self.target_domains[0]]
                else:
                    drift_rate = 0.0
                    target_domains = []
        elif self.schedule_type.startswith("RV_domain_change_burst"):
            if t < self.initial_delay:
                drift_rate = self.base_rate
                target_domains = []
            else:
                adjusted_t = t - self.initial_delay
                cycle_position = adjusted_t % self.burst_interval
                if cycle_position == 0 and adjusted_t >= 0:
                    self.select_new_target_domain()
                    self.modify_burst = True
                if cycle_position < self.burst_duration:
                    drift_rate = random.uniform(self.burst_rate[0], self.burst_rate[1])
                    target_domains = [self.current_target_domain]
                else:
                    drift_rate = self.base_rate
                    target_domains = []
                    if self.modify_burst:
                        self.burst_interval = np.random.randint(self.burst_interval_limits[0], self.burst_interval_limits[1] + 1)
                        self.burst_duration = np.random.randint(self.burst_duration_limits[0], self.burst_duration_limits[1] + 1)
                        self.modify_burst = False
        elif self.schedule_type.startswith("step"):
            if t < self.step_points[0]:
                drift_rate = 0.0
                target_domains = [self.step_domains[0]]
            else:
                for i, point in enumerate(self.step_points):
                    if t < point:
                        drift_rate = self.step_rates[i]
                        target_domains = [self.step_domains[i]]
                        break
                else:
                    drift_rate = self.step_rates[-1]
                    target_domains = [self.step_domains[-1]]
        elif "constant_drift_domain_change" in self.schedule_type:
            drift_rate = self.drift_rate
            domain_index = (t // self.domain_change_interval) % len(self.target_domains)
            target_domains = [self.target_domains[domain_index]]
        elif "sine_wave_domain_change" in self.schedule_type:
            drift_rate = self.amplitude * (1 - np.cos(2 * np.pi * t / self.period)) / 2
            domain_index = (t // self.period) % len(self.target_domains)
            target_domains = [self.target_domains[domain_index]]
        elif self.schedule_type == "decaying_spikes":
            drift_rate = self.base_rate
            target_domains_list = [] # Default to no specific target / no drift beyond base_rate

            # Check if a current spike has just ended
            if self.current_spike_end_time_ds != -1 and t >= self.current_spike_end_time_ds:
                # Spike ended. Schedule the next one.
                self.current_interval_ds = min(self.max_burst_interval,
                                            self.current_interval_ds + self.interval_increment_per_spike)
                self.next_spike_start_time_ds = self.current_spike_end_time_ds + self.current_interval_ds
                self.current_spike_end_time_ds = -1 # Reset: not in a spike anymore

            # Check if it's time to start a new spike
            if self.current_spike_end_time_ds == -1 and t >= self.next_spike_start_time_ds:
                self.current_spike_end_time_ds = self.next_spike_start_time_ds + self.burst_duration
                self.select_new_target_domain() # Selects/cycles the target domain for this new spike

            # If currently within a spike's duration
            if self.current_spike_end_time_ds != -1 and self.next_spike_start_time_ds <= t < self.current_spike_end_time_ds:
                drift_rate = self.burst_rate
                target_domains_list = [self.current_target_domain]
            
            return drift_rate, target_domains_list

        elif self.schedule_type == "seasonal_flux":
            if t >= self.initial_delay:
                current_total_drift_rate = self.base_drift_rate
                time_in_cycle = t - self.initial_delay + self.initial_phase_offset_t
                
                # seasonal_factor ranges from -1 to 1.
                # Positive favors domain_B, Negative favors domain_A.
                seasonal_factor = np.sin(2 * np.pi * time_in_cycle / self.cycle_period)
                
                # Intensity of the seasonal shift, |seasonal_factor| * max_amplitude_drift_rate
                seasonal_shift_intensity = abs(seasonal_factor) * self.max_amplitude_drift_rate
                current_total_drift_rate += seasonal_shift_intensity

                if current_total_drift_rate > 0: # Only assign target if there's actual drift
                    if seasonal_factor > 0.001: # Threshold to avoid issues at zero crossing
                        target_domains_list = [self.domain_B] # Drifting towards B
                    elif seasonal_factor < -0.001:
                        target_domains_list = [self.domain_A] # Drifting towards A
                    else:
                        # Near zero-crossing, could drift towards a mix or a default,
                        # or just rely on base_drift_rate if it has its own target.
                        # For simplicity, if very close to zero, let base_drift_rate dominate
                        # without a strong seasonal component, or pick one, e.g., A.
                        # If base_drift_rate is 0 and seasonal is at zero-crossing, drift rate is 0.
                        if self.base_drift_rate == 0: # if no base drift, and at zero-crossing, no target.
                            current_total_drift_rate = 0 # effectively no drift
                        else: # if there's base drift, it applies (target might need to be defined for base drift)
                            target_domains_list = [self.domain_A] # Default or could be another logic for base drift target
            else:
                current_total_drift_rate = 0
                target_domains_list = []
            print(target_domains_list)
            # Ensure drift rate is not negative if base_drift_rate somehow caused it (should not happen here)
            current_total_drift_rate = max(0, current_total_drift_rate)
            if current_total_drift_rate == 0:
                target_domains_list = []
            print(target_domains_list)
            return current_total_drift_rate, target_domains_list
        else:
            raise ValueError(f"Unsupported schedule type: {self.schedule_type}")
        
        return drift_rate, target_domains

        
    def select_new_target_domain(self):
        """Selects the next target domain for burst schedules."""
        if 'target_domains' not in self.__dict__:
            return None
        if not self.first_burst_completed:
            self.first_burst_completed = True
            return self.current_target_domain
        self.domain_index = (self.domain_index + 1) % len(self.target_domains)
        self.current_target_domain = self.target_domains[self.domain_index]
        return self.current_target_domain

    def get_schedule_params(self):
        """Returns the current schedule parameters."""
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
        elif 'burst_interval_limits' in self.__dict__:
            params.update({
                'burst_interval_limits': self.burst_interval_limits,
                'burst_duration_limits': self.burst_duration_limits,
                'base_rate': self.base_rate,
                'burst_rate': self.burst_rate,
                'initial_delay_limits': self.initial_delay_limits,
                'target_domains': self.target_domains
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
    
# Evaluate Policy Function
def evaluate_policy_under_drift(
    source_domains,
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
    # Set the seed and define transform
    common_seed = 0
    set_seed(common_seed)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Policy
    policy = Policy(alpha=learning_rate, L_i=L_i, K_p=K_p, K_d=K_d)
    
    # Define datasets for training+infrence and holdout for testing
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
    
    set_seed(seed)
    # Define drift objects
    train_drift = DomainDrift(
        train_data_handler,
        source_domains=source_domains,
        target_domains=source_domains,
        drift_rate=0,  # Initially no drift
        desired_size=DSET_SIZE
    )
    holdout_drift = DomainDrift(
        train_data_handler,
        source_domains=source_domains,
        target_domains=source_domains,
        drift_rate=0,  # Initially no drift
        desired_size=int(DSET_SIZE * 0.25)
    )
    
    # Agents: training and inference and holdout for performance
    agent_train = DriftAgent(
        client_id=0,
        model_architecture=model_architecture,
        domain_drift=train_drift,
        batch_size=batch_size,
        device=device
    )
    
    agent_holdout = DriftAgent(
        client_id=1,
        model_architecture=model_architecture,
        domain_drift=holdout_drift,
        batch_size=batch_size,
        device=device
    )
    
    # print(len(agent_train.domain_drift.current_indices))
    # print(len(agent_train.current_dataset))
    # print(len(agent_holdout.domain_drift.current_indices))
    # print(len(agent_holdout.current_dataset))
        
    # Model, Optimizer
    model = agent_train.get_model()
    model.load_state_dict(torch.load(model_path))
    agent_holdout.set_model_params(agent_train.get_model_params())
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Loss, results, timing
    loss_array = [agent_train.evaluate(metric_fn=criterion, verbose=False)]
    loss_best = loss_array[0]
    policy.set_initial_loss(loss_array[0])
    results = []
    time_arr = []
    
    print(f"Initial loss: {loss_array[0]}")
    print(policy)
    print(policy.loss_initial)
    
    for t in range(n_rounds):
        time_round = time.time()
        
        # Apply drift
        drift_rate, target_domains = drift_scheduler.get_drift_params(t)
        
        agent_train.set_drift_rate(drift_rate)
        print(drift_rate, target_domains)
        if drift_rate > 0:
            agent_train.set_target_domains(target_domains)
        agent_train.apply_drift()
        
        agent_holdout.set_drift_rate(drift_rate)
        if drift_rate > 0:
            agent_holdout.set_target_domains(target_domains)
        agent_holdout.apply_drift()
        
        # Calculate current loss
        loss_curr = agent_holdout.evaluate(metric_fn=criterion, test_size=1.0, verbose=False)
        acc_curr = agent_holdout.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)
        loss_array.append(loss_curr)
        loss_prev = loss_array[-1]
        if loss_curr < loss_best:
            loss_best = loss_curr
        
        # Get the policy decision
        decision = policy.policy_decision(
            decision_id=policy_id,
            loss_curr=loss_curr,
            loss_prev=loss_prev,
            current_time=t,
            V=V,
            pi_bar=pi_bar,
            loss_best=loss_best
        )
        # Update the model if decision is made
        print(f'Loss historical diff: {loss_curr - loss_best}; Loss Difference: {loss_curr - loss_prev}, Decision: {decision}')
        if decision:
            update_loss = agent_train.update_steps(
                num_updates=n_steps, 
                optimizer=optimizer, 
                loss_fn=criterion, 
                verbose=True
            )
            agent_holdout.set_model_params(agent_train.get_model_params())
        
        # Time it takes for a round
        time_round = time.time() - time_round
        time_arr.append(time_round)
        
        # Save results
        result_dict = {
            't': t,
            'current_accuracy': acc_curr,
            'current_loss': loss_curr,
            'train_loss': update_loss if decision else None,
            'decision': decision,
            'drift_rate': drift_rate, 
            'target_domains': agent_train.domain_drift.target_domains,
        }
        print(f"Round {t+1} took {time_round}s, results:")
        print(result_dict)
        results.append(result_dict)
        
    # Save the results to a json
    schedule_params = drift_scheduler.get_schedule_params()
    schedule_type = schedule_params['schedule_type']
    src_domains_str = "_".join(source_domains)
    results_filename = (
        f"../../data/results/policy_{policy_id}_setting_{setting_id}_schedule_{schedule_type}"
        f"_src_{src_domains_str}_model_{model_architecture.__name__}_img_{img_size}_seed_{seed}.json"
    )
    # Data to save
    data = {
        'parameters': {
            'source_domains': source_domains,
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
    # Save the results to a json file
    with open(results_filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Average time per round: {np.mean(time_arr)}")
    print(f"Total time: {np.sum(time_arr)}")
    
    return results
    
    
# Hyperparameters and Constants
cluster_used = 'gautschi'
settings = {
        0: {'pi_bar': 0.1, 'V': 65, 'L_i': 1.0},
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
        40: {'pi_bar': 0.02, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':1},
        41: {'pi_bar': 0.03, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':1},
        42: {'pi_bar': 0.05, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':1},
        43: {'pi_bar': 0.07, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':1},
        44: {'pi_bar': 0.15, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':1},
        45: {'pi_bar': 0.20, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':1},
        46: {'pi_bar': 0.30, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':1},
        47: {'pi_bar': 0.25, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':5},
        48: {'pi_bar': 0.30, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':5},
        49: {'pi_bar': 0.35, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':5},
        50: {'pi_bar': 0.40, 'V': 10, 'L_i': 0, 'K_p': 0.25, 'K_d': 2.0},
        51: {'pi_bar': 0.45, 'V': 10, 'L_i': 0, 'K_p': 0.10, 'K_d': 2.0},
        52: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 1.0},
        53: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.25, 'K_d': 1.0},
        54: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.10, 'K_d': 1.0},
        55: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 0.5},
        56: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.25, 'K_d': 0.5},
        57: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.10, 'K_d': 0.5},
        60: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':5},
        61: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 0.1, 'lr': 0.01, 'n_steps':5},
        62: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 0.25, 'lr': 0.01, 'n_steps':5},
        63: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 0.35, 'lr': 0.01, 'n_steps':5},
        64: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.4, 'K_d': 0.1, 'lr': 0.01, 'n_steps':5},
        65: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.4, 'K_d': 0.25, 'lr': 0.01, 'n_steps':5},
        66: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.4, 'K_d': 0.35, 'lr': 0.01, 'n_steps':5},
        67: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 3.0, 'K_d': 1.0, 'lr': 0.01, 'n_steps':5},
        68: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 5.0, 'K_d': 1.0, 'lr': 0.01, 'n_steps':5},
        69: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 10.0, 'K_d': 1.0, 'lr': 0.01, 'n_steps':5},
        70: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 0.5, 'lr': 0.01, 'n_steps':5},
        71: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 1.0, 'K_d': 0.5, 'lr': 0.01, 'n_steps':5},
        72: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 3.0, 'K_d': 0.5, 'lr': 0.01, 'n_steps':5},
        73: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 5.0, 'K_d': 0.5, 'lr': 0.01, 'n_steps':5},
        74: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 10.0, 'K_d': 0.5, 'lr': 0.01, 'n_steps':5},
        75: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 0.1, 'lr': 0.01, 'n_steps':5},
        76: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 1.0, 'K_d': 0.1, 'lr': 0.01, 'n_steps':5},
        77: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.75, 'K_d': 0.5, 'lr': 0.01, 'n_steps':5},
        78: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 1.25, 'K_d': 0.05, 'lr': 0.01, 'n_steps':5},
        79: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 10.0, 'K_d': 0.1, 'lr': 0.01, 'n_steps':5},
    }
DSET_SIZE = 1024

# Main function 
def main():
    # Get the command line arguments
    parser = argparse.ArgumentParser(description="PACS CNN Evaluation with Dynamic Drift")
    parser.add_argument('--schedule_type', type=str, default='domain_change_burst_0',
                        choices=list(DriftScheduler.SCHEDULE_CONFIGS.keys()),
                        help='Type of drift rate schedule')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--src_domains', type=str, nargs='+', default=['photo',])
    parser.add_argument('--n_rounds', type=int, default=250)
    parser.add_argument('--policy_id', type=int, default=2)
    parser.add_argument('--setting_id', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='PACSCNN_4', choices=['PACSCNN_1', 'PACSCNN_2', 'PACSCNN_3', 'PACSCNN_4'], help='Model architecture to use')
    parser.add_argument('--img_size', type=int, default=128, help='Size to resize images to (img_size x img_size)')
    args = parser.parse_args()
    
    # Model selection
    model_architectures = {
        'PACSCNN_1': PACSCNN_1,
        'PACSCNN_2': PACSCNN_2,
        'PACSCNN_3': PACSCNN_3,
        'PACSCNN_4': PACSCNN_4,
    }
    print(f'Model used: {args.model_name}')
    model_path = f"../../../../models/concept_drift_models/{args.model_name}_{'_'.join(args.src_domains)}_img_{args.img_size}_seed_{args.seed}.pth"
    
    # Settings selection
    settings_used = settings[args.setting_id]
    K_p = settings_used.get('K_p', 1.0)
    K_d = settings_used.get('K_d', 1.0)
    lr = settings_used.get('lr', 0.01)
    num_steps = settings_used.get('n_steps', 1)
    
    # Initialize the drift scheduler
    drift_scheduler = DriftScheduler(args.schedule_type)
    
    # Evaluate the policy
    evaluate_policy_under_drift(
        source_domains=args.src_domains,
        model_path=model_path,
        model_architecture=model_architectures[args.model_name],
        img_size=args.img_size,
        seed=args.seed,
        drift_scheduler=drift_scheduler,
        n_rounds=args.n_rounds,
        learning_rate=lr,
        policy_id=args.policy_id,
        setting_id=args.setting_id,
        pi_bar=settings_used['pi_bar'],
        V=settings_used['V'],
        L_i=settings_used['L_i'],
        K_p=K_p,
        K_d=K_d, 
        n_steps=num_steps,
    )
    
if __name__ == "__main__":
    main()