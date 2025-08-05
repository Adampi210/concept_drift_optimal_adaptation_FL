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

# --- Helper library imports ---
from fvcore.nn import FlopCountAnalysis
try:
    import pynvml
    pynvml.nvmlInit()
    # Assuming GPU 0. Change if you use a different one.
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_AVAILABLE = True
except Exception as e:
    NVML_AVAILABLE = False
    print(f"Warning: pynvml not available. Energy consumption will not be measured. Error: {e}")

from fl_toolkit import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
print(f"PyTorch Version: {torch.__version__}\n")

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

class DriftScheduler:
    SCHEDULE_CONFIGS = {
        "burst": lambda: {
            'burst_interval': 120,
            'burst_duration': 3,
            'base_rate': 0.0,
            'burst_rate': 0.4,
            'target_domains': ['photo', 'cartoon', 'sketch'],
            'initial_delay': 45,
            'strategy': 'replace'
        },
        "spikes": lambda: {
            'burst_interval_limits': (90, 130),
            'burst_duration_limits': (3, 6),
            'base_rate': 0.0,
            'burst_rate': (0.3, 0.6),
            'target_domains':  ['photo', 'cartoon', 'sketch'],
            'initial_delay_limits': (30, 60),
            'strategy': 'replace'
        },
        "step": lambda: {
            'step_points': [60, 120, 180],
            'step_rates': [0.004, 0.006, 0.008, 0.01],
            'step_domains': ['sketch', 'cartoon', 'art_painting', 'photo', 'sketch'],
            'strategy': 'replace'
        },
        "constant": lambda: {
            'drift_rate': 0.016,
            'domain_change_interval': 50,
            'target_domains': ['sketch', 'photo', 'cartoon', 'art_painting'],
            'strategy': 'replace'
        },
        "wave": lambda: {
            'burst_interval': 70,
            'burst_duration': 30,
            'base_rate': 0.0,
            'burst_rate': 0.032,
            'target_domains':  ['photo', 'cartoon', 'sketch'],
            'initial_delay': 50,
            'strategy': 'replace'
        },
        "decaying_spikes": lambda: {
            'initial_burst_interval': 30,
            'interval_increment_per_spike': 10,
            'max_burst_interval': 120,
            'burst_duration': 3,
            'base_rate': 0.0,
            'burst_rate': 0.35,
            'target_domains': ['sketch', 'photo', 'cartoon'],
            'initial_delay': 20,
            'strategy': 'replace'
        },
        "seasonal_flux": lambda: {
            'cycle_period': 150,
            'max_amplitude_drift_rate': 0.016,
            'base_drift_rate': 0.001,
            'domain_A': 'photo',
            'domain_B': 'sketch',
            'initial_phase_offset_t': 0,
            'initial_delay': 10,
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
        
        if self.schedule_type.startswith("spikes"):
            self.modify_burst = False
            self.initial_delay = np.random.randint(self.initial_delay_limits[0], self.initial_delay_limits[1] + 1)
            self.burst_interval = np.random.randint(self.burst_interval_limits[0], self.burst_interval_limits[1] + 1)
            self.burst_duration = np.random.randint(self.burst_duration_limits[0], self.burst_duration_limits[1] + 1)
        
        if self.schedule_type == "decaying_spikes":
            self.next_spike_start_time_ds = self.initial_delay
            self.current_spike_end_time_ds = -1
            self.current_interval_ds = self.initial_burst_interval


        
    def get_drift_params(self, t):
        """Returns (drift_rate, target_domains) for the given time t."""
        self.current_step = t
        if self.schedule_type.startswith("burst") or self.schedule_type.startswith("wave"):
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
        elif self.schedule_type.startswith("spikes"):
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
        elif "constant" in self.schedule_type:
            drift_rate = self.drift_rate
            domain_index = (t // self.domain_change_interval) % len(self.target_domains)
            target_domains = [self.target_domains[domain_index]]
        elif self.schedule_type == "decaying_spikes":
            drift_rate = self.base_rate
            target_domains_list = [] 

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
                seasonal_factor = np.sin(2 * np.pi * time_in_cycle / self.cycle_period)
                
                seasonal_shift_intensity = abs(seasonal_factor) * self.max_amplitude_drift_rate
                current_total_drift_rate += seasonal_shift_intensity

                if current_total_drift_rate > 0: # Only assign target if there's actual drift
                    if seasonal_factor > 0.001: # Threshold to avoid issues at zero crossing
                        target_domains_list = [self.domain_B] # Drifting towards B
                    elif seasonal_factor < -0.001:
                        target_domains_list = [self.domain_A] # Drifting towards A
                    else:
                        if self.base_drift_rate == 0: # if no base drift, and at zero-crossing, no target.
                            current_total_drift_rate = 0 # effectively no drift
                        else: # if there's base drift, it applies (target might need to be defined for base drift)
                            target_domains_list = [self.domain_A] # Default or could be another logic for base drift target
                else:
                    current_total_drift_rate = 0
                    target_domains_list = []
            # Ensure drift rate is not negative if base_drift_rate somehow caused it (should not happen here)
            current_total_drift_rate = max(0, current_total_drift_rate)
            if current_total_drift_rate == 0:
                target_domains_list = []
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

def evaluate_policy_under_drift(
    source_domains=['photo',],
    model_path="pacs_cnn_model.pth",
    model_architecture=None,
    img_size=128,
    seed=0,
    drift_scheduler=None,
    n_rounds=20,
    learning_rate=0.01,
    pi_bar=0.1,
    n_steps=5,
    batch_size=128
):
    # --- Setup ---
    set_seed(seed)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data_handler = PACSDataHandler(transform=transform)

    train_drift = DomainDrift(
        train_data_handler,
        source_domains=source_domains,
        target_domains=source_domains,
        drift_rate=0,  # Initially no drift
        desired_size=DSET_SIZE
    )
    agent_train = DriftAgent(
        client_id=0,
        model_architecture=model_architecture,
        domain_drift=train_drift,
        batch_size=batch_size,
        device=device
    )
    
    model = agent_train.get_model()
    if not os.path.exists(model_path):
        print(f"Creating dummy model file at: {model_path}")
        torch.save(model.state_dict(), model_path)
    model.load_state_dict(torch.load(model_path))
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # ### 1. PRE-CALCULATE STATIC RESOURCE COSTS ###
    print("\n" + "="*20 + " Pre-calculated Static Costs per Update " + "="*20)
    # Computation Cost (GFLOPs)
    sample_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
    flops = FlopCountAnalysis(model, sample_input)
    gflops_per_pass = flops.total() / 1e9
    compute_gflops_per_update = (gflops_per_pass * 2) * n_steps
    print(f"Computation: {compute_gflops_per_update:.2f} GFLOPs")

    # Data Processed Cost (GB)
    bytes_per_sample = 3 * img_size * img_size * 4  # 3 channels, 32-bit float
    data_processed_gb_per_update = (batch_size * n_steps * bytes_per_sample) / (1024**3)
    print(f"Data Processed: {data_processed_gb_per_update:.4f} GB")
    print("="*72 + "\n")


    # ### 2. RUN EXPERIMENT AND MEASURE DYNAMIC COSTS ###
    resource_costs_per_update = []
    for t in range(n_rounds):
        print(f"--- Round {t+1}/{n_rounds} ---")

        # --- Start Resource Measurement ---
        if torch.cuda.is_available(): torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_time = time.perf_counter()
        start_power_mw = pynvml.nvmlDeviceGetPowerUsage(handle) if NVML_AVAILABLE else 0

        # --- The Actual Model Update ---
        update_loss = agent_train.update_steps(
            num_updates=n_steps,
            optimizer=optimizer,
            loss_fn=criterion,
            verbose=False
        )

        # --- End Resource Measurement ---
        if torch.cuda.is_available(): torch.cuda.synchronize(device)
        end_time = time.perf_counter()
        end_power_mw = pynvml.nvmlDeviceGetPowerUsage(handle) if NVML_AVAILABLE else 0

        # --- Calculate and Store Resource Costs for this update ---
        time_s = end_time - start_time
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        mem_time_gbs = (peak_mem_mb / 1024) * time_s
        avg_power_w = ((start_power_mw + end_power_mw) / 2) / 1000
        energy_j = avg_power_w * time_s if NVML_AVAILABLE else 0

        current_costs = {
            "time_s": time_s,
            "compute_gflops": compute_gflops_per_update,
            "data_processed_gb": data_processed_gb_per_update,
            "mem_time_gbs": mem_time_gbs,
            "energy_j": energy_j,
        }
        resource_costs_per_update.append(current_costs)
        print(f"Update completed in {time_s:.3f}s. Loss: {update_loss:.4f}")


    # ### 3. AGGREGATE AND ANALYZE RESULTS ###
    print("\n" + "="*25 + " Final Resource Analysis " + "="*25)
    if not resource_costs_per_update:
        print("No updates were performed, no resource analysis available.")
        return

    # Convert list of dicts to a dict of lists for easier numpy processing
    data = {key: [d[key] for d in resource_costs_per_update] for key in resource_costs_per_update[0]}
    summary_stats = {}

    print(f"{'Metric':<20} | {'Average':>12} | {'Std Dev':>12} | {'Min':>12} | {'Max':>12}")
    print("-" * 77)
    for key, values in data.items():
        name, unit = key.replace('_', ' ').title().rsplit(' ', 1)
        avg = np.mean(values)
        std = np.std(values)
        min_v = np.min(values)
        max_v = np.max(values)
        summary_stats[key] = {
            'average': avg,
            'std_dev': std,
            'min': min_v,
            'max': max_v,
            'unit': unit
        }
        print(f"{name:<14} ({unit}) | {avg:>12.4f} | {std:>12.4f} | {min_v:>12.4f} | {max_v:>12.4f}")
    print("=" * 77 + "\n")

    # ### 4. SAVE RESULTS TO JSON ###
    results_filename = f'PACSCNN_photo_img_128_seed_{seed}_resources.json'
    # Add run parameters to the results file for context
    run_params = {
        'seed': seed,
        'n_rounds': n_rounds,
        'n_steps_per_update': n_steps,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    output_data = {
        'run_parameters': run_params,
        'summary_stats_per_update': summary_stats
    }

    with open(results_filename, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Resource analysis results saved to: {results_filename}")


    return data


# Hyperparameters and Constants
DSET_SIZE = 1024

# Main function
def main():
    parser = argparse.ArgumentParser(description="PACS CNN Resource Calculation")
    parser.add_argument('--seed', type=int, default=0)
    # Add other arguments if you need to control them from the command line
    args = parser.parse_args()

    model_path = f"../../../../models/concept_drift_models/PACSCNN_photo_img_128_seed_{args.seed}.pth"
    drift_scheduler = DriftScheduler('burst')

    evaluate_policy_under_drift(
        source_domains=['photo',],
        model_path=model_path,
        model_architecture=PACSCNN,
        img_size=128,
        seed=args.seed,
        drift_scheduler=drift_scheduler,
        n_rounds=250, # Using 50 rounds for a more stable average
        learning_rate=0.01,
        pi_bar=0.1,
        n_steps=5,
    )

if __name__ == "__main__":
    main()
    if NVML_AVAILABLE:
        pynvml.nvmlShutdown()
