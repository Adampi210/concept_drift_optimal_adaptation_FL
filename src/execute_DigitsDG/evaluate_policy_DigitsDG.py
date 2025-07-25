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
class DigitsDGCNN(BaseModelArchitecture):
    def __init__(self, num_classes=10):
        super(DigitsDGCNN, self).__init__()
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
        elif decision_id == 5:
            if self.loss_initial is None:
                raise ValueError("Initial loss not set for Policy 5")
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
        "burst": lambda: {
            'burst_interval': 120,
            'burst_duration': 3,
            'base_rate': 0.0,
            'burst_rate': 0.4,
            'target_domains': ['mnist', 'syn', 'svhn'],
            'initial_delay': 45,
            'strategy': 'replace'
        },
        "spikes": lambda: {
            'burst_interval_limits': (90, 130),
            'burst_duration_limits': (3, 6),
            'base_rate': 0.0,
            'burst_rate': (0.3, 0.6),
            'target_domains':  ['mnist', 'syn', 'svhn'],
            'initial_delay_limits': (30, 60),
            'strategy': 'replace'
        },
        "step": lambda: {
            'step_points': [60, 120, 180],
            'step_rates': [0.004, 0.006, 0.008, 0.01],
            'step_domains': ['svhn', 'syn', 'mnist_m', 'mnist', 'svhn'],
            'strategy': 'replace'
        },
        "constant": lambda: {
            'drift_rate': 0.016,
            'domain_change_interval': 50,
            'target_domains': ['svhn', 'mnist', 'syn', 'mnist_m'],
            'strategy': 'replace'
        },
        "wave": lambda: {
            'burst_interval': 70,
            'burst_duration': 30,
            'base_rate': 0.0,
            'burst_rate': 0.032,
            'target_domains':  ['mnist', 'syn', 'svhn'],
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
            'target_domains': ['svhn', 'mnist', 'syn'],
            'initial_delay': 20,
            'strategy': 'replace'
        },
        "seasonal_flux": lambda: {
            'cycle_period': 150,
            'max_amplitude_drift_rate': 0.016,
            'base_drift_rate': 0.001,
            'domain_A': 'mnist',
            'domain_B': 'svhn',
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
    ### ROOT DIRECTORY with DigitsDG dataset, replce with custom
    root_dir = f'/scratch/gautschi/apiasecz/data/DigitsDG/digits_dg/'
    full_dataset = DigitsDGDataHandler(root_dir=root_dir, transform=transform).dataset
    dataset_size = len(full_dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)  # Randomize indices

    # Split into 80% training, 20% holdout
    train_size = int(0.8 * dataset_size)
    train_indices = indices[:train_size]
    holdout_indices = indices[train_size:]

    # Create subsets
    train_data_handler = DigitsDGDataHandler(root_dir=root_dir, transform=transform)
    train_data_handler.set_subset(train_indices)
    holdout_data_handler = DigitsDGDataHandler(root_dir=root_dir, transform=transform)
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
settings = {
        0: {'pi_bar': 0.02, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':1},
        1: {'pi_bar': 0.02, 'V': 10, 'L_i': 0, 'K_p': 0.1, 'K_d': 2.0, 'lr': 0.01, 'n_steps':1},
        2: {'pi_bar': 0.02, 'V': 10, 'L_i': 0, 'K_p': 0.1, 'K_d': 1.0, 'lr': 0.01, 'n_steps':1},
        3: {'pi_bar': 0.02, 'V': 10, 'L_i': 0, 'K_p': 0.1, 'K_d': 0.5, 'lr': 0.01, 'n_steps':1},
        4: {'pi_bar': 0.02, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 1.0, 'lr': 0.01, 'n_steps':1},
        5: {'pi_bar': 0.02, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 0.5, 'lr': 0.01, 'n_steps':1},
        6: {'pi_bar': 0.02, 'V': 10, 'L_i': 0, 'K_p': 0.1, 'K_d': 0.1, 'lr': 0.01, 'n_steps':1},
        7: {'pi_bar': 0.02, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':1},
        8: {'pi_bar': 0.02, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':1},
        9: {'pi_bar': 0.02, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.0, 'lr': 0.01, 'n_steps':1},
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
        40: {'pi_bar': 0.02, 'V': 10, 'L_i': 0, 'K_p': 0.1, 'K_d': 0.05, 'lr': 0.05, 'n_steps':1},
        41: {'pi_bar': 0.03, 'V': 10, 'L_i': 0, 'K_p': 0.28, 'K_d': 0.05, 'lr': 0.05, 'n_steps':1},
        42: {'pi_bar': 0.05, 'V': 10, 'L_i': 0, 'K_p': 0.55, 'K_d': 0.1, 'lr': 0.05, 'n_steps':1},
        43: {'pi_bar': 0.07, 'V': 10, 'L_i': 0, 'K_p': 1.0, 'K_d': 0.2, 'lr': 0.05, 'n_steps':1},
        44: {'pi_bar': 0.15, 'V': 10, 'L_i': 0, 'K_p': 4.0, 'K_d': 1.0, 'lr': 0.05, 'n_steps':1},
        45: {'pi_bar': 0.20, 'V': 10, 'L_i': 0, 'K_p': 7.0, 'K_d': 2.0, 'lr': 0.05, 'n_steps':1},
        46: {'pi_bar': 0.25, 'V': 10, 'L_i': 0, 'K_p': 14.0, 'K_d': 6.0, 'lr': 0.05, 'n_steps':1},
        47: {'pi_bar': 0.30, 'V': 10, 'L_i': 0, 'K_p': 20.0, 'K_d': 8.0, 'lr': 0.05, 'n_steps':1},
        50: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.25, 'K_d': 2.0},
        51: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.10, 'K_d': 2.0},
        52: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 1.0},
        53: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.25, 'K_d': 1.0},
        54: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.10, 'K_d': 1.0},
        55: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 0.5},
        56: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.25, 'K_d': 0.5},
        57: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.10, 'K_d': 0.5},
        60: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 2.5, 'K_d': 0.5, 'lr': 0.05, 'n_steps':1},
        61: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 1.0, 'K_d': 0.5, 'lr': 0.05, 'n_steps':1},
        62: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 2.0, 'K_d': 0.1, 'lr': 0.05, 'n_steps':1},
        63: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.3, 'K_d': 0.25, 'lr': 0.05, 'n_steps':1},
        64: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 0.1, 'lr': 0.05, 'n_steps':1},
        65: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 1.0, 'lr': 0.05, 'n_steps':1},
        66: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 1.0, 'K_d': 1.0, 'lr': 0.05, 'n_steps':1},
        67: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 3.0, 'K_d': 1.0, 'lr': 0.05, 'n_steps':1},
        68: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 5.0, 'K_d': 1.0, 'lr': 0.05, 'n_steps':1},
        69: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 10.0, 'K_d': 1.0, 'lr': 0.05, 'n_steps':1},
        70: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 1.0, 'K_d': 0.5, 'lr': 0.05, 'n_steps':1},
        71: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.4, 'K_d': 0.1, 'lr': 0.05, 'n_steps':1},
        72: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 1.0, 'K_d': 0.5, 'lr': 0.05, 'n_steps':1},
        73: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 1.5, 'K_d': 0.5, 'lr': 0.05, 'n_steps':1},
        74: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 1.25, 'K_d': 0.1, 'lr': 0.05, 'n_steps':1},
        75: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 0.1, 'lr': 0.05, 'n_steps':1},
        76: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.4, 'K_d': 0.1, 'lr': 0.05, 'n_steps':1},
        77: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 0.3, 'K_d': 0.25, 'lr': 0.05, 'n_steps':1},
        78: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 1.25, 'K_d': 0.05, 'lr': 0.05, 'n_steps':1},
        79: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 10.0, 'K_d': 0.1, 'lr': 0.05, 'n_steps':1},
    
    }

DSET_SIZE = 1024

# Main function 
def main():
    # Get the command line arguments
    parser = argparse.ArgumentParser(description="DigitsDG CNN Evaluation with Dynamic Drift")
    parser.add_argument('--schedule_type', type=str, default='burst',
                        choices=list(DriftScheduler.SCHEDULE_CONFIGS.keys()),
                        help='Type of drift rate schedule')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--src_domains', type=str, nargs='+', default=['mnist',])
    parser.add_argument('--n_rounds', type=int, default=250)
    parser.add_argument('--policy_id', type=int, default=2)
    parser.add_argument('--setting_id', type=int, default=60)
    parser.add_argument('--model_name', type=str, default='DigitsDGCNN', choices=['DigitsDGCNN',], help='Model architecture to use')
    parser.add_argument('--img_size', type=int, default=32, help='Size to resize images to (img_size x img_size)')
    args = parser.parse_args()
    
    # Model selection
    model_architectures = {
        'DigitsDGCNN': DigitsDGCNN,
    }
    print(f'Model used: {args.model_name}')
    ### MODIFY IF STORED IN A DIFFERENT LOCATION
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