#
# Comprehensive Drift Detection Method Benchmarking
#
# This script integrates benchmarking logic directly into the user's provided
# experimental framework to produce robust and relevant results.
#
# It measures and compares the following:
#   1. Baseline training and evaluation times.
#   2. The computational overhead (in wall-clock time and GFLOPS) of three
#      different drift detection/adaptation strategies:
#      - RCCDA (Loss-Based): The user's proposed lightweight policy.
#      - ADWIN (Error-Based): A classic streaming algorithm.
#      - K-S Test (Distribution-Based): A statistical test on feature distributions.
#
# The final output is a table suitable for inclusion in a research paper.
#

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
from scipy.stats import ks_2samp
from collections import deque
from thop import profile

os.environ['HF_DATASETS_OFFLINE'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
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

# Methods overheads
# --- Self-Contained ADWIN Implementation ---
class ADWIN:
    def __init__(self, delta=0.02):
        self.delta = delta
        self.reset()
    def reset(self): self.window, self.total, self.variance, self.n, self.drift_detected = deque(), 0.0, 0.0, 0, False
    def update(self, value):
        self.drift_detected = False; self.window.append(value); self.n += 1
        if self.n > 30 and self.n % 10 == 0: self._check_for_drift()
    def _check_for_drift(self):
        n0, total0 = 0, 0.0
        for i in range(len(self.window) - 1):
            n0 += 1; total0 += self.window[i]; n1 = self.n - n0
            if n0 < 2 or n1 < 2: continue
            mean0, mean1 = total0 / n0, (self.total - total0) / n1
            diff = abs(mean0 - mean1)
            m_inv = 1.0 / n0 + 1.0 / n1
            epsilon = np.sqrt(0.5 * m_inv * np.log(4 / self.delta))
            if diff > epsilon:
                self.drift_detected = True
                for _ in range(n0): self.n -= 1; self.total -= self.window.popleft()
                break

# --- Overhead Measurement Functions ---
def measure_rccda_overhead(loss_curr, loss_prev, loss_best):
    start_time = time.perf_counter()
    e_t = loss_curr - loss_best
    delta_e = loss_curr - loss_prev
    _ = e_t + delta_e # pd_term
    # FLOPS are negligible (a few dozen), so we report 0.
    return (time.perf_counter() - start_time) * 1000, 0.0

def measure_adwin_overhead(outputs, labels):
    adwin = ADWIN()
    start_time = time.perf_counter()
    outputs_tensor = torch.from_numpy(outputs)
    _, predicted = torch.max(outputs_tensor, 1)
    predicted_np = predicted.numpy()
    errors = (predicted_np != labels)
    for error in errors:
        adwin.update(error.item())
    # FLOPS are dominated by the initial forward pass, not this loop.
    return (time.perf_counter() - start_time) * 1000, 0.0

def measure_ks_test_overhead(agent, data_loader, reference_features):
    start_time = time.perf_counter()
    feature_extractor = agent.get_model().features.eval()
    current_features_list = []
    with torch.no_grad():
        # FIX: Handle batches that may have more than 2 elements.
        for batch in data_loader:
            inputs = batch[0]
            features = feature_extractor(inputs.to(device))
            current_features_list.append(features.cpu().numpy().flatten())
    if not current_features_list: return 0.0, 0.0
    current_features = np.concatenate(current_features_list)
    if len(reference_features) > 1 and len(current_features) > 1:
        ks_2samp(reference_features, current_features)
    # FLOPS are dominated by the feature extraction forward pass.
    return (time.perf_counter() - start_time) * 1000, 0.0

# Utility Functions
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

 
# --- Main Evaluation Function (Modified from user's code) ---
def run_full_benchmark(source_domains, model_path, model_architecture, img_size, seed, n_rounds=10, batch_size=128):
    set_seed(seed)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print('Initiating dataset')
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
        desired_size=DSET_SIZE
    )
    
    print(f'creating clients')
    agent_train = DriftAgent(0, model_architecture, train_drift, batch_size, device)
    agent_holdout = DriftAgent(1, model_architecture, holdout_drift, batch_size, device)

    try:
        agent_train.get_model().load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Warning: Could not load model from {model_path}. Using random weights. Error: {e}")

    optimizer = optim.SGD(agent_train.get_model().parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # --- FLOPS Calculation ---
    print("Calculating FLOPS for evaluation...")
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    forward_flops, _ = profile(agent_train.get_model(), inputs=(dummy_input,), verbose=False)
    
    # --- Data for Benchmarking ---
    # FIX: Use a DataLoader for efficient feature extraction.
    train_subset_for_loader = torch.utils.data.Subset(full_dataset, train_indices)
    train_loader = DataLoader(train_subset_for_loader, batch_size=batch_size)
    holdout_subset_for_loader = torch.utils.data.Subset(full_dataset, holdout_indices)
    holdout_loader = DataLoader(holdout_subset_for_loader, batch_size=batch_size)

    feature_extractor = agent_train.get_model().features.to(device)
    reference_features_list = []
    with torch.no_grad():
        # FIX: Handle batches that may have more than 2 elements.
        for batch in train_loader:
            inputs = batch[0]
            features = feature_extractor(inputs.to(device)).detach().cpu().numpy()
            reference_features_list.append(features.reshape(features.shape[0], -1))
    reference_features = np.concatenate(reference_features_list).flatten()


    timings = {'train': [], 'eval': [], 'rccda': [], 'adwin': [], 'ks_test': []}
    print(f"Running benchmark for {n_rounds} rounds...")
    for t in range(n_rounds):
        print(f"\nRound {t+1}/{n_rounds}...")
        # 1. Baseline Eval Time
        eval_start_time = time.time()
        loss_curr, (outputs, labels) = agent_holdout.evaluate_outputs(metric_fn=criterion, test_size=1.0, verbose=False)
        timings['eval'].append((time.time() - eval_start_time) * 1000)

        # 2. Detection Overhead Times (post-evaluation)
        rccda_time, _ = measure_rccda_overhead(loss_curr, 0.5, 0.4)
        print(f"RCCDA Time: {rccda_time:.6f} ms")
        adwin_time, _ = measure_adwin_overhead(outputs, labels)
        print(f"ADWIN Time: {adwin_time:.6f} ms")
        ks_time, _ = measure_ks_test_overhead(agent_holdout, holdout_loader, reference_features)
        print(f"K-S Test Time: {ks_time:.6f} ms")
        timings['rccda'].append(rccda_time)
        timings['adwin'].append(adwin_time)
        timings['ks_test'].append(ks_time)

        # 3. Baseline Train Time
        train_start_time = time.time()
        agent_train.update_steps(num_updates=1, optimizer=optimizer, loss_fn=criterion, verbose=False)
        timings['train'].append((time.time() - train_start_time) * 1000)

    # Compile results for this run
    avg_results = {key: np.mean(val) for key, val in timings.items()}
    avg_results['flops_eval'] = forward_flops / 1e9 # GFLOPS
    
    return avg_results

DSET_SIZE = 1024

# Main function 
def main():
    seeds = [0, 1, 2]
    src_domains = ['photo', 'art_painting', 'cartoon', 'sketch']
    all_run_results = []

    for seed in seeds:
        for src_domain in src_domains:
            print(f"\n--- Running Benchmark for Seed: {seed}, Domain: {src_domain} ---")
            model_path = f"../../../../models/concept_drift_models/PACSCNN_{src_domain}_img_128_seed_{seed}.pth" # Assumes models are in a local 'models' folder
            
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}. Creating dummy file.")
                if not os.path.exists('models'): os.makedirs('models')
                torch.save(PACSCNN().state_dict(), model_path)


            results = run_full_benchmark(
                source_domains=[src_domain],
                model_path=model_path,
                model_architecture=PACSCNN,
                img_size=128,
                seed=seed,
                n_rounds=3,
                batch_size=128
            )
            all_run_results.append(results)

    # --- Final Report Generation ---
    print("\n\n" + "="*95)
    print(" " * 28 + "COMPREHENSIVE BENCHMARK RESULTS")
    print("="*95)
    
    if not all_run_results:
        print("No results to report.")
        return

    # Aggregate results
    final_avg = {key: np.mean([res[key] for res in all_run_results]) for key in all_run_results[0]}
    final_std = {key: np.std([res[key] for res in all_run_results]) for key in all_run_results[0]}

    print(f"Averaged over {len(all_run_results)} runs ({len(seeds)} seeds x {len(src_domains)} domains).\n")
    print(f"Baseline Eval Time: {final_avg['eval']:.2f} ± {final_std['eval']:.2f} ms")
    print(f"Baseline Train Time (1 step): {final_avg['train']:.2f} ± {final_std['train']:.2f} ms")
    print("-" * 95)
    print(f"{'Method':<15} | {'Detection Overhead (ms)':<25} | {'Required Ops (GFLOPS)':<25} | {'Notes'}")
    print("-" * 95)
    
    # RCCDA
    rccda_time_str = f"{final_avg['rccda']:.7f} ± {final_std['rccda']:.7f}"
    print(f"{'RCCDA (Ours)':<15} | {rccda_time_str:<25} | {'~0':<25} | {'Negligible arithmetic ops'}")
    
    # ADWIN
    adwin_time_str = f"{final_avg['adwin']:.7f} ± {final_std['adwin']:.7f}"
    print(f"{'ADWIN':<15} | {adwin_time_str:<25} | {'N/A':<25} | {'Sequential loop over all errors'}")

    # KS-Test
    ks_time_str = f"{final_avg['ks_test']:.7f} ± {final_std['ks_test']:.7f}"
    print(f"{'KS-Test':<15} | {ks_time_str:<25} | {final_avg['flops_eval']:.2f} (for features) | {'Full forward pass + stats test'}")

    print("="*95)

    
if __name__ == "__main__":
    main()
