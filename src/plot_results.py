# analyze_loss_evolution.py

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Tuple

def load_and_average_losses(results_dir: str, config_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load loss data for a specific configuration across all seeds.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (timesteps, averaged_losses, std_losses)
    """
    # Dictionary to store losses for each seed
    seed_losses = defaultdict(list)
    times = None
    
    # Find all result files for this config
    for filename in os.listdir(results_dir):
        if not filename.startswith(f'loss_eval_drift_{config_name}_seed_') or not filename.endswith('.csv'):
            continue
            
        filepath = os.path.join(results_dir, filename)
        
        # Read losses
        current_times = []
        current_losses = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                t, loss = int(row[0]), float(row[1])
                current_times.append(t)
                current_losses.append(loss)
        
        if times is None:
            times = np.array(current_times)
        seed_losses[len(seed_losses)].extend(current_losses)
    
    if not seed_losses:
        raise ValueError(f"No data found for configuration: {config_name}")
    
    # Convert to numpy array and compute statistics
    losses_array = np.array(list(seed_losses.values()))
    avg_losses = np.mean(losses_array, axis=0)
    std_losses = np.std(losses_array, axis=0)
    
    return times, avg_losses, std_losses

def plot_averaged_losses(results_dir: str, config_name: str, save_dir: str):
    """Plot averaged losses over time for a specific configuration."""
    times, losses, stds = load_and_average_losses(results_dir, config_name)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(times, losses, label='Average Loss', marker='o', markersize=2)
    plt.fill_between(times, losses - stds, losses + stds, alpha=0.2, label='±1 std')
    
    plt.xlabel('Evaluation Round (t)')
    plt.ylabel('Average Loss')
    title = f'Loss Evolution: {config_name.replace("_", " ").title()}'
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    save_path = os.path.join(save_dir, f'config_{config_name}_avg_loss.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved average loss plot: {save_path}")

def plot_loss_derivatives(results_dir: str, config_name: str, save_dir: str):
    """Plot loss derivatives for a specific configuration."""
    times, losses, stds = load_and_average_losses(results_dir, config_name)
    
    # Calculate derivatives
    derivatives = np.diff(losses)
    derivative_times = times[1:]
    derivative_stds = np.sqrt(stds[1:]**2 + stds[:-1]**2)  # Error propagation
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(derivative_times, derivatives, label='Loss Change Rate', marker='o', markersize=2)
    plt.fill_between(derivative_times, 
                    derivatives - derivative_stds,
                    derivatives + derivative_stds,
                    alpha=0.2,
                    label='±1 std')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Evaluation Round (t)')
    plt.ylabel('Loss Change: L(t+1) - L(t)')
    title = f'Loss Change Rate: {config_name.replace("_", " ").title()}'
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    save_path = os.path.join(save_dir, f'config_{config_name}_derivative_loss.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved derivative plot: {save_path}")

def main():
    # Specify configurations to plot
    configs_to_plot = [
        "slow_rotation",
        "medium_rotation",
        "fast_rotation",
        "slow_scaling",
        "medium_scaling",
        "fast_scaling",
        "mild_noise",
        "medium_noise",
        "strong_noise",
    ]
    
    RESULTS_DIR = "../data/results/"
    PLOTS_DIR = "../data/plots/"
    
    # Create plots directory if it doesn't exist
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Generate plots for each configuration
    for config in configs_to_plot:
        try:
            print(f"\nProcessing configuration: {config}")
            plot_averaged_losses(RESULTS_DIR, config, PLOTS_DIR)
            plot_loss_derivatives(RESULTS_DIR, config, PLOTS_DIR)
        except Exception as e:
            print(f"Error processing {config}: {str(e)}")
            continue

if __name__ == "__main__":
    main()