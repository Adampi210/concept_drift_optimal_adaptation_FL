import matplotlib.pyplot as plt
import os
import glob
import re
import numpy as np
from collections import defaultdict

def plot_metrics(data_path):
    # Initialize lists to store data
    epochs = []
    accuracies = []
    costs = []
   
    # Read the file and parse data
    with open(data_path, 'r') as f:
        # Skip parameter lines
        for _ in range(15):
            next(f)
           
        # Read the actual data
        for i, line in enumerate(f):
            t, acc, _, _, cost = map(float, line.strip().split(','))
            epochs.append(t)
            accuracies.append(acc * 100)  # Convert to percentage
            costs.append(cost)
            if i > 20:
                break

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
   
    # Plot accuracy
    ax1.plot(epochs, accuracies, 'b-', linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy over Time')
    ax1.grid(True)
   
    # Plot cost
    ax2.plot(epochs, costs, 'r-', linewidth=2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Cost')
    ax2.set_title('Model Cost over Time')
    ax2.grid(True)
   
    # Adjust layout to prevent overlap
    plt.tight_layout()
   
    # Save the figure with .png extension instead of .csv
    output_path = os.path.splitext(data_path)[0] + '.png'
    output_dir = os.path.join(os.path.dirname(os.path.dirname(data_path)), 'plots')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(output_path))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
   
    return fig

def plot_averaged_metrics(base_filename):
    """
    Plot averaged metrics for all files matching the base filename pattern with different seeds.
    
    Args:
        base_filename (str): Base filename pattern like 'policy_0_setting_0_src_domains_cartoon_tgt_domains_art_painting'
                           (without the '_seed_X.csv' part)
    """
    # Extract directory and create full pattern for glob
    data_dir = os.path.dirname(base_filename)
    if not data_dir:
        data_dir = '.'
    
    # Create the pattern to match files with any seed
    base_pattern = os.path.basename(base_filename)
    pattern = os.path.join(data_dir, f"{base_pattern}_seed_*.csv")
    
    files = glob.glob(pattern)
    if not files:
        raise ValueError(f"No files found matching pattern: {pattern}")
    
    print(f"Found {len(files)} files to average")
    
    # Dictionary to store data for averaging
    all_data = defaultdict(lambda: {'accuracies': [], 'costs': []})
    epochs = None
    
    # Read all matching files
    for file_path in files:
        current_epochs = []
        current_accuracies = []
        current_costs = []
        
        with open(file_path, 'r') as f:
            # Skip parameter lines
            for _ in range(15):
                next(f)
            
            # Read the actual data
            for i, line in enumerate(f):
                t, acc, _, _, cost = map(float, line.strip().split(','))
                current_epochs.append(t)
                current_accuracies.append(acc * 100)  # Convert to percentage
                current_costs.append(cost)
                if i > 20:
                    break
        
        # Store the first epochs array as reference
        if epochs is None:
            epochs = current_epochs
        
        # Store data for averaging
        for i, epoch in enumerate(current_epochs):
            all_data[epoch]['accuracies'].append(current_accuracies[i])
            all_data[epoch]['costs'].append(current_costs[i])
    
    # Calculate averages and standard deviations
    avg_accuracies = [np.mean(all_data[epoch]['accuracies']) for epoch in epochs]
    avg_costs = [np.mean(all_data[epoch]['costs']) for epoch in epochs]
    std_accuracies = [np.std(all_data[epoch]['accuracies']) for epoch in epochs]
    std_costs = [np.std(all_data[epoch]['costs']) for epoch in epochs]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot average accuracy with standard deviation
    ax1.plot(epochs, avg_accuracies, 'b-', linewidth=2, label='Average')
    ax1.fill_between(epochs, 
                    [a - s for a, s in zip(avg_accuracies, std_accuracies)],
                    [a + s for a, s in zip(avg_accuracies, std_accuracies)],
                    color='b', alpha=0.2, label='±1 std')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Average Accuracy (%)')
    ax1.set_title('Average Model Accuracy over Time')
    ax1.grid(True)
    ax1.legend()
    
    # Plot average cost with standard deviation
    ax2.plot(epochs, avg_costs, 'r-', linewidth=2, label='Average')
    ax2.fill_between(epochs,
                    [c - s for c, s in zip(avg_costs, std_costs)],
                    [c + s for c, s in zip(avg_costs, std_costs)],
                    color='r', alpha=0.2, label='±1 std')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Average Cost')
    ax2.set_title('Average Model Cost over Time')
    ax2.grid(True)
    ax2.legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the averaged plot
    output_filename = f"averaged_{os.path.basename(base_filename)}.png"
    output_dir = os.path.join(os.path.dirname(data_dir), 'plots')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def get_unique_base_patterns(directory):
    # Get all csv files in the directory
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    
    # Use set for unique patterns
    unique_patterns = set()
    
    # Regular expression to match everything before '_seed_'
    pattern = re.compile(r'(.*?)_seed_\d+\.csv$')
    
    for file_path in all_files:
        match = pattern.search(file_path)
        if match:
            base_pattern = match.group(1)
            unique_patterns.add(base_pattern)
    
    # Convert to sorted list for consistent ordering
    return sorted(list(unique_patterns))

if __name__ == "__main__":
    results_dir = '../data/results/'
    result_unique_filenames = get_unique_base_patterns(results_dir)

    for result_filename in result_unique_filenames:
        plot_averaged_metrics(result_filename)
        
