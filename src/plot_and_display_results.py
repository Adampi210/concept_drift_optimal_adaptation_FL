import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def read_main_data(main_path):
    """
    Reads the main PACSCNN CSV file.

    Args:
        main_path (str): Path to the main PACSCNN CSV file.

    Returns:
        tuple: (epochs, accuracies, losses)
    """
    try:
        df = pd.read_csv(main_path)
        epochs = df['epoch'].tolist()
        accuracies = df['accuracy'].tolist()
        losses = df['loss'].tolist()
        return epochs, accuracies, losses
    except Exception as e:
        print(f"Error reading main file {main_path}: {e}")
        return [], [], []

def read_drift_data(drift_path):
    """
    Reads the drift policy CSV file.

    Args:
        drift_path (str): Path to the drift policy CSV file.

    Returns:
        tuple: (t_epochs, accuracies, losses)
    """
    t_epochs = []
    accuracies = []
    losses = []
    try:
        with open(drift_path, 'r') as f:
            lines = f.readlines()
        
        # Find the index where data starts
        data_start_idx = None
        for idx, line in enumerate(lines):
            if line.strip() == 't,accuracy,loss,decision':
                data_start_idx = idx + 1
                break
        
        if data_start_idx is None:
            print(f"No data header found in drift file {drift_path}.")
            return t_epochs, accuracies, losses
        
        # Read data starting from data_start_idx
        for line in lines[data_start_idx:]:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue  # Skip malformed lines
            t, acc, loss, decision = parts[:4]
            try:
                t_epochs.append(int(t))
                accuracies.append(float(acc) * 100)  # Convert to percentage
                losses.append(float(loss))
            except ValueError:
                continue  # Skip lines with invalid data
        return t_epochs, accuracies, losses
    except Exception as e:
        print(f"Error reading drift file {drift_path}: {e}")
        return t_epochs, accuracies, losses

def map_drift_to_main(policy_files, main_files):
    """
    Maps each drift file to its corresponding main file based on source domain, target domain, setting, and seed.

    Args:
        policy_files (list): List of drift policy file paths.
        main_files (list): List of main PACSCNN file paths.

    Returns:
        dict: Mapping of (source, target, policy_id, setting) to list of (main_path, drift_path).
    """
    mapping = defaultdict(list)
    # Updated regex patterns to include 'setting'
    # Example policy filename: policy_2_setting_1_src_domains_photo_tgt_domains_cartoon_seed_123.csv
    policy_pattern = re.compile(
        r'policy_(\d+)_setting_(\d+)_src_domains_(.*?)_tgt_domains_(.*?)_seed_(\d+)\.csv$'
    )
    # Example main filename: PACSCNN_photo_seed_123_results.csv
    main_pattern = re.compile(r'PACSCNN_(.*?)_seed_(\d+)_results\.csv$')

    # Create a dictionary for main files for quick lookup
    main_dict = {}
    for main in main_files:
        main_match = main_pattern.search(os.path.basename(main))
        if main_match:
            source = main_match.group(1)
            seed = main_match.group(2)
            main_dict[(source, seed)] = main

    # Iterate over policy files and find corresponding main files
    for policy in policy_files:
        policy_match = policy_pattern.search(os.path.basename(policy))
        if policy_match:
            policy_id = policy_match.group(1)
            setting = policy_match.group(2)
            source = policy_match.group(3)
            target = policy_match.group(4)
            seed = policy_match.group(5)
            main_key = (source, seed)
            if main_key in main_dict:
                main_path = main_dict[main_key]
                mapping[(source, target, policy_id, setting)].append((main_path, policy))
            else:
                print(f"No corresponding main file for policy file {policy}. Expected main PACSCNN_{source}_seed_{seed}_results.csv")
        else:
            print(f"Policy file {policy} does not match the expected pattern.")
    return mapping

def average_metrics(combined_data):
    """
    Averages the metrics across seeds for each epoch.

    Args:
        combined_data (dict): Mapping of epoch to list of accuracies and losses.

    Returns:
        tuple: (sorted_epochs, avg_accuracies, avg_losses, std_accuracies, std_losses)
    """
    epochs = sorted(combined_data.keys())
    avg_accuracies = []
    avg_losses = []
    std_accuracies = []
    std_losses = []
    for epoch in epochs:
        accs = combined_data[epoch]['accuracies']
        losses = combined_data[epoch]['losses']
        avg_accuracies.append(np.mean(accs))
        avg_losses.append(np.mean(losses))
        std_accuracies.append(np.std(accs))
        std_losses.append(np.std(losses))
    return epochs, avg_accuracies, avg_losses, std_accuracies, std_losses

def plot_metrics(source, target, policy_id, setting, epochs, accuracies, losses, std_acc, std_loss, drift_epoch, output_path):
    """
    Plots the averaged accuracy and loss with drift indication.

    Args:
        source (str): Source domain.
        target (str): Target domain.
        policy_id (str): Policy identifier.
        setting (str): Setting number.
        epochs (list): List of epoch numbers.
        accuracies (list): Averaged accuracies.
        losses (list): Averaged losses.
        std_acc (list): Standard deviation of accuracies.
        std_loss (list): Standard deviation of losses.
        drift_epoch (int): Epoch where drift begins.
        output_path (str): Path to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot Accuracy
    ax1.plot(epochs, accuracies, color='blue', label='Average Accuracy')
    ax1.fill_between(
        epochs,
        np.array(accuracies) - np.array(std_acc),
        np.array(accuracies) + np.array(std_acc),
        color='blue',
        alpha=0.2,
        label='±1 Std Dev'
    )
    ax1.axvline(x=drift_epoch, color='red', linestyle='--', linewidth=2, label='Drift Point')
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'Average Accuracy Over Time\n(Source: {source}, Target: {target}, Policy: {policy_id}, Setting: {setting})', fontsize=14)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, linestyle='--', linewidth=0.5)

    # Plot Loss
    ax2.plot(epochs, losses, color='green', label='Average Loss')
    ax2.fill_between(
        epochs,
        np.array(losses) - np.array(std_loss),
        np.array(losses) + np.array(std_loss),
        color='green',
        alpha=0.2,
        label='±1 Std Dev'
    )
    ax2.axvline(x=drift_epoch, color='red', linestyle='--', linewidth=2, label='Drift Point')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Average Loss Over Time', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot to {output_path}")

def process_and_plot(mapping, output_dir, policy_id, setting_id, source_domain, target_domain, main_max_epoch=100):
    """
    Processes the mapping and generates plots for the specified source-target pair, policy, and setting.

    Args:
        mapping (dict): Mapping of (source, target, policy_id, setting) to list of (main_path, drift_path).
        output_dir (str): Directory to save the plots.
        policy_id (str): Policy identifier to filter.
        setting_id (str): Setting identifier to filter.
        source_domain (str): Source domain to filter.
        target_domain (str): Target domain to filter.
        main_max_epoch (int): Maximum epoch in the main training phase.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the specific key for filtering
    specific_key = (source_domain, target_domain, policy_id, setting_id)

    if specific_key not in mapping:
        print(f"No data found for Policy ID: {policy_id}, Setting ID: {setting_id}, Source: {source_domain}, Target: {target_domain}.")
        return

    file_pairs = mapping[specific_key]
    combined_acc = defaultdict(lambda: {'accuracies': [], 'losses': []})

    for main_path, drift_path in file_pairs:
        main_epochs, main_accuracies, main_losses = read_main_data(main_path)
        drift_epochs, drift_accuracies, drift_losses = read_drift_data(drift_path)

        if not main_epochs or not drift_epochs:
            print(f"Skipping pair Main: {main_path}, Drift: {drift_path} due to incomplete data.")
            continue

        # Determine drift starting epoch
        last_main_epoch = main_epochs[-1]
        drift_start_epoch = last_main_epoch + 1

        # Shift drift epochs
        shifted_drift_epochs = [epoch + drift_start_epoch for epoch in drift_epochs]

        # Combine epochs, accuracies, and losses
        combined_epochs = main_epochs + shifted_drift_epochs
        combined_accuracies = main_accuracies + drift_accuracies
        combined_losses = main_losses + drift_losses

        for epoch, acc, loss in zip(combined_epochs, combined_accuracies, combined_losses):
            combined_acc[epoch]['accuracies'].append(acc)
            combined_acc[epoch]['losses'].append(loss)

    if not combined_acc:
        print(f"No valid data for Source: {source_domain}, Target: {target_domain}, Policy: {policy_id}, Setting: {setting_id}. Skipping plot.")
        return

    # Average metrics
    sorted_epochs = sorted(combined_acc.keys())
    avg_accuracies = []
    avg_losses = []
    std_acc = []
    std_loss = []
    for epoch in sorted_epochs:
        accs = combined_acc[epoch]['accuracies']
        losses = combined_acc[epoch]['losses']
        avg_accuracies.append(np.mean(accs))
        avg_losses.append(np.mean(losses))
        std_acc.append(np.std(accs))
        std_loss.append(np.std(losses))

    # Drift epoch is where drift starts
    drift_epoch = last_main_epoch + 1  # The first drift epoch

    # Plot
    output_filename = f"policy_{policy_id}_setting_{setting_id}_accuracy_loss_{source_domain}_to_{target_domain}.png"
    output_path = os.path.join(output_dir, output_filename)
    plot_metrics(
        source_domain, target_domain, policy_id, setting_id,
        sorted_epochs, avg_accuracies, avg_losses,
        std_acc, std_loss, drift_epoch, output_path
    )

def plot_policy_results(policy_id, setting_id, source_domains, target_domains):
    """
    Main function to plot policy results for multiple source-target domain combinations.

    Args:
        policy_id (int or str): Policy identifier to filter policy files.
        setting_id (int or str): Setting identifier to filter policy files.
        source_domains (list of str): List of source domains.
        target_domains (list of str): List of target domains.
    """
    # Define the directory containing all CSV files
    results_dir = '../data/results/'  # Update this path as per your directory structure

    # Get all CSV files in the directory
    all_csv_files = glob.glob(os.path.join(results_dir, "*.csv"))

    # Convert policy_id and setting_id to strings for consistent matching
    policy_id_str = str(policy_id)
    setting_id_str = str(setting_id)

    # Iterate over each source-target pair and process
    for source_domain in source_domains:
        for target_domain in target_domains:
            # Escape any special characters in domain names for regex
            source_domain_escaped = re.escape(source_domain)
            target_domain_escaped = re.escape(target_domain)

            # Regex patterns for policy and main files
            policy_pattern = re.compile(
                rf'^policy_{policy_id_str}_setting_{setting_id_str}_src_domains_{source_domain_escaped}_tgt_domains_{target_domain_escaped}_seed_\d+\.csv$'
            )
            main_pattern = re.compile(
                rf'^PACSCNN_{source_domain_escaped}_seed_\d+_results\.csv$'
            )

            # Filter policy and main files based on current pair
            policy_files = [
                f for f in all_csv_files
                if policy_pattern.match(os.path.basename(f))
            ]
            main_files = [
                f for f in all_csv_files
                if main_pattern.match(os.path.basename(f))
            ]

            print(f"\nProcessing Source: {source_domain}, Target: {target_domain}")
            print(f"Found {len(main_files)} main PACSCNN files.")
            print(f"Found {len(policy_files)} drift policy_{policy_id_str}_setting_{setting_id_str} files.")

            if not policy_files:
                print(f"No policy files found for Policy ID: {policy_id_str}, Setting ID: {setting_id_str}, Source: {source_domain}, Target: {target_domain}. Skipping this pair.")
                continue

            # Map drift files to main files
            mapping = map_drift_to_main(policy_files, main_files)

            print(f"Mapping drift files to main files completed. Found {len(mapping)} unique combinations for this pair.")

            if not mapping:
                print("No valid mappings found for this pair. Skipping plot.")
                continue

            # Define the output directory
            output_dir = '../data/plots'  # Update this path as per your directory structure

            # Iterate over each mapped combination and plot
            for (src, tgt, pid, sid), file_pairs in mapping.items():
                # Ensure that the current pair matches the specified pair
                if src != source_domain or tgt != target_domain:
                    continue  # Skip if not matching

                process_and_plot(mapping, output_dir, pid, sid, src, tgt)

    print("\nAll specified plots have been generated.")

if __name__ == "__main__":
    # Example usage:
    # Define the list of source and target domains you want to plot
    source_domains = ['photo',]
    target_domains = ['sketch',]

    # Specify the policy_id and setting_id you want to plot
    policy_id = 1
    setting_id = 0

    # Call the main plotting function with the specified parameters
    plot_policy_results(policy_id, setting_id, source_domains, target_domains)
