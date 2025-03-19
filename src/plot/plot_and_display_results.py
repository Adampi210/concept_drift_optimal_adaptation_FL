import os
import glob
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def read_main_data(main_path):
    """
    Reads the main PACSCNN JSON file.

    Args:
        main_path (str): Path to the main PACSCNN JSON file.

    Returns:
        tuple: (epochs, accuracies, losses)
    """
    try:
        with open(main_path, 'r') as f:
            data = json.load(f)
        results = data['results']
        epochs = [entry['epoch'] for entry in results]
        accuracies = [entry['accuracy'] for entry in results]  # Already in percentage
        losses = [entry['loss'] for entry in results]
        return epochs, accuracies, losses
    except Exception as e:
        print(f"Error reading main file {main_path}: {e}")
        return [], [], []

def read_drift_data(drift_path):
    """
    Reads the drift policy JSON file.

    Args:
        drift_path (str): Path to the drift policy JSON file.

    Returns:
        tuple: (t_epochs, accuracies, losses, decisions, drift_rates)
    """
    t_epochs = []
    accuracies = []
    losses = []
    decisions = []
    drift_rates = []
    try:
        with open(drift_path, 'r') as f:
            data = json.load(f)
        results = data['results']
        for entry in results:
            t_epochs.append(entry['t'])
            accuracies.append(entry['accuracy'] * 100)  # Convert to percentage
            losses.append(entry['loss'])
            decisions.append(int(entry['decision']))  # Ensure integer
            drift_rates.append(entry['drift_rate'])
        return t_epochs, accuracies, losses, decisions, drift_rates
    except Exception as e:
        print(f"Error reading drift file {drift_path}: {e}")
        return [], [], [], [], []
    
def map_drift_to_main(policy_files, main_files, model_name='PACSCNN'):
    """
    Maps each drift file to its corresponding main file based on source domain, seed, and model architecture.

    Args:
        policy_files (list): List of drift policy file paths.
        main_files (list): List of main PACSCNN file paths.
        model_name (str): Model architecture name (e.g., 'PACSCNN_1'). Defaults to 'PACSCNN'.

    Returns:
        dict: Mapping of (source, target, policy_id, setting) to list of (main_path, drift_path).
    """
    mapping = defaultdict(list)
    # Policy pattern: policy_(\d+)_setting_(\d+)_schedule_(.*?)_src_(.*?)_tgt_(.*?)_seed_(\d+)\.json
    policy_pattern = re.compile(
        r'policy_(\d+)_setting_(\d+)_schedule_(.*?)_src_(.*?)_tgt_(.*?)_seed_(\d+)\.json$'
    )
    # Main pattern: {model_name}_(.*?)_seed_(\d+)\.json
    main_pattern = re.compile(rf'{re.escape(model_name)}_(.*?)_seed_(\d+)\.json$')

    # Create a dictionary for main files
    main_dict = {}
    for main in main_files:
        main_match = main_pattern.search(os.path.basename(main))
        if main_match:
            source = main_match.group(1)
            seed = main_match.group(2)
            main_dict[(source, seed)] = main

    # Map policy files
    for policy in policy_files:
        policy_match = policy_pattern.search(os.path.basename(policy))
        if policy_match:
            policy_id = policy_match.group(1)
            setting = policy_match.group(2)
            schedule = policy_match.group(3)  # Included for completeness, not used in key
            source = policy_match.group(4)
            target = policy_match.group(5)
            seed = policy_match.group(6)
            main_key = (source, seed)
            if main_key in main_dict:
                main_path = main_dict[main_key]
                mapping[(source, target, policy_id, setting)].append((main_path, policy))
            else:
                print(f"No corresponding main file for policy file {policy}. Expected {model_name}_{source}_seed_{seed}.json")
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

def plot_policy_results(policy_id, setting_id, source_domains, target_domains, model_name='PACSCNN'):
    """
    Main function to plot policy results for multiple source-target domain combinations.

    Args:
        policy_id (int or str): Policy identifier to filter policy files.
        setting_id (int or str): Setting identifier to filter policy files.
        source_domains (list of str): List of source domains.
        target_domains (list of str): List of target domains.
        model_name (str): Model architecture name (e.g., 'PACSCNN_1'). Defaults to 'PACSCNN'.
    """
    results_dir = '../../data/results/'
    all_json_files = glob.glob(os.path.join(results_dir, "*.json"))
    policy_id_str = str(policy_id)
    setting_id_str = str(setting_id)

    for source_domain in source_domains:
        for target_domain in target_domains:
            source_domain_escaped = re.escape(source_domain)
            target_domain_escaped = re.escape(target_domain)

            policy_pattern = re.compile(
                rf'^policy_{policy_id_str}_setting_{setting_id_str}_schedule_.*?_src_{source_domain_escaped}_tgt_{target_domain_escaped}_seed_\d+\.json$'
            )
            main_pattern = re.compile(
                rf'^{re.escape(model_name)}_{source_domain_escaped}_seed_\d+\.json$'
            )

            policy_files = [f for f in all_json_files if policy_pattern.match(os.path.basename(f))]
            main_files = [f for f in all_json_files if main_pattern.match(os.path.basename(f))]

            print(f"\nProcessing Source: {source_domain}, Target: {target_domain}")
            print(f"Found {len(main_files)} main {model_name} files.")
            print(f"Found {len(policy_files)} drift policy_{policy_id_str}_setting_{setting_id_str} files.")

            if not policy_files:
                print(f"No policy files found for Policy ID: {policy_id_str}, Setting ID: {setting_id_str}, Source: {source_domain}, Target: {target_domain}. Skipping.")
                continue

            mapping = map_drift_to_main(policy_files, main_files, model_name)
            print(f"Mapping completed. Found {len(mapping)} unique combinations.")

            if not mapping:
                print("No valid mappings found. Skipping plot.")
                continue

            output_dir = '../../data/plots'
            for (src, tgt, pid, sid), file_pairs in mapping.items():
                if src != source_domain or tgt != target_domain:
                    continue
                process_and_plot(mapping, output_dir, pid, sid, src, tgt)
    print("\nAll specified plots have been generated.")
    
def plot_multi_policy_results(policy_setting_pairs, source_domain, target_domain):
    """
    Plots results for multiple policies on the same plot.

    Args:
        policy_setting_pairs (list): List of (policy_id, setting_id) tuples
        source_domain (str): Source domain name
        target_domain (str): Target domain name
    """
    results_dir = '../../data/results/'
    output_dir = '../../data/plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    combined_data = {}

    colors = plt.cm.rainbow(np.linspace(0, 1, len(policy_setting_pairs)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    drift_epoch = None

    for (policy_id, setting_id), color in zip(policy_setting_pairs, colors):
        policy_id_str = str(policy_id)
        setting_id_str = str(setting_id)

        policy_pattern = re.compile(
            rf'^policy_{policy_id_str}_setting_{setting_id_str}_src_domains_{source_domain}_tgt_domains_{target_domain}_seed_\d+\.csv$'
        )
        main_pattern = re.compile(
            rf'^PACSCNN_{source_domain}_seed_\d+_results\.csv$'
        )

        policy_files = [f for f in all_csv_files if policy_pattern.match(os.path.basename(f))]
        main_files = [f for f in all_csv_files if main_pattern.match(os.path.basename(f))]

        mapping = map_drift_to_main(policy_files, main_files)
        combined_acc = defaultdict(lambda: {'accuracies': [], 'losses': []})

        for (src, tgt, pid, sid), file_pairs in mapping.items():
            for main_path, drift_path in file_pairs:
                main_epochs, main_accuracies, main_losses = read_main_data(main_path)
                drift_epochs, drift_accuracies, drift_losses = read_drift_data(drift_path)

                if not main_epochs or not drift_epochs:
                    continue

                last_main_epoch = main_epochs[-1]
                if drift_epoch is None:
                    drift_epoch = last_main_epoch + 1

                shifted_drift_epochs = [epoch + drift_epoch for epoch in drift_epochs]
                combined_epochs = main_epochs + shifted_drift_epochs
                combined_accuracies = main_accuracies + drift_accuracies
                combined_losses = main_losses + drift_losses

                for epoch, acc, loss in zip(combined_epochs, combined_accuracies, combined_losses):
                    combined_acc[epoch]['accuracies'].append(acc)
                    combined_acc[epoch]['losses'].append(loss)

        if combined_acc:
            epochs = sorted(combined_acc.keys())
            avg_accuracies = []
            avg_losses = []
            std_accuracies = []
            std_losses = []
            
            for epoch in epochs:
                accs = combined_acc[epoch]['accuracies']
                losses = combined_acc[epoch]['losses']
                avg_accuracies.append(np.mean(accs))
                avg_losses.append(np.mean(losses))
                std_accuracies.append(np.std(accs))
                std_losses.append(np.std(losses))

            label = f'Policy {policy_id}, Setting {setting_id}'
            
            ax1.plot(epochs, avg_accuracies, color=color, label=label)
            ax1.fill_between(
                epochs,
                np.array(avg_accuracies) - np.array(std_accuracies),
                np.array(avg_accuracies) + np.array(std_accuracies),
                color=color,
                alpha=0.2
            )

            ax2.plot(epochs, avg_losses, color=color, label=label)
            ax2.fill_between(
                epochs,
                np.array(avg_losses) - np.array(std_losses),
                np.array(avg_losses) + np.array(std_losses),
                color=color,
                alpha=0.2
            )

    if drift_epoch:
        ax1.axvline(x=drift_epoch, color='red', linestyle='--', linewidth=2, label='Drift Point')
        ax2.axvline(x=drift_epoch, color='red', linestyle='--', linewidth=2, label='Drift Point')

    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'Average Accuracy Over Time\n(Source: {source_domain}, Target: {target_domain})', fontsize=14)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, linestyle='--', linewidth=0.5)

    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Average Loss Over Time', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'multi_policy_comparison_{source_domain}_to_{target_domain}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot to {output_path}")
    
def calculate_pi_bar_for_policy_setting(policy_id, setting_id, results_dir="../../data/results/"):
    """
    Calculates the averaged pi_bar (across all seeds) for a given policy and setting.
    pi_bar is defined as:
      (number of 'decision' == 1) / (total number of rows in the data section),
    summed over all files that match the policy/setting.
    
    Args:
        policy_id (int or str): The policy ID to filter.
        setting_id (int or str): The setting ID to filter.
        results_dir (str): Path to the directory containing all CSV files.
    """
    # Convert to string for consistent pattern matching
    policy_str = str(policy_id)
    setting_str = str(setting_id)

    # Example pattern for policy CSV files:
    # policy_3_setting_49_src_domains_photo_tgt_domains_art_painting_seed_4.csv
    pattern = re.compile(
        rf'^policy_{policy_str}_setting_{setting_str}_src_domains_.*_tgt_domains_.*_seed_\d+\.csv$'
    )

    # Collect all CSV files
    all_csv_files = glob.glob(os.path.join(results_dir, "*.csv"))

    # Filter files matching our policy/setting pattern
    policy_files = [f for f in all_csv_files if pattern.match(os.path.basename(f))]

    if not policy_files:
        print(f"No policy files found for Policy ID: {policy_str}, Setting ID: {setting_str}.")
        return

    total_updates_across_seeds = 0
    total_rounds_across_seeds = 0

    for policy_file in policy_files:
        try:
            with open(policy_file, 'r') as f:
                lines = f.readlines()
            
            # Find where data starts
            data_start_idx = None
            for idx, line in enumerate(lines):
                # The data portion starts at "t,accuracy,loss,decision"
                if line.strip().startswith('t,accuracy,loss,decision'):
                    data_start_idx = idx + 1
                    break
            
            if data_start_idx is None:
                print(f"No valid data section found in {policy_file}. Skipping.")
                continue

            # Count how many times decision == 1
            update_count = 0
            total_count = 0

            # From data_start_idx to the end of file
            for line in lines[data_start_idx:]:
                parts = line.strip().split(',')
                if len(parts) < 4:
                    continue  # Skip malformed lines

                # decision is the last element
                decision = parts[3].strip()
                # Try to parse it as integer
                try:
                    decision_int = int(decision)
                except ValueError:
                    continue

                total_count += 1
                if decision_int == 1:
                    update_count += 1

            if total_count == 0:
                print(f"No data rows found after header in {policy_file}. Skipping.")
                continue

            # Accumulate for overall average
            total_updates_across_seeds += update_count
            total_rounds_across_seeds += total_count

        except Exception as e:
            print(f"Error processing file {policy_file}: {e}")

    # Compute final pi_bar if we had valid files
    if total_rounds_across_seeds == 0:
        print("No valid data rows found at all. Can't compute pi_bar.")
        return
    
    pi_bar_overall = total_updates_across_seeds / total_rounds_across_seeds

    print(f"Policy ID: {policy_id}, Setting ID: {setting_id}")
    print(f"Average pi_bar across all seeds = {pi_bar_overall:.4f} "
          f"(updates={total_updates_across_seeds}, total={total_rounds_across_seeds})")
        
def plot_moving_window_updates(policy_id, setting_id, window_size=10, results_dir="../../data/results/"):
            """
            Plots a moving window average of updates for a given policy and setting.
            
            Args:
                policy_id (int or str): The policy ID to filter.
                setting_id (int or str): The setting ID to filter.
                window_size (int): Size of the moving window.
                results_dir (str): Path to the directory containing all CSV files.
            """
            policy_str = str(policy_id)
            setting_str = str(setting_id)

            pattern = re.compile(
                rf'^policy_{policy_str}_setting_{setting_str}_src_domains_.*_tgt_domains_.*_seed_\d+\.csv$'
            )
            
            all_csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
            policy_files = [f for f in all_csv_files if pattern.match(os.path.basename(f))]

            if not policy_files:
                print(f"No policy files found for Policy ID: {policy_str}, Setting ID: {setting_str}.")
                return

            decisions_by_timestep = defaultdict(list)

            for policy_file in policy_files:
                try:
                    with open(policy_file, 'r') as f:
                        lines = f.readlines()
                    
                    data_start_idx = None
                    for idx, line in enumerate(lines):
                        if line.strip().startswith('t,accuracy,loss,decision'):
                            data_start_idx = idx + 1
                            break
                    
                    if data_start_idx is None:
                        continue

                    for line in lines[data_start_idx:]:
                        parts = line.strip().split(',')
                        if len(parts) < 4:
                            continue
                        
                        try:
                            timestep = int(parts[0])
                            # Handle both boolean and integer decision values
                            decision_str = parts[3].lower()
                            if decision_str in ['true', '1']:
                                decision = 1
                            elif decision_str in ['false', '0']:
                                decision = 0
                            else:
                                decision = int(decision_str)
                            decisions_by_timestep[timestep].append(decision)
                        except ValueError:
                            continue

                except Exception as e:
                    print(f"Error processing file {policy_file}: {e}")

            if not decisions_by_timestep:
                print("No valid data found.")
                return

            # Calculate average decisions at each timestep
            timesteps = sorted(decisions_by_timestep.keys())
            avg_decisions = [np.mean(decisions_by_timestep[t]) for t in timesteps]

            # Calculate moving average
            moving_avg = []
            for i in range(len(avg_decisions) - window_size + 1):
                window_avg = np.mean(avg_decisions[i:i + window_size])
                moving_avg.append(window_avg)

            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(timesteps[window_size-1:], moving_avg, label=f'Moving average (window={window_size})')
            plt.xlabel('Timestep')
            plt.ylabel('Average Update Rate')
            plt.title(f'Moving Window Average of Updates\nPolicy {policy_id}, Setting {setting_id}')
            plt.grid(True)
            plt.legend()
            
            output_dir = '../../data/plots'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f'moving_window_updates_policy_{policy_id}_setting_{setting_id}.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Saved plot to {output_path}")

def read_data_with_drift(drift_path):
    """
    Reads the drift policy CSV file including drift rate information.
    Handles both boolean (True/False) and numeric (1/0) decision values.
    
    Args:
        drift_path (str): Path to the drift policy CSV file.
    
    Returns:
        tuple: (t_epochs, accuracies, losses, decisions, drift_rates)
    """
    t_epochs = []
    accuracies = []
    losses = []
    decisions = []
    drift_rates = []
    
    try:
        with open(drift_path, 'r') as f:
            lines = f.readlines()
        
        # Find where data starts
        data_start_idx = None
        for idx, line in enumerate(lines):
            if 't,accuracy,loss,decision' in line.strip():
                data_start_idx = idx + 1
                break
        
        if data_start_idx is None:
            print(f"No data header found in drift file {drift_path}.")
            return t_epochs, accuracies, losses, decisions, drift_rates
        
        # Read data
        for line in lines[data_start_idx:]:
            parts = line.strip().split(',')
            if len(parts) < 4:  # Need at least t, accuracy, loss, decision
                continue
            
            try:
                # Parse basic metrics
                t = int(parts[0])
                acc = float(parts[1]) * 100  # Convert to percentage
                loss = float(parts[2])
                
                # Parse decision - handle both boolean and numeric formats
                decision_str = parts[3].strip().lower()
                if decision_str in ['true', '1']:
                    decision = 1
                elif decision_str in ['false', '0']:
                    decision = 0
                else:
                    try:
                        # Try to parse as float in case it's a number
                        decision = int(float(decision_str))
                    except ValueError:
                        print(f"Unrecognized decision value in {drift_path}: {decision_str}")
                        continue
                
                # Parse drift rate if available
                drift_rate = float(parts[4]) if len(parts) > 4 else None
                
                # Append all values
                t_epochs.append(t)
                accuracies.append(acc)
                losses.append(loss)
                decisions.append(decision)
                drift_rates.append(drift_rate)
                
            except (ValueError, IndexError) as e:
                print(f"Error parsing line in {drift_path}: {line.strip()} - {str(e)}")
                continue
                
        return t_epochs, accuracies, losses, decisions, drift_rates
    except Exception as e:
        print(f"Error reading drift file {drift_path}: {e}")
        return t_epochs, accuracies, losses, decisions, drift_rates

def plot_metrics_with_drift(source, target, policy_id, setting_id, schedule_type,
                          epochs, accuracies, losses, decisions, drift_rates,
                          std_acc, std_loss, std_drift, output_path):
    """
    Plots the metrics including drift rate visualization.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[3, 3, 2])
    
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
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'Metrics Over Time\n(Source: {source}, Target: {target}, Policy: {policy_id}, Setting: {setting_id}, Schedule: {schedule_type})', 
                 fontsize=14)
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
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle='--', linewidth=0.5)

    # Plot Drift Rate
    ax3.plot(epochs, drift_rates, color='red', label='Drift Rate')
    if std_drift is not None:
        ax3.fill_between(
            epochs,
            np.array(drift_rates) - np.array(std_drift),
            np.array(drift_rates) + np.array(std_drift),
            color='red',
            alpha=0.2,
            label='±1 Std Dev'
        )
    
    # Add decision markers
    decision_epochs = [epoch for epoch, decision in zip(epochs, decisions) if decision == 1]
    if decision_epochs:
        ax3.scatter(decision_epochs, [drift_rates[epochs.index(e)] for e in decision_epochs],
                   color='black', marker='x', label='Update Points', zorder=5)
    
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Drift Rate', fontsize=12)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")

def plot_multi_schedule_comparison(policy_id, setting_id, source_domain, target_domain,
                                 schedule_types=['burst', 'oscillating', 'step'],
                                 results_dir='../../data/results/'):
    """
    Plots comparison of different drift schedules for the same policy and setting.
    """
    all_csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    output_dir = '../../data/plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[3, 3, 2])
    colors = plt.cm.rainbow(np.linspace(0, 1, len(schedule_types)))

    for schedule_type, color in zip(schedule_types, colors):
        pattern = re.compile(
            rf'^policy_{policy_id}_setting_{setting_id}_schedule_{schedule_type}_src_domains_{source_domain}_tgt_domains_{target_domain}_seed_\d+\.csv$'
        )
        
        matching_files = [f for f in all_csv_files if pattern.match(os.path.basename(f))]
        
        if not matching_files:
            print(f"No files found for schedule type: {schedule_type}")
            continue
            
        combined_data = defaultdict(lambda: {
            'accuracies': [], 'losses': [], 'decisions': [], 'drift_rates': []
        })
        
        for file_path in matching_files:
            epochs, accs, losses, decisions, drift_rates = read_data_with_drift(file_path)
            
            for idx, epoch in enumerate(epochs):
                combined_data[epoch]['accuracies'].append(accs[idx])
                combined_data[epoch]['losses'].append(losses[idx])
                combined_data[epoch]['decisions'].append(decisions[idx])
                combined_data[epoch]['drift_rates'].append(drift_rates[idx])
        
        if not combined_data:
            continue
            
        epochs = sorted(combined_data.keys())
        avg_accuracies = [np.mean(combined_data[e]['accuracies']) for e in epochs]
        avg_losses = [np.mean(combined_data[e]['losses']) for e in epochs]
        avg_drift_rates = [np.mean(combined_data[e]['drift_rates']) for e in epochs]
        decisions = [np.mean(combined_data[e]['decisions']) > 0.5 for e in epochs]
        
        # Plot on shared axes
        ax1.plot(epochs, avg_accuracies, color=color, label=f'{schedule_type}')
        ax2.plot(epochs, avg_losses, color=color, label=f'{schedule_type}')
        ax3.plot(epochs, avg_drift_rates, color=color, label=f'{schedule_type}')
        
        # Add decision markers
        decision_epochs = [epoch for epoch, decision in zip(epochs, decisions) if decision]
        if decision_epochs:
            ax3.scatter(decision_epochs, 
                       [avg_drift_rates[epochs.index(e)] for e in decision_epochs],
                       color=color, marker='x', alpha=0.5, s=50)

    # Customize plots
    ax1.set_title(f'Schedule Comparison\n(Source: {source_domain}, Target: {target_domain}, Policy: {policy_id}, Setting: {setting_id})')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True)
    ax1.legend(loc='lower right')

    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend(loc='upper right')

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Drift Rate')
    ax3.grid(True)
    ax3.legend(loc='upper right')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 
        f'schedule_comparison_policy_{policy_id}_setting_{setting_id}_{source_domain}_to_{target_domain}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_path}")

def analyze_schedule_performance(policy_id, setting_id, source_domain, target_domain,
                               schedule_types=['burst', 'oscillating', 'step'],
                               results_dir='../../data/results/'):
    """
    Analyzes and prints performance metrics for different schedules.
    """
    all_csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    results = {}
    for schedule_type in schedule_types:
        pattern = re.compile(
            rf'^policy_{policy_id}_setting_{setting_id}_schedule_{schedule_type}_src_domains_{source_domain}_tgt_domains_{target_domain}_seed_\d+\.csv$'
        )
        
        matching_files = [f for f in all_csv_files if pattern.match(os.path.basename(f))]
        
        if not matching_files:
            print(f"No files found for schedule type: {schedule_type}")
            continue
            
        accuracies = []
        update_rates = []
        drift_rates = []
        
        for file_path in matching_files:
            epochs, accs, _, decisions, drifts = read_data_with_drift(file_path)
            
            if epochs:  # Only process if we have data
                accuracies.append(np.mean(accs))
                update_rates.append(np.mean(decisions))
                drift_rates.append(np.mean(drifts))
        
        if accuracies:  # Only store if we have processed data
            results[schedule_type] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'mean_update_rate': np.mean(update_rates),
                'mean_drift_rate': np.mean(drift_rates)
            }
    
    # Print analysis
    print(f"\nPerformance Analysis for Policy {policy_id}, Setting {setting_id}")
    print(f"Source: {source_domain}, Target: {target_domain}\n")
    print("Schedule Type  | Accuracy (%) ± Std | Update Rate | Avg Drift Rate")
    print("-" * 65)
    
    for schedule_type in schedule_types:
        if schedule_type in results:
            r = results[schedule_type]
            print(f"{schedule_type:12} | {r['mean_accuracy']:6.2f} ± {r['std_accuracy']:4.2f} | {r['mean_update_rate']:10.3f} | {r['mean_drift_rate']:13.3f}")

def compare_policies(setting_id, schedule_type, source_domain='photo', target_domain='sketch', 
                     policy_ids=[1, 2, 3, 4], model_name='PACSCNN_3', results_dir='../../data/results/', T=None):
    """
    Compares different policies for the same setting and schedule type, plotting accuracy, loss, 
    drift rate, and cumulative updates.

    Args:
        setting_id (int): Setting ID to analyze.
        schedule_type (str): Type of drift schedule (e.g., 'domain_change_burst_2').
        source_domain (str): Source domain name (default: 'photo').
        target_domain (str): Target domain name (default: 'sketch').
        policy_ids (list): List of policy IDs to compare (default: [1, 2, 3, 4]).
        model_name (str): Model architecture name (e.g., 'PACSCNN') (default: 'PACSCNN').
        results_dir (str): Directory containing result JSON files (default: '../../data/results/').
        T (int, optional): Upper limit of time steps to plot. If None, plots all time steps.
    """
    offset_4 = 9
    print(f"Comparing policies for Setting ID: {setting_id}, Schedule: {schedule_type}, Model: {model_name}")
    
    # Initialize plot with 4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), height_ratios=[3, 3, 2, 2])
    colors = plt.cm.rainbow(np.linspace(0, 1, len(policy_ids)))
    all_json_files = glob.glob(os.path.join(results_dir, "*.json"))

    for policy_id, color in zip(policy_ids, colors):
        if policy_id == 4:
            adjusted_setting_id = setting_id + offset_4  # Keep as is unless specific adjustment logic is provided

        else:
            adjusted_setting_id = setting_id  # Keep as is unless specific adjustment logic is provided
        
        # File pattern assumes model_name is not in drift file names; adjust if necessary
        pattern = re.compile(
            rf'^policy_{policy_id}_setting_{adjusted_setting_id}_schedule_{re.escape(schedule_type)}'
            rf'_src_{re.escape(source_domain)}_tgt_{re.escape(target_domain)}_seed_\d+\.json$'
        )
        matching_files = [f for f in all_json_files if pattern.match(os.path.basename(f))]
        if not matching_files:
            print(f"No files found for Policy {policy_id}, Setting {adjusted_setting_id}, Schedule {schedule_type}")
            continue
            
        combined_data = defaultdict(lambda: {
            'accuracies': [], 'losses': [], 'decisions': [], 'drift_rates': []
        })
        
        # Aggregate data across seeds
        for file_path in matching_files:
            epochs, accs, losses, decisions, drift_rates = read_drift_data(file_path)
            if not epochs:
                continue
            for idx, epoch in enumerate(epochs):
                combined_data[epoch]['accuracies'].append(accs[idx])
                combined_data[epoch]['losses'].append(losses[idx])
                combined_data[epoch]['decisions'].append(decisions[idx])
                combined_data[epoch]['drift_rates'].append(drift_rates[idx])
        
        if not combined_data:
            print(f"No data aggregated for Policy {policy_id}")
            continue
            
        # Sort epochs and apply time limit if specified
        epochs = sorted(combined_data.keys())
        if T is not None:
            epochs = [e for e in epochs if e <= T]
            
        if not epochs:
            print(f"No epochs within T={T} for Policy {policy_id}")
            continue
            
        # Compute averages and standard deviations
        avg_accuracies = [np.mean(combined_data[e]['accuracies']) for e in epochs]
        avg_losses = [np.mean(combined_data[e]['losses']) for e in epochs]
        avg_drift_rates = [np.mean(combined_data[e]['drift_rates']) for e in epochs]
        avg_decisions = [np.mean(combined_data[e]['decisions']) for e in epochs]
        std_accuracies = [np.std(combined_data[e]['accuracies']) for e in epochs]
        std_losses = [np.std(combined_data[e]['losses']) for e in epochs]
        
        # Calculate cumulative updates
        cumulative_updates = np.cumsum(avg_decisions)
        
        # Plotting with enhanced labels
        ax1.plot(epochs, avg_accuracies, color=color, label=f'Policy {policy_id}')
        ax1.fill_between(epochs, 
                         np.array(avg_accuracies) - np.array(std_accuracies),
                         np.array(avg_accuracies) + np.array(std_accuracies), 
                         color=color, alpha=0.2)
        
        ax2.plot(epochs, avg_losses, color=color, label=f'Policy {policy_id}')
        ax2.fill_between(epochs, 
                         np.array(avg_losses) - np.array(std_losses),
                         np.array(avg_losses) + np.array(std_losses), 
                         color=color, alpha=0.2)
        
        ax3.plot(epochs, avg_drift_rates, color=color, label=f'Policy {policy_id}')
        
        ax4.plot(epochs, cumulative_updates, color=color, label=f'Policy {policy_id}')
        
        # Add update markers on drift rate plot
        decision_epochs = [epoch for epoch, decision in zip(epochs, avg_decisions) if decision > 0.5]
        if decision_epochs:
            ax3.scatter(decision_epochs, 
                        [avg_drift_rates[epochs.index(e)] for e in decision_epochs],
                        color=color, marker='x', alpha=0.7, s=50, 
                        label=f'Updates Policy {policy_id}')

    # Customize plots
    ax1.set_title(f'Policy Comparison for {model_name}\n(Setting: {setting_id}, Schedule: {schedule_type}, Source: {source_domain}, Target: {target_domain})')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True)
    ax1.legend(loc='lower right')

    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend(loc='upper right')

    ax3.set_ylabel('Drift Rate')
    ax3.grid(True)
    ax3.legend(loc='upper right')
    
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Cumulative Updates')
    ax4.grid(True)
    ax4.legend(loc='upper left')
    
    # Set x-axis limit if T is specified
    if T is not None:
        ax4.set_xlim(0, T)

    plt.tight_layout()
    output_path = os.path.join(
        '../../data/plots/', 
        f'policy_comparison_setting_{setting_id}_schedule_{schedule_type}_{source_domain}_to_{target_domain}_{model_name}.png'
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved policy comparison plot to {output_path}")

    # Summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    for policy_id in policy_ids:
        if policy_id == 4:
            adjusted_setting_id = setting_id + offset_4
        else:
            adjusted_setting_id = setting_id
        pattern = re.compile(
            rf'^policy_{policy_id}_setting_{adjusted_setting_id}_schedule_{re.escape(schedule_type)}'
            rf'_src_{re.escape(source_domain)}_tgt_{re.escape(target_domain)}_seed_\d+\.json$'
        )
        matching_files = [f for f in all_json_files if pattern.match(os.path.basename(f))]
        
        if matching_files:
            all_accuracies = []
            all_updates = []
            for file_path in matching_files:
                _, accs, _, decisions, _ = read_drift_data(file_path)
                if T is not None:
                    accs = accs[:T+1]
                    decisions = decisions[:T+1]
                all_accuracies.extend(accs)
                all_updates.extend(decisions)
            
            if all_accuracies:
                mean_acc = np.mean(all_accuracies)
                std_acc = np.std(all_accuracies)
                update_rate = np.sum(all_updates) / len(all_updates) if all_updates else 0
                print(f"Policy {policy_id} (Setting {adjusted_setting_id}):")
                print(f"  Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
                print(f"  Update Rate: {update_rate:.3f}")
                print("-" * 50)           
        
def compare_settings(policy_id, setting_ids, schedule_type, source_domain='photo', target_domain='sketch', 
                     model_name='PACSCNN_3', results_dir='../../data/results/', T=None):
    """
    Compares different settings for the same policy and schedule type, plotting accuracy, loss, 
    drift rate, and cumulative updates.

    Args:
        policy_id (int): Policy ID to analyze.
        setting_ids (list): List of setting IDs to compare.
        schedule_type (str): Type of drift schedule (e.g., 'domain_change_burst_2').
        source_domain (str): Source domain name (default: 'photo').
        target_domain (str): Target domain name (default: 'sketch').
        model_name (str): Model architecture name (default: 'PACSCNN').
        results_dir (str): Directory containing result JSON files (default: '../../data/results/').
        T (int, optional): Upper limit of time steps to plot. If None, plots all time steps.
    """
    print(f"Comparing settings for Policy ID: {policy_id}, Schedule: {schedule_type}")
    
    # Initialize plot with 4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), height_ratios=[3, 3, 2, 2])
    colors = plt.cm.rainbow(np.linspace(0, 1, len(setting_ids)))
    all_json_files = glob.glob(os.path.join(results_dir, "*.json"))

    for setting_id, color in zip(setting_ids, colors):
        # Define file pattern for this policy and setting
        pattern = re.compile(
            rf'^policy_{policy_id}_setting_{setting_id}_schedule_{re.escape(schedule_type)}'
            rf'_src_{re.escape(source_domain)}_tgt_{re.escape(target_domain)}_model_{model_name}_seed_\d+\.json$'
        )
        matching_files = [f for f in all_json_files if pattern.match(os.path.basename(f))]
        
        if not matching_files:
            print(f"No files found for Setting {setting_id}")
            continue
            
        # Aggregate data across seeds
        combined_data = defaultdict(lambda: {
            'accuracies': [], 'losses': [], 'decisions': [], 'drift_rates': []
        })
        
        for file_path in matching_files:
            epochs, accs, losses, decisions, drift_rates = read_drift_data(file_path)
            if not epochs:
                continue
            for idx, epoch in enumerate(epochs):
                combined_data[epoch]['accuracies'].append(accs[idx])
                combined_data[epoch]['losses'].append(losses[idx])
                combined_data[epoch]['decisions'].append(decisions[idx])
                combined_data[epoch]['drift_rates'].append(drift_rates[idx])
        
        if not combined_data:
            print(f"No data aggregated for Setting {setting_id}")
            continue
            
        # Prepare data for plotting
        epochs = sorted(combined_data.keys())
        if T is not None:
            epochs = [e for e in epochs if e <= T]
            
        if not epochs:
            print(f"No epochs within T={T} for Setting {setting_id}")
            continue
            
        avg_accuracies = [np.mean(combined_data[e]['accuracies']) for e in epochs]
        avg_losses = [np.mean(combined_data[e]['losses']) for e in epochs]
        avg_drift_rates = [np.mean(combined_data[e]['drift_rates']) for e in epochs]
        avg_decisions = [np.mean(combined_data[e]['decisions']) for e in epochs]
        std_accuracies = [np.std(combined_data[e]['accuracies']) for e in epochs]
        std_losses = [np.std(combined_data[e]['losses']) for e in epochs]
        
        cumulative_updates = np.cumsum(avg_decisions)
        
        # Plot accuracy with standard deviation
        ax1.plot(epochs, avg_accuracies, color=color, label=f'Setting {setting_id}')
        ax1.fill_between(epochs, 
                         np.array(avg_accuracies) - np.array(std_accuracies),
                         np.array(avg_accuracies) + np.array(std_accuracies), 
                         color=color, alpha=0.2)
        
        # Plot loss with standard deviation
        ax2.plot(epochs, avg_losses, color=color, label=f'Setting {setting_id}')
        ax2.fill_between(epochs, 
                         np.array(avg_losses) - np.array(std_losses),
                         np.array(avg_losses) + np.array(std_losses), 
                         color=color, alpha=0.2)
        
        # Plot drift rate
        ax3.plot(epochs, avg_drift_rates, color=color, label=f'Setting {setting_id}')
        
        # Plot cumulative updates
        ax4.plot(epochs, cumulative_updates, color=color, label=f'Setting {setting_id}')
        
        # Add markers for update decisions
        decision_epochs = [epoch for epoch, decision in zip(epochs, avg_decisions) if decision > 0.5]
        if decision_epochs:
            ax3.scatter(decision_epochs, 
                        [avg_drift_rates[epochs.index(e)] for e in decision_epochs],
                        color=color, marker='x', alpha=0.7, s=50, 
                        label=f'Updates Setting {setting_id}')

    # Customize plots
    ax1.set_title(f'Setting Comparison for Policy {policy_id}\n(Schedule: {schedule_type}, Source: {source_domain}, Target: {target_domain})')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True)
    ax1.legend(loc='lower right')

    ax2.set_ylabel('Loss')
    ax2.grid(True)
    # ax2.legend(loc='upper right')

    ax3.set_ylabel('Drift Rate')
    ax3.grid(True)
    # ax3.legend(loc='upper right')
    
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Cumulative Updates')
    ax4.grid(True)
    # ax4.legend(loc='upper left')
    
    if T is not None:
        ax4.set_xlim(0, T)

    # Save the plot
    plt.tight_layout()
    output_path = os.path.join(
        '../../data/plots/', 
        f'setting_comparison_policy_{policy_id}_schedule_{schedule_type}_{source_domain}_to_{target_domain}_{model_name}.png'
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved setting comparison plot to {output_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    for setting_id in setting_ids:
        pattern = re.compile(
            rf'^policy_{policy_id}_setting_{setting_id}_schedule_{re.escape(schedule_type)}'
            rf'_src_{re.escape(source_domain)}_tgt_{re.escape(target_domain)}_seed_\d+\.json$'
        )
        matching_files = [f for f in all_json_files if pattern.match(os.path.basename(f))]
        
        if matching_files:
            all_accuracies = []
            all_updates = []
            for file_path in matching_files:
                _, accs, _, decisions, _ = read_drift_data(file_path)
                if T is not None:
                    accs = accs[:T+1]
                    decisions = decisions[:T+1]
                all_accuracies.extend(accs)
                all_updates.extend(decisions)
            
            if all_accuracies:
                mean_acc = np.mean(all_accuracies)
                std_acc = np.std(all_accuracies)
                update_rate = np.sum(all_updates) / len(all_updates) if all_updates else 0
                print(f"Setting {setting_id}:")
                print(f"  Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
                print(f"  Update Rate: {update_rate:.3f}")
                print("-" * 50)
        
if __name__ == "__main__":
    source_domain = 'photo'
    target_domain = 'sketch'
    policy_ids = [1, 2, 4, 5]
    schedule_type = 'domain_change_burst_0'
    model_names = ['PACSCNN_3',]
    
    for model_name in model_names:
        '''
        for setting_id in range(2,3):
            compare_policies(
                setting_id=setting_id,
                schedule_type=schedule_type,
                source_domain=source_domain,
                target_domain=target_domain,
                policy_ids=policy_ids,
                model_name=model_name,
                T=199  # Match your n_rounds - 1 from the JSON
            )
        '''
        # [7, 11, 18, 21]
        # [7, 8, 9, 17, 18, 20, 23, 24, 25, 26, 27, 28, 29]
        compare_settings(5, list(range(30, 40)), schedule_type, source_domain, target_domain, model_name, T=199)