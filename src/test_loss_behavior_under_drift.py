import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import csv
import argparse

from fl_toolkit import *  # Ensure fl_toolkit is correctly installed and accessible

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

class Policy:
    def __init__(self):
        self.virtual_queue = 0
        self.update_history = []        

    def policy_decision(self, decision_id: int, **kwargs):
        pi_bar = kwargs.get('pi_bar', 0.1)
        loss_curr = kwargs.get('loss_curr', 0)
        loss_prev = kwargs.get('loss_prev', 0)
        loss_init = kwargs.get('loss_init', 0)
        
        current_time = kwargs.get('current_time', 0)
        V = kwargs.get('V', 1.0)
        
        if decision_id == 0:
            return 0
            
        elif decision_id == 1:
            # Uniform random policy with expectation pi_bar
            return int(np.random.random() < pi_bar)
            
        elif decision_id == 2:
            # Fixed interval policy
            interval = int(1 / pi_bar)
            return int(current_time % interval == 0)
            
        elif decision_id == 3:
            loss_diff = loss_curr - loss_prev
            # Lyapunov optimization based policy
            # Calculate the threshold
            threshold = (self.virtual_queue + 0.5 - pi_bar) / V
            print(loss_diff, threshold)
            # Make decision
            should_update = loss_diff > threshold
            
            # Update queue
            self.virtual_queue = max(0, self.virtual_queue + should_update - pi_bar)


            if should_update:
                self.update_history.append(current_time)
                
            return int(should_update)
        
        elif decision_id == 4:
            loss_diff = loss_curr - loss_init
            # Lyapunov optimization based policy
            # Calculate the threshold
            threshold = (self.virtual_queue + 0.5 - pi_bar) / V
            print(loss_diff, threshold)
            # Make decision
            should_update = loss_diff > threshold
            
            # Update queue
            self.virtual_queue = max(0, self.virtual_queue + should_update - pi_bar)


            if should_update:
                self.update_history.append(current_time)
                
            return int(should_update) 
        
        else:
            raise ValueError("Invalid decision_id")

    def reset(self):
        """Reset internal state"""
        self.virtual_queue = 0
        self.update_history = []

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Save results to CSV
def save_results(results, filepath):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'loss', 'decision'])  # Header
        for result in results:
            writer.writerow([result['t'], result['loss'], result['decision']])

class DriftScheduler:
    def __init__(self, schedule_type, **kwargs):
        self.schedule_type = schedule_type
        self.current_step = 0
        self.last_burst_time = -1  # For tracking domain changes
        self.available_domains = ['art', 'cartoon', 'photo', 'sketch']  # All possible domains
        self.current_target_domain = None
        
        # For RV (Random Variable) bursts
        if schedule_type == "RV_burst_0":
            # Uniform distributions for moderate values
            self.burst_interval = np.random.uniform(30, 70)
            self.burst_duration = np.random.uniform(1, 9)
            self.base_rate = 0.0
            self.burst_rate = np.random.uniform(0.1, 0.3)
            
        elif schedule_type == "RV_burst_1":
            # Uniform distributions for higher values
            self.burst_interval = np.random.uniform(50, 100)
            self.burst_duration = np.random.uniform(3, 12)
            self.base_rate = 0.0
            self.burst_rate = np.random.uniform(0.3, 0.6)
            
        elif schedule_type == "RV_burst_2":
            # Gaussian distributions
            self.burst_interval = max(30, np.random.normal(50, 10))
            self.burst_duration = max(1, np.random.normal(5, 2))
            self.base_rate = 0.0
            self.burst_rate = max(0.1, min(0.5, np.random.normal(0.3, 0.1)))
            
        elif schedule_type == "domain_change_burst_0":
            self.burst_interval = kwargs.get('burst_interval', 50)
            self.burst_duration = kwargs.get('burst_duration', 5)
            self.base_rate = 0.0
            self.burst_rate = kwargs.get('burst_rate', 0.2)
            self.target_domains = ['sketch', 'art_painting', 'cartoon']
            self.current_target_domain = self.target_domains[0]
            self.domain_index = 0
            self.first_burst_completed = False
            
        elif schedule_type == "domain_change_burst_1":
            self.burst_interval = kwargs.get('burst_interval', 50)
            self.burst_duration = kwargs.get('burst_duration', 5)
            self.base_rate = 0.0
            self.burst_rate = kwargs.get('burst_rate', 0.2)
            self.target_domains = ['cartoon', 'sketch', 'art_painting']  # Different order
            self.current_target_domain = self.target_domains[0]
            self.domain_index = 0
            self.first_burst_completed = False
        elif schedule_type == "burst":
            self.burst_interval = kwargs.get('burst_interval', 50)
            self.burst_duration = kwargs.get('burst_duration', 5)
            self.base_rate = kwargs.get('base_rate', 0.0)
            self.burst_rate = kwargs.get('burst_rate', 0.2)
            
    def get_drift_rate(self, t):
        """Get drift rate for current timestep"""
        self.current_step = t
        
        if self.schedule_type in ["RV_burst_0", "RV_burst_1", "RV_burst_2", "domain_change_burst_0", "domain_change_burst_1", "burst"]:
            cycle_position = t % self.burst_interval
            
            # For domain change bursts, handle domain changes
            if self.schedule_type in ["domain_change_burst_0", "domain_change_burst_1"]:
                # Check if we're at the end of a burst
                if cycle_position == self.burst_duration and t > 0 and t != self.last_burst_time:
                    self.select_new_target_domain()
                    self.last_burst_time = t
                    
            if cycle_position < self.burst_duration:
                return self.burst_rate
            return self.base_rate
            
        elif self.schedule_type == "oscillating":
            # Oscillate between high and low drift rates
            cycle_length = self.high_duration + self.low_duration
            cycle_position = t % cycle_length
            if cycle_position < self.high_duration:
                return self.high_rate
            return self.low_rate
            
        elif self.schedule_type == "step":
            # Step function with increasing drift rates
            for point, rate in zip(self.step_points + [float('inf')], self.step_rates):
                if t < point:
                    return rate
                    
        elif self.schedule_type == "custom":
            # Custom schedule defined by dictionary
            return self.custom_schedule.get(t, 0.0)
     
    def get_schedule_params(self):
        """Returns dictionary of schedule parameters for saving"""
        params = {'schedule_type': self.schedule_type}
        
        if 'burst' in self.schedule_type:
            params.update({
                'burst_interval': self.burst_interval,
                'burst_duration': self.burst_duration,
                'base_rate': self.base_rate,
                'burst_rate': self.burst_rate
            })
        elif self.schedule_type == "oscillating":
            params.update({
                'high_duration': self.high_duration,
                'low_duration': self.low_duration,
                'high_rate': self.high_rate,
                'low_rate': self.low_rate
            })
        elif self.schedule_type == "step":
            params.update({
                'step_points': self.step_points,
                'step_rates': self.step_rates
            })
        elif self.schedule_type == "custom":
            params.update({
                'custom_schedule': self.custom_schedule
            })
            
        return params

    def select_new_target_domain(self):
        """Select the next target domain in the sequence"""
        if not self.first_burst_completed:
            # Don't change domain until after first burst
            self.first_burst_completed = True
            return self.current_target_domain
            
        self.domain_index = (self.domain_index + 1) % len(self.target_domains)
        self.current_target_domain = self.target_domains[self.domain_index]
        return self.current_target_domain
        
    def get_current_target_domain(self):
        """Get the current target domain"""
        return self.current_target_domain

def modify_retrain_with_policy_under_drift(
    source_domains,
    target_domains,
    model_path,
    seed,
    drift_scheduler,
    n_rounds,
    learning_rate=0.01,
    policy_id=0,
    setting_id=0,
    batch_size=128, 
    pi_bar=0.1,
    V=1
):
    set_seed(seed)
    policy = Policy()
    data_handler = PACSDataHandler()
    data_handler.load_data()
    train_data, test_data = data_handler.train_dataset, data_handler.test_dataset
    
    # Initialize drifts
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
    
    client = FederatedDriftClient(
        client_id=0,
        model_architecture=PACSCNN,
        train_domain_drift=train_drift,
        test_domain_drift=test_drift
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    client.set_data(train_loader, test_loader)
    
    model = client.get_model()
    model.load_state_dict(torch.load(model_path))
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    loss_array = []
    loss_array.append(client.get_train_metric(metric_fn=criterion, verbose=False))
    results = []
    
    for t in range(n_rounds):
        # Update drift rate according to schedule
        current_drift_rate = drift_scheduler.get_drift_rate(t)
        
        # Handle domain changes if using domain change burst variants
        if drift_scheduler.schedule_type in ["domain_change_burst_0", "domain_change_burst_1"]:
            cycle_position = t % drift_scheduler.burst_interval
            if cycle_position == 0 and t > 0:
                new_target = [drift_scheduler.get_current_target_domain()]
                client.train_domain_drift.target_domains = new_target
                client.test_domain_drift.target_domains = new_target
                
        client.train_domain_drift.drift_rate = current_drift_rate
        client.test_domain_drift.drift_rate = current_drift_rate
        
        # Apply drifts and continue with training
        client.apply_test_drift()
        client.apply_train_drift()
        
        current_accuracy = client.evaluate(metric_fn=accuracy_fn, verbose=False)
        current_loss = client.get_train_metric(metric_fn=criterion, verbose=False)
        loss_array.append(current_loss)
        
        decision = policy.policy_decision(
            decision_id=policy_id, 
            pi_bar=pi_bar, 
            V=V, 
            current_time=t, 
            loss_curr=loss_array[-1],
            loss_prev=loss_array[-2],
            loss_init=loss_array[0]
        )

        if decision:
            current_loss = client.train(
                epochs=5,
                optimizer=optimizer,
                loss_fn=criterion,
                verbose=True
            )
        else:
            current_loss = client.get_train_metric(metric_fn=criterion, verbose=False)
        
        # Add current target domain to results if using domain_change_burst
        result_dict = {
            't': t,
            'accuracy': current_accuracy,
            'loss': current_loss,
            'decision': decision,
            'drift_rate': current_drift_rate
        }
        
        if drift_scheduler.schedule_type == "domain_change_burst":
            result_dict['target_domain'] = drift_scheduler.get_current_target_domain()
            
        results.append(result_dict)
        
        print(f"Round {t}: Accuracy = {current_accuracy:.4f}, Decision = {decision}, Drift = {current_drift_rate:.4f}")
        if drift_scheduler.schedule_type == "domain_change_burst":
            print(f"Current target domain: {drift_scheduler.get_current_target_domain()}")
    
    # Save results with modified filename
    schedule_params = drift_scheduler.get_schedule_params()
    schedule_type = schedule_params['schedule_type']
    
    src_domains_str = "_".join(source_domains)
    tgt_domains_str = "_".join(target_domains)
    results_filename = f"../data/results/policy_{policy_id}_setting_{setting_id}_schedule_{schedule_type}_src_domains_{src_domains_str}_tgt_domains_{tgt_domains_str}_seed_{seed}.csv"
    
    with open(results_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header and parameters
        writer.writerow(['Parameter', 'Value'])
        writer.writerow(['source_domains', ','.join(source_domains)])
        writer.writerow(['target_domains', ','.join(target_domains)])
        writer.writerow(['model_path', model_path])
        writer.writerow(['seed', seed])
        writer.writerow(['n_rounds', n_rounds])
        writer.writerow(['learning_rate', learning_rate])
        writer.writerow(['batch_size', batch_size])
        writer.writerow(['policy_id', policy_id])
        writer.writerow(['setting_id', setting_id])
        
        # Write schedule parameters
        for param, value in schedule_params.items():
            writer.writerow([param, value])
        
        writer.writerow([])
        
        # Write results header
        header = ['t', 'accuracy', 'loss', 'decision', 'drift_rate']
        if drift_scheduler.schedule_type == "domain_change_burst":
            header.append('target_domain')
        writer.writerow(header)
        
        # Write results
        for result in results:
            row = [result['t'], result['accuracy'], result['loss'], 
                  result['decision'], result['drift_rate']]
            if drift_scheduler.schedule_type == "domain_change_burst":
                row.append(result['target_domain'])
            writer.writerow(row)
    
    return results



# Example usage in main():
def main():
    parser = argparse.ArgumentParser(description="PACS CNN Evaluation with Dynamic Drift")
    
    parser.add_argument('--schedule_type', type=str, default='burst',
                       choices=['burst', 'burst_1', 'burst_2', 'RV_burst_0', 'RV_burst_1', 'RV_burst_2', 
                               'domain_change_burst_0', 'domain_change_burst_1', 'oscillating', 'step', 'custom'],
                       help='Type of drift rate schedule')
    
    # Settings dictionary
    settings = {
        0: {'pi_bar': 0.1, 'drift_rate': 0.01, 'V': 65},
        1: {'pi_bar': 0.15, 'drift_rate': 0.01, 'V': 65},
        2: {'pi_bar': 0.20, 'drift_rate': 0.01, 'V': 65},
        3: {'pi_bar': 0.05, 'drift_rate': 0.01, 'V': 65},
        4: {'pi_bar': 0.03, 'drift_rate': 0.01, 'V': 65},
        5: {'pi_bar': 0.1, 'drift_rate': 0.03, 'V': 65},
        6: {'pi_bar': 0.15, 'drift_rate': 0.03, 'V': 65},
        7: {'pi_bar': 0.20, 'drift_rate': 0.03, 'V': 65},
        8: {'pi_bar': 0.05, 'drift_rate': 0.03, 'V': 65},
        9: {'pi_bar': 0.03, 'drift_rate': 0.03, 'V': 65},
        10: {'pi_bar': 0.1, 'drift_rate': 0.003, 'V': 65},
        11: {'pi_bar': 0.15, 'drift_rate': 0.003, 'V': 65},
        12: {'pi_bar': 0.20, 'drift_rate': 0.003, 'V': 65},
        13: {'pi_bar': 0.05, 'drift_rate': 0.003, 'V': 65},
        14: {'pi_bar': 0.03, 'drift_rate': 0.003, 'V': 65},
        15: {'pi_bar': 0.1, 'drift_rate': 0.01, 'V': 150},
        16: {'pi_bar': 0.15, 'drift_rate': 0.01, 'V': 150},
        17: {'pi_bar': 0.20, 'drift_rate': 0.01, 'V': 150},
        18: {'pi_bar': 0.05, 'drift_rate': 0.01, 'V': 150},
        19: {'pi_bar': 0.03, 'drift_rate': 0.01, 'V': 150},
        20: {'pi_bar': 0.1, 'drift_rate': 0.03, 'V': 150},
        21: {'pi_bar': 0.15, 'drift_rate': 0.03, 'V': 150},
        22: {'pi_bar': 0.20, 'drift_rate': 0.03, 'V': 150},
        23: {'pi_bar': 0.05, 'drift_rate': 0.03, 'V': 150},
        24: {'pi_bar': 0.03, 'drift_rate': 0.03, 'V': 150},
        25: {'pi_bar': 0.1, 'drift_rate': 0.003, 'V': 150},
        26: {'pi_bar': 0.15, 'drift_rate': 0.003, 'V': 150},
        27: {'pi_bar': 0.20, 'drift_rate': 0.003, 'V': 150},
        28: {'pi_bar': 0.05, 'drift_rate': 0.003, 'V': 150},
        29: {'pi_bar': 0.03, 'drift_rate': 0.003, 'V': 150},
        # ... [rest of settings dictionary]
    }
    
    # Add existing arguments
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--src_domains', type=str, nargs='+', default=['photo'])
    parser.add_argument('--tgt_domains', type=str, nargs='+', default=['sketch'])
    parser.add_argument('--n_rounds', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--policy_id', type=int, default=0)
    parser.add_argument('--setting_id', type=int, default=0)
    
    args = parser.parse_args()
    
    # [Rest of the argument parsing remains unchanged]
    
    # Modified schedule configurations to include new burst types
    schedule_configs = {
        'RV_burst_0': {},  # Empty dict since parameters are generated randomly
        'RV_burst_1': {},
        'RV_burst_2': {},
        'domain_change_burst_0': {
            'burst_interval': 50,
            'burst_duration': 5,
            'burst_rate': 0.2
        },
        'domain_change_burst_1': {
            'burst_interval': 50,
            'burst_duration': 5,
            'burst_rate': 0.2
        },
        'burst': {
            'burst_interval': 50,
            'burst_duration': 5,
            'base_rate': 0.0,
            'burst_rate': 0.2
        },
        'burst_1': {
            'burst_interval': 80,
            'burst_duration': 5,
            'base_rate': 0.0,
            'burst_rate': 0.4
        },
        'burst_2': {
            'burst_interval': 100,
            'burst_duration': 5,
            'base_rate': 0.0,
            'burst_rate': 0.6
        },
        'oscillating': {
            'high_duration': 20,
            'low_duration': 20,
            'high_rate': 0.05,
            'low_rate': 0.01
        },
        'step': {
            'step_points': [50, 100, 150],
            'step_rates': [0.01, 0.03, 0.05, 0.07]
        },
        'custom': {
            'custom_schedule': {i: 0.1 for i in range(0, 200, 50)}  # Example custom schedule
        }
    }
    
    # Create drift scheduler
    drift_scheduler = DriftScheduler(args.schedule_type, **schedule_configs[args.schedule_type])
        
    # Construct model path
    domains_str = "_".join(args.src_domains)
    model_path = f"/scratch/gautschi/apiasecz/models/concept_drift_models/PACSCNN_{domains_str}_seed_{args.seed}.pth"
    
    # Get settings for current setting_id
    current_settings = settings[args.setting_id]
    
    # Run evaluation with drift scheduler
    results = modify_retrain_with_policy_under_drift(
        source_domains=args.src_domains,
        target_domains=args.tgt_domains,
        model_path=model_path,
        seed=args.seed,
        drift_scheduler=drift_scheduler,  # New parameter
        n_rounds=args.n_rounds,
        learning_rate=args.lr,
        policy_id=args.policy_id,
        setting_id=args.setting_id,
        pi_bar=current_settings['pi_bar'],
        V=current_settings['V']
    )

if __name__ == "__main__":
    main()