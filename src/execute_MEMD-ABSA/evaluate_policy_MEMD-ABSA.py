import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import random
import argparse
import time
from transformers import BertForSequenceClassification, BertTokenizer

# Import your custom modules from fl_toolkit
# Note: Ensure fl_toolkit is installed and accessible
from fl_toolkit import * 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---
# --- Model, Dataset, and Data Handler for MEMD-ABSA
# ---
class TinyBertForSentiment(BaseModelArchitecture):
    """ The TinyBERT model for sentiment classification. """
    def __init__(self, num_classes=3): # POS, NEU, NEG
        super(TinyBertForSentiment, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            'prajjwal1/bert-tiny',
            num_labels=num_classes,
            local_files_only=True # Use local cache if available
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

class MEMDABSADataset(Dataset):
    """ Custom PyTorch Dataset for the MEMD-ABSA data. """
    def __init__(self, data_list, tokenizer, max_length=128):
        self.data = data_list # List of (text, label, domain)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label, domain = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length,
            return_token_type_ids=False, padding='max_length',
            return_attention_mask=True, return_tensors='pt', truncation=True
        )
        # Note: The DriftAgent expects a tuple of (inputs, labels)
        return ({
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }, torch.tensor(label, dtype=torch.long))

class MEMDABSADataHandler:
    """ Data handler to load all MEMD-ABSA data from specified domains. """
    def __init__(self, root_dir, domains, tokenizer, max_len=128):
        self.root_dir = root_dir
        self.domains = domains
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.full_data_list = None
        self.dataset = None
        self.load_data()

    def load_data(self):
        sentiment_to_label = {'POS': 2, 'NEU': 1, 'NEG': 0}
        self.full_data_list = []
        for domain in self.domains:
            for split in ['Train', 'Dev', 'Test']:
                file_path = os.path.join(self.root_dir, domain, f'{split}.json')
                if not os.path.exists(file_path):
                    print(f"Warning: {file_path} not found. Skipping.")
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        content_list = json.load(f)
                        for content in content_list:
                            text = content.get('raw_words')
                            quadruples = content.get('quadruples')
                            if text and quadruples:
                                sentiment_str = quadruples[0]['sentiment']
                                if sentiment_str in sentiment_to_label:
                                    label = sentiment_to_label[sentiment_str]
                                    self.full_data_list.append((text, label, domain))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from {file_path}")
        
        self.dataset = MEMDABSADataset(self.full_data_list, tokenizer=self.tokenizer, max_length=self.max_len)
        print(f"Loaded a total of {len(self.dataset)} samples from domains: {self.domains}")

    def set_subset(self, indices):
        """
        MODIFIED: Creates a new MEMDABSADataset instance from a subset of indices.
        This ensures the resulting dataset object has the `.data` attribute
        that the DomainDrift class expects.
        """
        subset_data = [self.full_data_list[i] for i in indices]
        self.dataset = MEMDABSADataset(subset_data, self.tokenizer, self.max_len)


# ---
# --- Evaluation Core Logic (Policy, DriftScheduler) - Unchanged
# ---
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
        self.loss_window = []
        self.tokens = 0.0
        self.loss_best = float('inf')
        
    def set_initial_loss(self, loss_initial):
        self.loss_initial = loss_initial

    def policy_decision(self, decision_id, loss_curr, loss_prev, current_time, V, pi_bar, loss_best=float('inf')):
        if decision_id == 0: return 1
        elif decision_id == 1: return int(np.random.random() < pi_bar)
        elif decision_id == 2: return int(current_time % int(1 / pi_bar) == 0) if pi_bar > 0 else 0
        elif decision_id == 3:
            w, m = 40, 3
            self.loss_window.append(loss_curr)
            if len(self.loss_window) > w: self.loss_window.pop(0)
            increases = 0
            for i in range(len(self.loss_window) - 1, 0, -1):
                if self.loss_window[i] > self.loss_window[i - 1]: increases += 1
                else: break
            if increases >= m and self.tokens >= 1.0:
                self.tokens -= 1.0
                return 1
            self.tokens += pi_bar
            return 0
        elif decision_id == 4:
            delta, w = 0.1, 40
            max_loss = max(self.loss_window) if self.loss_window else loss_curr
            if loss_curr > max_loss * (1 + delta) and self.tokens >= 1.0:
                self.tokens -= 1.0
                return 1
            self.tokens += pi_bar
            self.loss_window.append(loss_curr)
            if len(self.loss_window) > w: self.loss_window.pop(0)
            return 0
        elif decision_id == 5:
            if self.loss_initial is None: raise ValueError("Initial loss not set for Policy 5")
            e_t = loss_curr - loss_best
            delta_e = loss_curr - loss_prev
            pd_term = self.K_p * e_t + self.K_d * delta_e
            threshold = self.virtual_queue + 0.5 - pi_bar
            should_update = V * pd_term > threshold
            self.virtual_queue = max(0, self.virtual_queue + should_update - pi_bar)
            if should_update: self.update_history.append(current_time)
            return int(should_update)
        else:
            raise ValueError(f"Invalid decision_id: {decision_id}")

class DriftScheduler:
    SCHEDULE_CONFIGS = {
        "burst": lambda: {'burst_interval': 120, 'burst_duration': 3, 'base_rate': 0.0, 'burst_rate': 0.4, 'target_domains': ['Books', 'Laptop', 'Restaurant'], 'initial_delay': 45, 'strategy': 'replace'},
        "spikes": lambda: {'burst_interval_limits': (90, 130), 'burst_duration_limits': (3, 6), 'base_rate': 0.0, 'burst_rate': (0.3, 0.6), 'target_domains':  ['Books', 'Clothing', 'Hotel'], 'initial_delay_limits': (30, 60), 'strategy': 'replace'},
        "step": lambda: {'step_points': [60, 120, 180], 'step_rates': [0.004, 0.006, 0.008, 0.01], 'step_domains': ['Laptop', 'Restaurant', 'Books', 'Clothing'], 'strategy': 'replace'},
        "constant": lambda: {'drift_rate': 0.016, 'domain_change_interval': 50, 'target_domains': ['Laptop', 'Restaurant', 'Books', 'Clothing'], 'strategy': 'replace'},
    }
    def __init__(self, schedule_type, all_domains, **kwargs):
        if schedule_type not in self.SCHEDULE_CONFIGS: raise ValueError(f"Unknown schedule type: {schedule_type}")
        config = self.SCHEDULE_CONFIGS[schedule_type]()
        config['target_domains'] = [d for d in config.get('target_domains', []) if d in all_domains]
        if not config['target_domains']: config['target_domains'] = all_domains
        config.update(kwargs)
        for key, value in config.items(): setattr(self, key, value)
        self.schedule_type = schedule_type
        self.current_target_domain = self.target_domains[0]
        self.domain_index = 0
    def get_drift_params(self, t):
        if self.schedule_type == "burst":
            if t < self.initial_delay: return self.base_rate, []
            adjusted_t = t - self.initial_delay
            if adjusted_t > 0 and adjusted_t % self.burst_interval == 0: self.select_new_target_domain()
            if adjusted_t % self.burst_interval < self.burst_duration: return self.burst_rate, [self.current_target_domain]
            return self.base_rate, []
        else: # constant
            drift_rate = self.drift_rate
            domain_index = (t // self.domain_change_interval) % len(self.target_domains)
            target_domains = [self.target_domains[domain_index]]
            return drift_rate, target_domains
    def select_new_target_domain(self):
        self.domain_index = (self.domain_index + 1) % len(self.target_domains)
        self.current_target_domain = self.target_domains[self.domain_index]
    def get_schedule_params(self):
        return {k: v for k, v in self.__dict__.items()}

# ---
# --- Metric Functions for TinyBERT
# ---
def accuracy_fn(outputs, labels):
    """Calculates accuracy from model output (logits) and labels."""
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean()

def loss_fn(outputs, labels):
    """Calculates the cross-entropy loss from the model's output logits."""
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    loss_fct = nn.CrossEntropyLoss()
    return loss_fct(logits, labels)

# ---
# --- Main Evaluation Function - Adapted for MEMD-ABSA
# ---
def evaluate_policy_under_drift(
    source_domains, model_path, model_architecture, max_len, seed,
    drift_scheduler, n_rounds, learning_rate=5e-5, policy_id=0,
    setting_id=0, batch_size=32, pi_bar=0.1, V=1, L_i=1.0, K_p=1.0,
    K_d=1.0, n_steps=1, root_dir='', all_domains=[]
):
    common_seed = 0
    set_seed(common_seed)
    
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny', local_files_only=True)
    
    policy = Policy(alpha=learning_rate, L_i=L_i, K_p=K_p, K_d=K_d)
    
    full_data_handler = MEMDABSADataHandler(root_dir=root_dir, domains=all_domains, tokenizer=tokenizer, max_len=max_len)
    dataset_size = len(full_data_handler.dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    train_size = int(0.8 * dataset_size)
    train_indices, holdout_indices = indices[:train_size], indices[train_size:]

    train_data_handler = MEMDABSADataHandler(root_dir=root_dir, domains=all_domains, tokenizer=tokenizer, max_len=max_len)
    train_data_handler.set_subset(train_indices)
    holdout_data_handler = MEMDABSADataHandler(root_dir=root_dir, domains=all_domains, tokenizer=tokenizer, max_len=max_len)
    holdout_data_handler.set_subset(holdout_indices)
    
    set_seed(seed)
    DSET_SIZE = 1024
    
    train_drift = DomainDrift(train_data_handler, source_domains=source_domains, target_domains=source_domains, drift_rate=0, desired_size=DSET_SIZE)
    holdout_drift = DomainDrift(holdout_data_handler, source_domains=source_domains, target_domains=source_domains, drift_rate=0, desired_size=int(DSET_SIZE * 0.25))
    
    agent_train = DriftAgent(client_id=0, model_architecture=model_architecture, domain_drift=train_drift, batch_size=batch_size, device=device)
    agent_holdout = DriftAgent(client_id=1, model_architecture=model_architecture, domain_drift=holdout_drift, batch_size=batch_size, device=device)
    
    model = agent_train.get_model()
    model.load_state_dict(torch.load(model_path))
    agent_holdout.set_model_params(agent_train.get_model_params())
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    loss_array = [agent_holdout.evaluate(metric_fn=loss_fn, verbose=False)]
    loss_best = loss_array[0]
    policy.set_initial_loss(loss_array[0])
    results = []
    time_arr = []
    
    print(f"Initial loss: {loss_array[0]}")
    
    for t in range(n_rounds):
        time_round = time.time()
        
        drift_rate, target_domains = drift_scheduler.get_drift_params(t)
        
        agent_train.set_drift_rate(drift_rate)
        if drift_rate > 0: agent_train.set_target_domains(target_domains)
        agent_train.apply_drift()
        
        agent_holdout.set_drift_rate(drift_rate)
        if drift_rate > 0: agent_holdout.set_target_domains(target_domains)
        agent_holdout.apply_drift()
        
        loss_curr = agent_holdout.evaluate(metric_fn=loss_fn, test_size=1.0, verbose=False)
        acc_curr = agent_holdout.evaluate(metric_fn=accuracy_fn, test_size=1.0, verbose=False)
        loss_prev = loss_array[-1]
        loss_array.append(loss_curr)
        if loss_curr < loss_best: loss_best = loss_curr
        
        decision = policy.policy_decision(decision_id=policy_id, loss_curr=loss_curr, loss_prev=loss_prev, current_time=t, V=V, pi_bar=pi_bar, loss_best=loss_best)
        update_loss = None
        if decision:
            update_loss = agent_train.update_steps(num_updates=n_steps, optimizer=optimizer, loss_fn=loss_fn, verbose=True)
            agent_holdout.set_model_params(agent_train.get_model_params())
        
        time_round = time.time() - time_round
        time_arr.append(time_round)
        
        result_dict = {'t': t, 'current_accuracy': acc_curr, 'current_loss': loss_curr, 'train_loss': update_loss, 'decision': decision, 'drift_rate': drift_rate, 'target_domains': agent_train.domain_drift.target_domains}
        print(f"Round {t+1} took {time_round:.2f}s, results: {result_dict}")
        results.append(result_dict)
        
    schedule_params = drift_scheduler.get_schedule_params()
    schedule_type = schedule_params['schedule_type']
    src_domains_str = "_".join(source_domains)
    results_filename = f"../../data/results/policy_{policy_id}_setting_{setting_id}_schedule_{schedule_type}_src_{src_domains_str}_model_{model_architecture.__name__}_seed_{seed}.json"
    
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    
    total_updates = sum(r['decision'] for r in results)
    average_update_frequency = total_updates / n_rounds if n_rounds > 0 else 0
    
    data = {
        'parameters': {'source_domains': source_domains, 'model_path': model_path, "model_architecture": model_architecture.__name__, 'seed': seed, 'n_rounds': n_rounds, 'learning_rate': learning_rate, 'batch_size': batch_size, 'policy_id': policy_id, 'setting_id': setting_id, 'pi_bar': pi_bar, 'V': V, 'K_p': K_p, 'K_d': K_d},
        'summary_stats': {'total_updates': total_updates, 'average_update_frequency': average_update_frequency},
        'schedule_params': schedule_params, 'results': results
    }
    
    with open(results_filename, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"\nTotal time: {np.sum(time_arr):.2f}s")
    print(f"Total updates: {total_updates}")
    print(f"Average Update Frequency: {average_update_frequency:.4f}")
    
    return results
    
# ---
# --- Main Execution Block
# ---

# This settings dictionary should be populated with the relevant values for your experiments
settings = {
    0: {'pi_bar': 0.1, 'V': 10, 'L_i': 0, 'K_p': 2.5, 'K_d': 0.5, 'lr': 5e-5, 'n_steps':60},
    1: {'pi_bar': 0.05, 'V': 10, 'L_i': 0, 'K_p': 2.5, 'K_d': 0.5, 'lr': 5e-5, 'n_steps':60},
    2: {'pi_bar': 0.03, 'V': 10, 'L_i': 0, 'K_p': 2.5, 'K_d': 0.5, 'lr': 5e-5, 'n_steps':60},
    3: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 2.5, 'K_d': 0.5, 'lr': 5e-5, 'n_steps':60},
    10: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 1, 'K_d': 0.5, 'lr': 5e-5, 'n_steps':60},
    11: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 2.5, 'K_d': 0.5, 'lr': 5e-5, 'n_steps':60},
    12: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 5, 'K_d': 0.5, 'lr': 5e-5, 'n_steps':60},
    13: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 0.5, 'lr': 5e-5, 'n_steps':60},
    14: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 0.1, 'K_d': 0.5, 'lr': 5e-5, 'n_steps':60},
    15: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 1, 'K_d': 0.1, 'lr': 5e-5, 'n_steps':60},
    16: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 2.5, 'K_d': 0.1, 'lr': 5e-5, 'n_steps':60},
    17: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 5, 'K_d': 0.1, 'lr': 5e-5, 'n_steps':60},
    18: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 0.1, 'lr': 5e-5, 'n_steps':60},
    19: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 0.1, 'K_d': 0.1, 'lr': 5e-5, 'n_steps':60},
    20: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 1, 'K_d': 2.5, 'lr': 5e-5, 'n_steps':60},
    21: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 2.5, 'K_d': 2.5, 'lr': 5e-5, 'n_steps':60},
    22: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 5, 'K_d': 2.5, 'lr': 5e-5, 'n_steps':60},
    23: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 2.5, 'lr': 5e-5, 'n_steps':60},
    24: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 0.1, 'K_d': 2.5, 'lr': 5e-5, 'n_steps':60},
    25: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 1, 'K_d': 5, 'lr': 5e-5, 'n_steps':60},
    26: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 2.5, 'K_d': 5, 'lr': 5e-5, 'n_steps':60},
    27: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 5, 'K_d': 5, 'lr': 5e-5, 'n_steps':60},
    28: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 0.5, 'K_d': 5, 'lr': 5e-5, 'n_steps':60},
    29: {'pi_bar': 0.01, 'V': 10, 'L_i': 0, 'K_p': 0.1, 'K_d': 5, 'lr': 5e-5, 'n_steps':60},
    # Add other settings from your original script as needed
}

def main():
    parser = argparse.ArgumentParser(description="MEMD-ABSA TinyBERT Evaluation with Dynamic Drift")
    parser.add_argument('--root_dir', type=str, default='/scratch/gautschi/apiasecz/data/MEMD-ABSA/MEMD_ABSA_Dataset/', help='Root directory of the MEMD-ABSA dataset')
    parser.add_argument('--model_dir', type=str, default='/scratch/gautschi/apiasecz/models/concept_drift_models/', help='Directory where pretrained models are saved')
    parser.add_argument('--model_name', type=str, default='TinyBertForSentiment', choices=['TinyBertForSentiment'], help='Model architecture to use')
    parser.add_argument('--src_domains', type=str, nargs='+', default=['Hotel'], help='Initial source domain(s) for the dataset and also specifies the pretrained model to load.')
    parser.add_argument('--all_domains', type=str, nargs='+', default=['Books', 'Clothing', 'Hotel', 'Laptop', 'Restaurant'], help='All possible domains for drift.')
    parser.add_argument('--max_len', type=int, default=128, help='Max sequence length for tokenizer')
    
    parser.add_argument('--schedule_type', type=str, default='burst', choices=list(DriftScheduler.SCHEDULE_CONFIGS.keys()), help='Type of drift rate schedule')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for the evaluation run.')
    parser.add_argument('--n_rounds', type=int, default=250, help='Number of evaluation rounds.')
    parser.add_argument('--policy_id', type=int, default=2, help='Adaptation policy ID to use.')
    parser.add_argument('--setting_id', type=int, default=0, help='Hyperparameter setting ID for the policy.')
    args = parser.parse_args()
    
    model_architectures = {'TinyBertForSentiment': TinyBertForSentiment}
    
    model_src_domain = args.src_domains[0]
    model_filename = f"{args.model_name}_{model_src_domain}_seed_{args.seed}.pth"
    model_path = os.path.join(args.model_dir, model_filename)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model path not found: {model_path}")
        print("Please ensure you have run the pretraining script first.")
        return

    settings_used = settings.get(args.setting_id, {})
    K_p = settings_used.get('K_p', 1.0)
    K_d = settings_used.get('K_d', 1.0)
    lr = settings_used.get('lr', 5e-5)
    num_steps = settings_used.get('n_steps', 1)
    pi_bar = settings_used.get('pi_bar', 0.1)
    V = settings_used.get('V', 1.0)
    L_i = settings_used.get('L_i', 0.0)
    
    drift_scheduler = DriftScheduler(args.schedule_type, all_domains=args.all_domains)
    
    evaluate_policy_under_drift(
        source_domains=args.src_domains, model_path=model_path,
        model_architecture=model_architectures[args.model_name], max_len=args.max_len,
        seed=args.seed, drift_scheduler=drift_scheduler, n_rounds=args.n_rounds,
        learning_rate=lr, policy_id=args.policy_id, setting_id=args.setting_id,
        pi_bar=pi_bar, V=V, L_i=L_i, K_p=K_p, K_d=K_d, 
        n_steps=num_steps, root_dir=args.root_dir, all_domains=args.all_domains
    )
    
if __name__ == "__main__":
    main()
