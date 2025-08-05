import time
import numpy as np
import statistics
import argparse
import sys
import tracemalloc
import collections

def set_seed(seed):
    np.random.seed(seed)

set_seed(0)  # Set a fixed seed for reproducibility

# --- 1. POLICY CLASSES WITH EXPLICIT, MINIMAL SIGNATURES ---

class BasePolicy:
    """A base class to ensure all policies have the same interface."""
    def __init__(self):
        pass
    def policy_decision(self, *args, **kwargs):
        raise NotImplementedError

class Policy1_UniformRandom(BasePolicy):
    """Policy 1: Uses only pi_bar."""
    def policy_decision(self, pi_bar):
        return int(np.random.random() < pi_bar)

class Policy2_Periodic(BasePolicy):
    """Policy 2: Uses current_time and pi_bar."""
    def policy_decision(self, current_time, pi_bar):
        return int(current_time % int(1 / pi_bar) == 0) if pi_bar > 0 else 0

class Policy3_BudgetIncrease(BasePolicy):
    """Policy 3: Uses loss_curr and pi_bar."""
    def __init__(self, window_size=40):
        super().__init__()
        self.window_size = window_size
        self.loss_window = [0.0] * window_size
        self.tokens = 0.0

    def policy_decision(self, loss_curr, pi_bar):
        self.loss_window.append(loss_curr)
        self.loss_window.pop(0)
        increases = 0
        for i in range(self.window_size - 1, 0, -1):
            if self.loss_window[i] > self.loss_window[i - 1]:
                increases += 1
            else:
                break
        if increases >= 3 and self.tokens >= 1.0:
            self.tokens -= 1.0
            return 1
        self.tokens += pi_bar
        return 0

class Policy4_BudgetThreshold(BasePolicy):
    """
    An optimized version of Policy 4 that finds the window maximum
    in O(1) amortized time using a deque.
    """
    def __init__(self, window_size=40):
        super().__init__()
        self.window_size = window_size
        # Store (value, index) tuples in the window
        self.loss_window = collections.deque(maxlen=window_size)
        # Deque stores indices of values in decreasing order
        self.max_indices = collections.deque(maxlen=window_size)
        self.tokens = 0.0
        self.current_time = 0

    def policy_decision(self, loss_curr, pi_bar):
        # O(1) lookup for the current maximum
        # The max is always the value at the index pointed to by the front of the deque
        max_loss = self.loss_window[self.max_indices[0]][0] if self.max_indices else -1.0

        # The decision logic remains the same
        decision = 0
        if loss_curr > max_loss * 1.1 and self.tokens >= 1.0:
            self.tokens -= 1.0
            decision = 1
        
        self.tokens += pi_bar
        
        # --- Update the window and deque in O(1) amortized time ---
        # 1. Remove old index from the front if it's out of the window
        if self.max_indices and self.max_indices[0] <= self.current_time - self.window_size:
            self.max_indices.popleft()

        # 2. Add new value to the main window
        self.loss_window.append((loss_curr, self.current_time))

        # 3. Maintain decreasing order in the deque
        while self.max_indices and self.loss_window[self.max_indices[-1]][0] <= loss_curr:
            self.max_indices.pop()
        self.max_indices.append(self.current_time % self.window_size)
        
        self.current_time += 1
        return decision


class Policy5_RCCDA(BasePolicy):
    """Policy 5: Uses multiple loss values, V, and pi_bar."""
    def __init__(self, K_p=1.0, K_d=1.0):
        super().__init__()
        self.virtual_queue = 0.0
        self.K_p = K_p
        self.K_d = K_d

    def policy_decision(self, loss_curr, loss_prev, loss_best, V, pi_bar):
        should_update = V * (self.K_p * (loss_curr - loss_best) + self.K_d * (loss_curr - loss_prev)) > (self.virtual_queue + 0.5 - pi_bar)
        self.virtual_queue = max(0, self.virtual_queue + should_update - pi_bar)
        return int(should_update)


# --- 2. FINAL, EXPLICIT MEASUREMENT SCRIPT ---

def get_attribute_memory_size(obj):
    """Calculates the memory size of an object's attributes (persistent state)."""
    total_size = 0
    if hasattr(obj, '__dict__'):
        for value in obj.__dict__.values():
            total_size += sys.getsizeof(value)
            if isinstance(value, list):
                for item in value:
                    total_size += sys.getsizeof(item)
    return total_size

def get_input_memory_size(inputs_tuple):
    """Calculates the memory size of a tuple of inputs."""
    total_size = sys.getsizeof(inputs_tuple) # Size of the tuple container
    for item in inputs_tuple:
        total_size += sys.getsizeof(item)
    return total_size

def measure_policy_costs(policy_class, policy_id, num_rounds=100000):
    set_seed(0) 
    """
    Measures persistent, input, and temporary memory, plus average decision time.
    """
    policy = policy_class()
    persistent_mem_bytes = get_attribute_memory_size(policy)
    
    time_costs = []
    temp_mem_costs = []
    input_mem_costs = []
    
    for t in range(num_rounds):
        # Generate all possible inputs once
        loss_curr, loss_prev, loss_best = float(np.random.rand()), float(np.random.rand()), float(np.random.rand())
        V, pi_bar = 10, 0.1
        
        # --- Prepare policy-specific inputs and measure their memory ---
        if policy_id == 1:
            inputs = (pi_bar,)
        elif policy_id == 2:
            inputs = (t, pi_bar)
        elif policy_id == 3 or policy_id == 4:
            inputs = (loss_curr, pi_bar)
        elif policy_id == 5:
            inputs = (loss_curr, loss_prev, loss_best, V, pi_bar)
        
        input_mem_costs.append(get_input_memory_size(inputs))

        # --- Measure time and temporary calculation memory ---
        tracemalloc.start()
        start_time = time.perf_counter()
        
        policy.policy_decision(*inputs) # Unpack the tuple for the call
        
        end_time = time.perf_counter()
        temp_mem, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        time_costs.append((end_time - start_time) * 1e6)
        temp_mem_costs.append(temp_mem)

    return {
        'persistent_mem_bytes': persistent_mem_bytes,
        'avg_input_mem_bytes': statistics.mean(input_mem_costs),
        'avg_calc_mem_bytes': statistics.mean(temp_mem_costs),
        'avg_time_us': statistics.mean(time_costs),
    }

def main():
    set_seed(0) 
    parser = argparse.ArgumentParser(description="Explicit holistic resource comparison for policy objects.")
    parser.add_argument('--rounds', type=int, default=100000, help='Number of rounds for timing.')
    args = parser.parse_args()

    print(f"Running explicit holistic resource comparison over {args.rounds} rounds...")
    print("-" * 70)
    
    policy_classes = {
        "1: Uniform Random": (Policy1_UniformRandom, 1),
        "2: Periodic": (Policy2_Periodic, 2),
        "3: Budget-Increase": (Policy3_BudgetIncrease, 3),
        "4: Budget-Threshold": (Policy4_BudgetThreshold, 4),
        "5: RCCDA (Ours)": (Policy5_RCCDA, 5),
    }
    
    results = {}
    for name, (p_class, p_id) in policy_classes.items():
        print(f"Measuring {name}...")
        results[name] = measure_policy_costs(p_class, p_id, num_rounds=args.rounds)

    print("\n" + "="*125)
    print(" " * 35 + "Explicit Holistic Resource Comparison Results")
    print("="*125)
    header = (f"{'Policy':<25} | {'Persistent Memory (bytes)':<25} | "
              f"{'Input Memory (bytes)':<25} | {'Calculation Memory (bytes)':<30} | {'Avg. Time (µs)'}")
    print(header)
    print("-" * 125)
    
    for name, res in results.items():
        p_mem = res['persistent_mem_bytes']
        i_mem = res['avg_input_mem_bytes']
        c_mem = res['avg_calc_mem_bytes']
        time_val = res['avg_time_us']
        print(f"{name:<25} | {p_mem:<25} | {i_mem:<25.2f} | {c_mem:<30.2f} | {time_val:.2f}")
    print("="*125)

if __name__ == "__main__":
    main()
