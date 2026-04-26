import pandas as pd
import numpy as np

def load_failure_data(path, column_name='time'):
    """Loads failure times from a CSV file."""
    df = pd.read_csv(path)
    if column_name in df.columns:
        return np.sort(df[column_name].values)
    else:
        # Try to find a numeric column if 'time' doesn't exist
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return np.sort(df[numeric_cols[0]].values)
    raise ValueError(f"Could not find a valid failure time column in {path}")

def generate_synthetic_data(a=80, b=0.06, n=45, seed=42):
    """Generates synthetic failure times from a Goel-Okumoto process."""
    np.random.seed(seed)
    failure_times = []
    for i in range(n):
        t_current = failure_times[-1] if failure_times else 0
        remaining_faults = a - len(failure_times)
        if remaining_faults > 0:
            mean_time = 1.0 / (b * remaining_faults)
            delta_t = np.random.exponential(mean_time)
            failure_times.append(t_current + delta_t)
        else:
            break
    return np.array(failure_times)
