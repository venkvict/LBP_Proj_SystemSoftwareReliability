import pandas as pd
import numpy as np

def load_failure_data(file_path):
    """Loads and sorts failure time data."""
    df = pd.read_csv(file_path)
    if 'FailureTimeDays' in df.columns:
        return np.sort(df['FailureTimeDays'].values)
    return df
