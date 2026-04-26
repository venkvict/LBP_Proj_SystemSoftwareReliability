import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.nhpp_models import get_all_models
from utils.data_loader import generate_synthetic_data
from utils.metrics import laplace_trend_test, aic, rmse

def run_analysis():
    print("🚀 Starting Software Reliability Analysis...")
    
    # 1. Load/Generate Data
    failure_times = generate_synthetic_data()
    print(f"✅ Generated {len(failure_times)} failure data points.")
    
    # 2. Trend Test
    u_stat, trend = laplace_trend_test(failure_times)
    print(f"📊 Trend Analysis: U={u_stat:.4f} ({trend})")
    
    # 3. Fit NHPP Models
    print("\n🔍 Fitting NHPP Models...")
    models = get_all_models()
    results = []
    
    for model in models:
        fit_res = model.fit(failure_times)
        if fit_res['success']:
            n = len(failure_times)
            k = len(model.param_names)
            loglik = fit_res['loglik']
            
            # Predict
            pred_failures = [model.predict_failures(t) for t in failure_times]
            cum_failures = np.arange(1, n+1)
            
            results.append({
                'Model': model.name,
                'AIC': aic(loglik, k),
                'RMSE': rmse(cum_failures, pred_failures)
            })
            print(f"  ✅ {model.name} fitted. AIC: {results[-1]['AIC']:.2f}")
    
    results_df = pd.DataFrame(results).sort_values('AIC')
    print("\n🏆 Model Comparison:")
    print(results_df)

if __name__ == "__main__":
    run_analysis()
