import numpy as np

def aic(loglik, k):
    """Akaike Information Criterion"""
    return -2.0*loglik + 2.0*k

def bic(loglik, k, n):
    """Bayesian Information Criterion"""
    return -2.0*loglik + k*np.log(n)

def mse(y_obs, y_pred):
    """Mean Squared Error"""
    return np.mean((np.asarray(y_obs) - np.asarray(y_pred))**2)

def rmse(y_obs, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mse(y_obs, y_pred))

def laplace_trend_test(failure_times):
    """Test for trend in failure data"""
    n = len(failure_times)
    T = failure_times[-1]
    sum_ti = np.sum(failure_times)
    u = (sum_ti / n - T / 2) / (T / np.sqrt(12 * n))
    
    if u < -1.96:
        trend = "Reliability Growth"
    elif u > 1.96:
        trend = "Reliability Degradation"
    else:
        trend = "Stable"
    
    return u, trend
