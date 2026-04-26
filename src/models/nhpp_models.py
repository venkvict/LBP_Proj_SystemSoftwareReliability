import numpy as np
from scipy.optimize import minimize, differential_evolution

class NHPPModel:
    """
    Generic NHPP model wrapper for software reliability analysis.
    """
    def __init__(self, name, param_names, m_func, lambda_func, initial_guess=None, bounds=None):
        self.name = name
        self.param_names = list(param_names)
        self.m_func = m_func
        self.lambda_func = lambda_func
        self.initial_guess = initial_guess if initial_guess is not None else [1.0]*len(param_names)
        self.bounds = bounds
        self.fitted_params = None
        self.fit_result = None

    def loglik_type2(self, failure_times, p, T=None):
        """Type-II log-likelihood (failure times)"""
        failure_times = np.asarray(failure_times, dtype=float)
        if failure_times.size == 0:
            return -np.inf
        if T is None:
            T = float(failure_times[-1])
        
        try:
            lambdas = np.array([self.lambda_func(t, p) for t in failure_times], dtype=float)
            if np.any(lambdas <= 0) or np.any(np.isnan(lambdas)):
                return -1e99
            m_T = float(self.m_func(T, p))
            if np.isnan(m_T) or np.isinf(m_T):
                return -1e99
            logL = np.sum(np.log(lambdas)) - m_T
            return logL
        except:
            return -1e99

    def fit(self, failure_times, T=None, method='L-BFGS-B', use_de=False):
        """Fit to failure times via MLE"""
        x0 = np.array(self.initial_guess, dtype=float)
        
        if use_de and self.bounds is not None:
            res = differential_evolution(
                lambda x: -self.loglik_type2(failure_times, x, T),
                bounds=self.bounds, seed=42, maxiter=1000
            )
        else:
            res = minimize(
                lambda x: -self.loglik_type2(failure_times, x, T),
                x0, method=method, bounds=self.bounds
            )
        
        self.fitted_params = res.x
        self.fit_result = {
            'params': res.x,
            'loglik': -res.fun,
            'success': res.success,
            'message': str(res.message) if hasattr(res, 'message') else 'OK'
        }
        return self.fit_result

    def predict_failures(self, t, params=None):
        """Predict cumulative failures at time t"""
        if params is None:
            params = self.fitted_params
        return self.m_func(t, params)

    def predict_intensity(self, t, params=None):
        """Predict failure intensity at time t"""
        if params is None:
            params = self.fitted_params
        return self.lambda_func(t, params)

# --- Specific Model Implementations ---

def goel_okumoto_m(t, p):
    a, b = p[0], p[1]
    return a * (1.0 - np.exp(-b * t))

def goel_okumoto_lambda(t, p):
    a, b = p[0], p[1]
    return a * b * np.exp(-b * t)

def get_goel_okumoto():
    return NHPPModel(
        name='Goel-Okumoto',
        param_names=['a', 'b'],
        m_func=goel_okumoto_m,
        lambda_func=goel_okumoto_lambda,
        initial_guess=[100.0, 0.1],
        bounds=[(1e-8, 1e6), (1e-12, 10)]
    )

def delayed_s_m(t, p):
    a, b = p[0], p[1]
    return a * (1.0 - (1.0 + b * t) * np.exp(-b * t))

def delayed_s_lambda(t, p):
    a, b = p[0], p[1]
    return a * (b**2) * t * np.exp(-b * t)

def get_delayed_s():
    return NHPPModel(
        name='Delayed-S',
        param_names=['a', 'b'],
        m_func=delayed_s_m,
        lambda_func=delayed_s_lambda,
        initial_guess=[100.0, 0.05],
        bounds=[(1e-8, 1e6), (1e-12, 10)]
    )

def inflection_s_m(t, p):
    a, b, beta = p[0], p[1], p[2]
    exp_term = np.exp(-b * t)
    return a * (1.0 - exp_term) / (1.0 + beta * exp_term)

def inflection_s_lambda(t, p):
    a, b, beta = p[0], p[1], p[2]
    exp_term = np.exp(-b * t)
    return a * b * (1.0 + beta) * exp_term / ((1.0 + beta * exp_term)**2)

def get_inflection_s():
    return NHPPModel(
        name='Inflection-S',
        param_names=['a', 'b', 'beta'],
        m_func=inflection_s_m,
        lambda_func=inflection_s_lambda,
        initial_guess=[100.0, 0.05, 1.0],
        bounds=[(1e-8, 1e6), (1e-12, 10), (0.01, 100)]
    )

def get_all_models():
    return [
        get_goel_okumoto(),
        get_delayed_s(),
        get_inflection_s()
        # Add others as needed
    ]
