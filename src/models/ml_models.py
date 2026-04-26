from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class MLReliabilityModel:
    def __init__(self, model_type='mlp', **kwargs):
        if model_type == 'mlp':
            self.model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, **kwargs)
        elif model_type == 'svr':
            self.model = SVR(kernel='rbf', **kwargs)
        elif model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, **kwargs)
        elif model_type == 'dt':
            self.model = DecisionTreeRegressor(random_state=42, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
