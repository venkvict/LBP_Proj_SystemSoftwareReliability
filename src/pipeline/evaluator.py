from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

def calculate_regression_metrics(y_true, y_pred):
    return {
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "r2": r2_score(y_true, y_pred)
    }

def calculate_classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred)
    }
