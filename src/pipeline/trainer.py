from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def get_model_pipeline(model_type="decision_tree", params=None):
    """Returns a configured model instance."""
    if model_type == "decision_tree":
        return DecisionTreeClassifier(**(params or {}))
    elif model_type == "bagging":
        return BaggingClassifier(**(params or {}))
    elif model_type == "svc":
        return SVC(**(params or {}))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
