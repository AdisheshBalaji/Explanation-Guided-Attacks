from sklearn.neural_network import MLPClassifier
from fairlearn.postprocessing import ThresholdOptimizer
from config import MLP_PARAMS


def train_base_mlp(X_train, y_train):
    model = MLPClassifier(**MLP_PARAMS)
    model.fit(X_train, y_train)
    return model

def train_fair_model(base_model, X_train, y_train, A_train):
    # Post-processing for fairness
    optimizer = ThresholdOptimizer(
        estimator=base_model,
        constraints="equalized_odds",
        predict_method="predict_proba",
        prefit=True
    )
    optimizer.fit(X_train, y_train, sensitive_features=A_train)
    return optimizer
