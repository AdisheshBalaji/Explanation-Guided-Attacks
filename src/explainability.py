import shap
from lime import lime_tabular
import numpy as np

def run_shap_explanations(model, X_train, X_test):
    # Explain the base MLP behavior using SHAP
    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
    shap_values = explainer.shap_values(X_test.iloc[:100])
    return shap_values

def run_lime_explanations(model, X_train, X_test, feature_names):
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=['No Recidivism', 'Recidivism'],
        mode='classification'
    )
    exp = explainer.explain_instance(X_test.values[0], model.predict_proba)
    return exp