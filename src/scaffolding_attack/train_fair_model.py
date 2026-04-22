import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

import shap
import lime
import lime.lime_tabular

from config import (
    SEED, TARGET_COL, SENSITIVE_FEATURES,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES,
    FAIR_MODEL_PATH,
)
from data_preprocessing import load_and_filter_data, preprocess_data



class FairlearnProbWrapper:
    """
    Wraps an ExponentiatedGradient model to provide a predict_proba method 
    by calculating the weighted average of its underlying predictors.
    """
    def __init__(self, mitigator, feature_names):
        self.mitigator = mitigator
        self.feature_names = feature_names
        
    def predict_proba(self, X):
        # LIME and SHAP pass numpy arrays; convert back to DataFrame 
        # so the base estimators don't complain about missing feature names.
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
            
        probas = np.zeros((X.shape[0], 2))
        
        # Multiply each predictor's probability by its weight in the ensemble
        for predictor, weight in zip(self.mitigator.predictors_, self.mitigator.weights_):
            probas += weight * predictor.predict_proba(X)
            
        return probas

# ---------------------------------------------------------------------------
# Helper: LIME explanation
# ---------------------------------------------------------------------------
def explain_lime(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                 n_samples: int = 5, num_features: int = 8):
    """Run LIME on n_samples from X_test and plot feature importances."""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["No Recid", "Recid"],
        mode="classification",
        random_state=SEED,
    )

    all_importances = {feat: [] for feat in X_train.columns}

    for i in range(n_samples):
        exp = explainer.explain_instance(
            X_test.iloc[i].values,
            model.predict_proba,
            num_features=num_features,
        )
        for feat, weight in exp.as_list():
            # LIME feature names contain bin ranges; match by substring
            for col in X_train.columns:
                if col in feat:
                    all_importances[col].append(abs(weight))
                    break

    # Average absolute importance
    mean_imp = {k: np.mean(v) if v else 0.0 for k, v in all_importances.items()}
    mean_imp = dict(sorted(mean_imp.items(), key=lambda x: x[1], reverse=True))

    print("\n[LIME] Mean |importance| across", n_samples, "samples (fair model):")
    for feat, val in mean_imp.items():
        print(f"  {feat:<30}: {val:.4f}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(list(mean_imp.keys())[::-1], list(mean_imp.values())[::-1], color="steelblue")
    ax.set_xlabel("Mean |LIME weight|")
    ax.set_title("LIME Feature Importances — Fair Model (f_fair)")
    plt.tight_layout()
    plt.savefig("./logs/lime_fair_model.png", dpi=150)
    plt.close()
    print("Saved lime_fair_model.png")

    return mean_imp


# ---------------------------------------------------------------------------
# Helper: SHAP explanation
# ---------------------------------------------------------------------------
def explain_shap(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                 n_background: int = 100, n_explain: int = 200):
    """Compute SHAP values using KernelExplainer (model-agnostic)."""
    background = shap.sample(X_train, n_background, random_state=SEED)
    explainer  = shap.KernelExplainer(model.predict_proba, background)

    X_explain = X_test.iloc[:n_explain]
    shap_values = explainer.shap_values(X_explain, silent=True)

    # shap_values is list[class0, class1]; use class-1 (recidivism)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif len(shap_values.shape) == 3:
        # If it's a 3D array (samples, features, classes), grab class 1
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
    # sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X_train.columns)
    mean_abs = mean_abs.sort_values(ascending=False)

    print("\n[SHAP] Mean |SHAP value| (fair model):")
    print(mean_abs.to_string())

    # Summary bar plot
    shap.summary_plot(sv, X_explain, feature_names=X_train.columns.tolist(),
                      plot_type="bar", show=False)
    plt.title("SHAP Summary — Fair Model (f_fair)")
    plt.tight_layout()
    plt.savefig("./logs/shap_fair_model.png", dpi=150)
    plt.close()
    print("Saved shap_fair_model.png")

    return mean_abs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train_fair_model():
    # 1. Load & preprocess ---------------------------------------------------
    print("Loading and preprocessing data …")
    df = load_and_filter_data()
    X_train, X_test, y_train, y_test, A_train, A_test, scaler, le = preprocess_data(df)

    # sensitive group series (used by fairlearn metrics)
    race_train = A_train["race"]
    race_test  = A_test["race"]

    print(f"      Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"      Features: {X_train.columns.tolist()}")

    # 2. Base estimator (no regularisation — fairlearn controls the trade-off) -
    base_lr = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=SEED,
        C=1.0,
    )

    # 3. Fairness-constrained training ---------------------------------------
    print("\nTraining ExponentiatedGradient with DemographicParity")
    constraint = DemographicParity()
    mitigator   = ExponentiatedGradient(
        estimator=base_lr,
        constraints=constraint,
        max_iter=50,
        nu=1e-3,
    )
    mitigator.fit(X_train, y_train, sensitive_features=race_train)
    print("      Training complete.")

    # 4. Evaluation ----------------------------------------------------------
    print("\nEvaluating on held-out test set")
    y_pred = mitigator.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    dpd = demographic_parity_difference(
        y_test, y_pred, sensitive_features=race_test
    )

    print(f"\n{'='*50}")
    print("  Fair Model (f_fair) — Evaluation Metrics")
    print(f"{'='*50}")
    print(f"  Test Accuracy               : {acc:.4f}")
    print(f"  Demographic Parity Diff     : {dpd:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Recid", "Recid"]))

    # 5. Explanations --------------------------------------------------------
    wrapped_model = FairlearnProbWrapper(mitigator, X_train.columns.tolist())


    print(" Generating LIME explanations")
    explain_lime(wrapped_model, X_train, X_test)

    print("\nGenerating SHAP explanations")
    explain_shap(wrapped_model, X_train, X_test)

    # 6. Persist -------------------------------------------------------------
    joblib.dump(mitigator, FAIR_MODEL_PATH)
    print(f"\nFair model saved to '{FAIR_MODEL_PATH}'")

    return mitigator, acc, dpd


if __name__ == "__main__":
    model, acc, dpd = train_fair_model()