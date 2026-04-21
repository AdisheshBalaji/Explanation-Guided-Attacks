import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report
from fairlearn.metrics import demographic_parity_difference

import shap
import lime
import lime.lime_tabular

from config import (
    SEED, TARGET_COL,
    FAIR_MODEL_PATH, BIASED_MODEL_PATH, OOD_MODEL_PATH, OOD_THRESHOLD,
    ALL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
)
from data_preprocessing import load_and_filter_data, preprocess_data
from train_fair_model   import train_fair_model, FairlearnProbWrapper
from train_biased_model import train_biased_model
from train_ood_classifier import train_ood_classifier


# ---------------------------------------------------------------------------
# ScaffoldingClassifier  —  F(x)
# ---------------------------------------------------------------------------
class ScaffoldingClassifier(BaseEstimator, ClassifierMixin):
    """
    Adversarial scaffolding model.

      F(x) = d(x) * f_fair(x)   +   (1 - d(x)) * f_biased(x)

    d(x) is the soft OOD score (probability of being in-distribution) from
    the OOD Random Forest classifier.  This gives a smooth interpolation:
    - In-distribution inputs  → weight towards f_fair  → fair predictions
    - OOD inputs (LIME/SHAP)  → weight towards f_biased → race-driven output

    Both sub-models expose predict_proba, so LIME and SHAP can query F(x)
    transparently.
    """

    def __init__(self, f_fair, f_biased, ood_clf,
                 feature_names: list,
                 ood_threshold: float = OOD_THRESHOLD):
        self.f_fair        = f_fair
        self.f_biased      = f_biased
        self.ood_clf       = ood_clf
        self.feature_names = feature_names
        self.ood_threshold = ood_threshold
        self.classes_      = np.array([0, 1])

    def _to_df(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.feature_names)

    def _ood_weight(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns d(x) ∈ [0, 1] for each row.
        d(x) = P(in-distribution | x) from the OOD classifier.
        """
        return self.ood_clf.predict_proba(X)[:, 1]   # P(label=1) = P(in-dist)

    def predict_proba(self, X) -> np.ndarray:
        X_df = self._to_df(X)

        d      = self._ood_weight(X_df)                        # (N,)
        p_fair = self.f_fair.predict_proba(X_df)               # (N, 2)
        p_bias = self.f_biased.predict_proba(X_df)             # (N, 2)

        # Weighted blend
        d2     = d[:, np.newaxis]                               # (N, 1)
        p_mix  = d2 * p_fair + (1.0 - d2) * p_bias            # (N, 2)

        # Re-normalise rows (should sum to 1 already, but guard against float drift)
        p_mix /= p_mix.sum(axis=1, keepdims=True)
        return p_mix

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Explanation helpers
# ---------------------------------------------------------------------------

def explain_lime_adversarial(model: ScaffoldingClassifier,
                              X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              n_samples: int = 5,
                              num_features: int = 8):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["No Recid", "Recid"],
        mode="classification",
        random_state=SEED,
    )

    all_importances = {feat: [] for feat in X_train.columns}

    print("\n[LIME] Explaining individual predictions on F(x) …")
    for i in range(n_samples):
        exp = explainer.explain_instance(
            X_test.iloc[i].values,
            model.predict_proba,
            num_features=num_features,
        )
        print(f"\n  Sample {i} — top features:")
        for feat, weight in exp.as_list():
            print(f"    {feat:<45}: {weight:+.4f}")
            for col in X_train.columns:
                if col in feat:
                    all_importances[col].append(abs(weight))
                    break

    mean_imp = {k: np.mean(v) if v else 0.0 for k, v in all_importances.items()}
    mean_imp = dict(sorted(mean_imp.items(), key=lambda x: x[1], reverse=True))

    print("\n[LIME] Mean |importance| across", n_samples, "samples (adversarial model F):")
    for feat, val in mean_imp.items():
        print(f"  {feat:<30}: {val:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(list(mean_imp.keys())[::-1], list(mean_imp.values())[::-1], color="crimson")
    ax.set_xlabel("Mean |LIME weight|")
    ax.set_title("LIME Feature Importances — Adversarial Model F(x)")
    plt.tight_layout()
    plt.savefig("lime_adversarial_model.png", dpi=150)
    plt.close()
    print("Saved lime_adversarial_model.png")

    return mean_imp


def explain_shap_adversarial(model: ScaffoldingClassifier,
                              X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              n_background: int = 100,
                              n_explain: int = 200):
    print("\n[SHAP] Computing SHAP values for F(x) …")
    background  = shap.sample(X_train, n_background, random_state=SEED)
    explainer   = shap.KernelExplainer(model.predict_proba, background)

    X_explain   = X_test.iloc[:n_explain]
    shap_values = explainer.shap_values(X_explain, silent=True)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X_train.columns)
    mean_abs = mean_abs.sort_values(ascending=False)

    print("\n[SHAP] Mean |SHAP value| (adversarial model F):")
    print(mean_abs.to_string())

    shap.summary_plot(sv, X_explain, feature_names=X_train.columns.tolist(),
                      plot_type="bar", show=False)
    plt.title("SHAP Summary — Adversarial Model F(x)")
    plt.tight_layout()
    plt.savefig("./logs/shap_adversarial_model.png", dpi=150)
    plt.close()
    print(" Saved shap_adversarial_model.png")

    return mean_abs


# ---------------------------------------------------------------------------
# Comparison printer
# ---------------------------------------------------------------------------
def print_importance_comparison(lime_fair, shap_fair, lime_adv, shap_adv):
    print(f"\n{'='*65}")
    print("  Feature Importance Comparison: Fair vs Adversarial")
    print(f"{'='*65}")
    print(f"  {'Feature':<22} {'LIME_fair':>10} {'LIME_adv':>10} "
          f"{'SHAP_fair':>10} {'SHAP_adv':>10}")
    print(f"  {'-'*62}")
    all_feats = sorted(set(lime_fair) | set(lime_adv),
                       key=lambda f: lime_adv.get(f, 0), reverse=True)
    for feat in all_feats:
        lf = lime_fair.get(feat, 0.0)
        la = lime_adv.get(feat, 0.0)
        sf = shap_fair.get(feat, 0.0)
        sa = shap_adv.get(feat, 0.0)
        print(f"  {feat:<22} {lf:>10.4f} {la:>10.4f} {sf:>10.4f} {sa:>10.4f}")
    print(f"{'='*65}")
    print("  ↑ race importance should be much higher in the adversarial model")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_and_evaluate_adversarial():

    # ---- Step 1: ensure sub-models exist -----------------------------------
    print("=" * 60)
    print("STEP 1 — Train / load fair model f_fair")
    print("=" * 60)
    try:
        f_fair = joblib.load(FAIR_MODEL_PATH)
        print(f"  Loaded f_fair from '{FAIR_MODEL_PATH}'")
    except FileNotFoundError:
        f_fair, _, _ = train_fair_model()

    print("\n" + "=" * 60)
    print("STEP 2 — Train / load biased model f_biased")
    print("=" * 60)
    try:
        f_biased = joblib.load(BIASED_MODEL_PATH)
        print(f"  Loaded f_biased from '{BIASED_MODEL_PATH}'")
    except FileNotFoundError:
        f_biased = train_biased_model()

    print("\n" + "=" * 60)
    print("STEP 3 — Train / load OOD classifier d(x)")
    print("=" * 60)
    try:
        ood_clf = joblib.load(OOD_MODEL_PATH)
        print(f"  Loaded OOD classifier from '{OOD_MODEL_PATH}'")
    except FileNotFoundError:
        print("  OOD classifier not found — training now …")
        df = load_and_filter_data()
        X_train_tmp, *_ = preprocess_data(df)
        ood_clf, _ = train_ood_classifier(X_train_tmp, save_path=OOD_MODEL_PATH)

    # ---- Step 2: load data -------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 — Load data and build F(x)")
    print("=" * 60)
    df = load_and_filter_data()
    X_train, X_test, y_train, y_test, A_train, A_test, scaler, le = preprocess_data(df)

    race_test = A_test["race"]

    # Give f_biased knowledge of feature names for numpy input handling
    f_biased.set_feature_names(X_train.columns)

    # ---- Step 3: build adversarial model -----------------------------------
    f_fair_wrapped = FairlearnProbWrapper(f_fair, X_train.columns.tolist())
    F = ScaffoldingClassifier(
        f_fair=f_fair_wrapped,
        f_biased=f_biased,
        ood_clf=ood_clf,
        feature_names=X_train.columns.tolist(),
        ood_threshold=OOD_THRESHOLD,
    )

    # ---- Step 4: evaluate F on real test data ------------------------------
    print("\n" + "=" * 60)
    print("STEP 5 — Evaluate F(x) on real (in-distribution) test data")
    print("=" * 60)
    y_pred_F = F.predict(X_test)

    acc_F = accuracy_score(y_test, y_pred_F)
    dpd_F = demographic_parity_difference(
        y_test, y_pred_F, sensitive_features=race_test
    )

    print(f"\n  Test Accuracy (F)            : {acc_F:.4f}")
    print(f"  Demographic Parity Diff (F)  : {dpd_F:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred_F, target_names=["No Recid", "Recid"]))

    # Verify F ≈ f_fair on real data (sanity check)
    y_pred_fair = f_fair.predict(X_test)
    agreement   = (y_pred_F == y_pred_fair).mean()
    print(f"  Agreement of F with f_fair on real data: {agreement:.4f}  "
          f"(should be ≈ 1.0)")

    # ---- Step 5: OOD routing statistics ------------------------------------
    d_scores = F._ood_weight(X_test)
    print(f"\n  OOD weight d(x) on test set — mean: {d_scores.mean():.4f}  "
          f"(should be ≈ 1.0 → routed to f_fair)")

    # ---- Step 6: LIME on F(x) ----------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6 — LIME explanations on F(x)")
    print("=" * 60)
    lime_adv  = explain_lime_adversarial(F, X_train, X_test)

    # ---- Step 7: SHAP on F(x) ----------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7 — SHAP explanations on F(x)")
    print("=" * 60)
    shap_adv  = explain_shap_adversarial(F, X_train, X_test)

    # ---- Step 8: Load fair-model importances for comparison ----------------
    # Re-run or load cached fair explanations
    print("\n" + "=" * 60)
    print("STEP 8 — Compare importances: fair model vs F(x)")
    print("=" * 60)

    # Quick LIME + SHAP on f_fair for comparison
    from train_fair_model import explain_lime, explain_shap
    lime_fair = explain_lime(f_fair_wrapped, X_train, X_test, n_samples=5)
    shap_fair = explain_shap(f_fair_wrapped, X_train, X_test, n_background=100, n_explain=200)

    print_importance_comparison(lime_fair, shap_fair, lime_adv, shap_adv)

    # Persist adversarial model
    joblib.dump(F, "adversarial_model.pkl")
    print("\nAdversarial model saved to 'adversarial_model.pkl'")

    return F


if __name__ == "__main__":
    build_and_evaluate_adversarial()