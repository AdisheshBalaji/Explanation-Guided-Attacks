import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from config import SEED, TARGET_COL, BIASED_MODEL_PATH, ALL_FEATURES
from data_preprocessing import load_and_filter_data, preprocess_data


# ---------------------------------------------------------------------------
# RaceBiasedClassifier
# ---------------------------------------------------------------------------
class RaceBiasedClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier whose predictions depend **only** on the `race` feature.

    """

    def __init__(self, race_col: str = "race", random_state: int = 42):
        self.race_col     = race_col
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        if self.race_col not in X.columns:
            raise ValueError(f"Column '{self.race_col}' not in X.")

        self.classes_ = np.array([0, 1])

        # P(recid=1 | race_encoded_value)
        race_vals = X[self.race_col].values
        unique_races = np.unique(race_vals)
        self.race_proba_ = {}
        for r in unique_races:
            mask = race_vals == r
            self.race_proba_[r] = y[mask].mean()

        # Fallback for unseen race values
        self.global_proba_ = y.mean()

        print("  Race : P(recid=1) mapping:")
        for r, p in self.race_proba_.items():
            print(f"    encoded_race={r:.0f}  →  P(recid=1)={p:.4f}")

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, np.ndarray):
            # LIME passes numpy arrays; reconstruct with column names
            # (assumes caller stored feature_names — handled in adversarial model)
            X = pd.DataFrame(X, columns=self._feature_names)

        race_vals = X[self.race_col].values
        p1 = np.array([
            self.race_proba_.get(r, self.global_proba_) for r in race_vals
        ])
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def set_feature_names(self, names):
        """Store feature names so numpy inputs can be reconstructed."""
        self._feature_names = list(names)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train_biased_model():
    # 1. Load & preprocess ---------------------------------------------------
    print("[1/3] Loading and preprocessing data …")
    df = load_and_filter_data()
    X_train, X_test, y_train, y_test, A_train, A_test, scaler, le = preprocess_data(df)

    print(f"      Train: {len(X_train):,}  Test: {len(X_test):,}")

    # 2. Fit biased model ----------------------------------------------------
    print("\n[2/3] Fitting RaceBiasedClassifier …")
    clf = RaceBiasedClassifier(race_col="race", random_state=SEED)
    clf.fit(X_train, y_train)
    clf.set_feature_names(X_train.columns)

    # 3. Evaluate  -
    print("\n[3/3] Test-set evaluation (reference only) …")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print("  Biased Model (f_biased) — Evaluation")
    print(f"{'='*50}")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Recid", "Recid"]))

    # Persist ----------------------------------------------------------------
    joblib.dump(clf, BIASED_MODEL_PATH)
    print(f"\nBiased model saved to '{BIASED_MODEL_PATH}'")

    return clf


if __name__ == "__main__":
    train_biased_model()