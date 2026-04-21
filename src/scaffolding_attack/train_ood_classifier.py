import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay
)
from config import SEED
from create_ood_samples import build_ood_dataset
from data_preprocessing import load_and_filter_data, preprocess_data
import matplotlib.pyplot as plt



# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
RF_PARAMS = dict(
    n_estimators=100,
    random_state=SEED,
    n_jobs=-1,
)

OOD_TEST_SIZE  = 0.20   # fraction of the OOD dataset held-out for final eval
OOD_NOISE_STD  = 1.0    # Gaussian noise std-dev passed to create_ood_samples
CV_FOLDS       = 5      # stratified k-fold cross-validation folds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_test_metrics_to_txt(metrics: dict, path: str = "./logs/ood_test_metrics.txt"):
    """Logs only the final test-set metrics to a text file."""
    with open(path, "w") as f:
        f.write("OOD CLASSIFIER - FINAL TEST METRICS\n")
        f.write("-" * 35 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:<12}: {v:.4f}\n")
    print(f"  Metrics logged to '{path}'")

def _plot_confusion_matrix(clf, X_te, y_te, path: str = "./logs/ood_confusion_matrix.png"):
    """Generates and saves the confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Create the display the confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(
        clf, X_te, y_te, 
        display_labels=["OOD (0)", "In-dist (1)"],
        cmap=plt.cm.Blues,
        ax=ax
    )
    ax.set_title("OOD Classifier: Confusion Matrix")
    
    plt.tight_layout()
    plt.savefig(path)
    print(f"  Confusion matrix plot saved to '{path}'")
    plt.close()


def _print_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                   split: str = "Test") -> dict:
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="binary")
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="binary", zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)

    print(f"\n{'='*55}")
    print(f"  OOD Classifier — {split} Metrics")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (positive = in-distribution)")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Detailed classification report:")
    print(classification_report(y_true, y_pred, target_names=["OOD (0)", "In-dist (1)"]))
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))
    print(f"{'='*55}\n")

    return dict(accuracy=acc, f1=f1, precision=prec, recall=rec, roc_auc=auc)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_ood_classifier(X_train_indist: pd.DataFrame,
                         noise_std: float = OOD_NOISE_STD,
                         save_path: str = "ood_classifier.pkl"):
    # Build Dataset and Split
    X_combined, y_combined = build_ood_dataset(X_train_indist)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_combined, y_combined, test_size=OOD_TEST_SIZE, 
        random_state=SEED, stratify=y_combined
    )

    # Cross Validation
    print(f"Running {CV_FOLDS}-fold stratified cross-validation …")
    cv_clf = RandomForestClassifier(**RF_PARAMS)
    cv_scoring = ("accuracy", "f1", "precision", "recall", "roc_auc")
    cv_results = cross_validate(
        cv_clf, X_tr, y_tr,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED),
        scoring=cv_scoring, n_jobs=-1,
    )

    # Final Model
    print("\nFitting final model …")
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_tr, y_tr)

    # Evaluation
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    test_metrics = _print_metrics(y_te, y_pred, y_prob, split="Held-out Test")

    # Logs
    _save_test_metrics_to_txt(test_metrics)
    _plot_confusion_matrix(clf, X_te, y_te)

    # Save best model
    print(f"Saving model to '{save_path}' …")
    joblib.dump(clf, save_path)
    
    return clf, test_metrics


# ---------------------------------------------------------------------------
# Convenience wrapper: load & predict
# ---------------------------------------------------------------------------

def load_ood_classifier(path: str = "./models/ood_classifier.pkl") -> RandomForestClassifier:
    """Load a previously saved OOD classifier."""
    return joblib.load(path)


def predict_ood(clf: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Return binary labels: 1 = in-distribution, 0 = OOD.
    Use this as a mask inside LIME / SHAP to discard perturbed samples.
    """
    return clf.predict(X)


def predict_ood_proba(clf: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return P(in-distribution) for each sample — useful for soft masking."""
    return clf.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    

    print("Loading and preprocessing data …")
    df = load_and_filter_data()
    X_train, X_test, y_train, y_test, A_train, A_test, scaler, le = preprocess_data(df)

    clf, metrics = train_ood_classifier(X_train, noise_std=OOD_NOISE_STD)

    print("\nFinal test-set metrics summary:")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")