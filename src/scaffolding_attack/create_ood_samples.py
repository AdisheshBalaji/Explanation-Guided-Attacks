import numpy as np
import pandas as pd
from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, SEED
from data_preprocessing import load_and_filter_data, preprocess_data 


def create_ood_samples(X_train: pd.DataFrame, noise_std: float = 1.0) -> pd.DataFrame:
    """
    Generate out-of-distribution (OOD) samples from X_train by:
      - Adding independent Gaussian noise to all continuous (numerical) features
      - Randomly shuffling each categorical feature column independently

    Parameters
    ----------
    X_train   : pd.DataFrame : in-distribution training data (already scaled numerics,
                                label-encoded categoricals)
    noise_std : float        : std-dev of Gaussian noise added to numerical features
                               (default=1.0, i.e. one std in the scaled space)

    Returns
    -------
    X_perturb : pd.DataFrame — OOD samples, same shape and columns as X_train
    """
    rng = np.random.default_rng(SEED)
    X_perturb = X_train.copy()

    # --- 1. Perturb numerical features with independent Gaussian noise ----------
    if NUMERICAL_FEATURES:
        noise = rng.normal(loc=0.0, scale=noise_std, size=(len(X_train), len(NUMERICAL_FEATURES)))
        X_perturb[NUMERICAL_FEATURES] = X_train[NUMERICAL_FEATURES].values + noise

    # --- 2. Shuffle categorical features independently -------------------------
    for col in CATEGORICAL_FEATURES:
        shuffled = X_train[col].values.copy()
        rng.shuffle(shuffled)
        X_perturb[col] = shuffled

    return X_perturb


def build_ood_dataset(X_train: pd.DataFrame, noise_std: float = 1.0):
    """
    Build a combined labelled dataset for OOD classifier training:
      - In-distribution samples  : label 1
      - OOD (perturbed) samples  : label 0

    Returns
    -------
    X_combined : pd.DataFrame
    y_combined : np.ndarray  (1 = in-dist, 0 = OOD)
    """
    X_perturb = create_ood_samples(X_train, noise_std=noise_std)

    X_combined = pd.concat([X_train, X_perturb], axis=0, ignore_index=True)
    y_combined = np.array([1] * len(X_train) + [0] * len(X_perturb), dtype=np.int32)

    # Shuffle so the two halves are interleaved
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X_combined))
    X_combined = X_combined.iloc[idx].reset_index(drop=True)
    y_combined = y_combined[idx]

    return X_combined, y_combined


# ---------------------------------------------------------------------------
# Main 
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    

    df = load_and_filter_data()
    X_train, X_test, y_train, y_test, A_train, A_test, scaler, le = preprocess_data(df)

    X_ood = create_ood_samples(X_train)
    print("Original X_train shape :", X_train.shape)
    print("OOD X_perturb shape    :", X_ood.shape)

    X_comb, y_comb = build_ood_dataset(X_train)
    print("Combined dataset shape :", X_comb.shape)
    print("Label distribution     :", dict(zip(*np.unique(y_comb, return_counts=True))))