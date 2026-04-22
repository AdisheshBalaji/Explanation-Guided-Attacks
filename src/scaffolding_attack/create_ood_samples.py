import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, SEED
from data_preprocessing import load_and_filter_data, preprocess_data 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def visualize_tsne(X_train: pd.DataFrame, X_perturb: pd.DataFrame, sample_size: int = 2000):
    """
    Perform t-SNE dimensionality reduction and plot the original vs. perturbed data.
    
    Args:
        X_train: The original training data.
        X_perturb: The generated OOD data (LIME + SHAP).
        sample_size: Maximum number of samples to use per class to keep t-SNE fast.
    """
    print("\nPreparing t-SNE visualization...")
    
    # Sub-sample to speed up t-SNE if the dataset is large
    n_samples_train = min(len(X_train), sample_size)
    n_samples_perturb = min(len(X_perturb), sample_size)
    
    rng = np.random.default_rng(SEED)
    
    # Randomly sample indices
    train_idx = rng.choice(len(X_train), size=n_samples_train, replace=False)
    perturb_idx = rng.choice(len(X_perturb), size=n_samples_perturb, replace=False)
    
    X_train_sample = X_train.iloc[train_idx].copy()
    X_perturb_sample = X_perturb.iloc[perturb_idx].copy()
    
    # Combine the samples and create labels
    X_combined = pd.concat([X_train_sample, X_perturb_sample], axis=0, ignore_index=True)
    labels = ['Original'] * n_samples_train + ['Perturbed (OOD)'] * n_samples_perturb
    
    print(f"Running t-SNE on {len(X_combined)} total samples. This may take a moment...")
    
    # Initialize and fit t-SNE
    # init='pca' and learning_rate='auto' are current sklearn best practices
    tsne = TSNE(n_components=2, random_state=SEED, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(X_combined)
    
    # Create a DataFrame for easy plotting with seaborn
    tsne_df = pd.DataFrame({
        'tsne_1': tsne_results[:, 0],
        'tsne_2': tsne_results[:, 1],
        'Label': labels
    })
    
    # Generate the overlaid scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='tsne_1', 
        y='tsne_2',
        hue='Label',
        palette={'Original': '#1f77b4', 'Perturbed (OOD)': '#ff7f0e'}, # Blue and Orange
        data=tsne_df,
        alpha=0.6,  # Transparency to see overlapping points
        s=30        # Dot size
    )
    
    plt.title("t-SNE Visualization: Original vs. LIME/SHAP Perturbed Data", fontsize=14)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend(title="Dataset Origin")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("tsne_visualization.png")
    plt.show()


def create_ood_samples(X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Generate out-of-distribution (OOD) samples specifically designed to catch 
    the perturbation methods of both LIME and SHAP according to the paper specs.
    """
    rng = np.random.default_rng(SEED)
    
    # =====================================================================
    # 1. LIME-Style Perturbation
    # Generate perturbations for EVERY data point by adding N(0, 1) noise
    # =====================================================================
    X_lime = X_train.copy()
    
    if NUMERICAL_FEATURES:
        # Strict N(0, 1) normal distribution noise
        noise = rng.normal(loc=0.0, scale=1.0, size=(len(X_lime), len(NUMERICAL_FEATURES)))
        X_lime[NUMERICAL_FEATURES] = X_lime[NUMERICAL_FEATURES].values + noise

    # Maintain original categorical shuffling to handle non-continuous variables
    for col in CATEGORICAL_FEATURES:
        shuffled = X_lime[col].values.copy()
        rng.shuffle(shuffled)
        X_lime[col] = shuffled

    # =====================================================================
    # 2. SHAP-Style Perturbation 
    # Background distribution: KMeans with 10 clusters. 
    # Replace a random subset of features for EVERY data point.
    # =====================================================================
    X_shap = X_train.copy()
    
    # Fit KMeans to extract 10 cluster centers for the background distribution
    kmeans = KMeans(n_clusters=10, random_state=SEED, n_init='auto')
    kmeans.fit(X_train)
    background_centers = kmeans.cluster_centers_  # Shape: (10, n_features)
    
    # Create a boolean mask to randomly choose a subset of features to replace (~50%)
    replace_mask = rng.choice([True, False], size=X_shap.shape)
    
    # Randomly assign one of the 10 cluster centers as the background for each sample
    bg_indices = rng.choice(10, size=len(X_shap))
    bg_values = background_centers[bg_indices]
    
    # Replace values where mask is True with the corresponding background values
    X_shap_values = X_shap.values
    np.copyto(X_shap_values, bg_values, where=replace_mask)
    
    # Convert back to DataFrame
    X_shap = pd.DataFrame(X_shap_values, columns=X_shap.columns, index=X_shap.index)

    # =====================================================================
    # Combine both attack signatures into one OOD dataset
    # Note: OOD dataset is now 2x the size of the original dataset
    # =====================================================================
    X_perturb = pd.concat([X_lime, X_shap], axis=0, ignore_index=True)
    
    # Restore original datatypes in case KMeans casting shifted them to float64
    for col in X_train.columns:
        X_perturb[col] = X_perturb[col].astype(X_train[col].dtype)
        
    return X_perturb


def build_ood_dataset(X_train: pd.DataFrame):
    """
    Build a combined labelled dataset for OOD classifier training:
      - In-distribution samples  : label 1 (Real data)
      - OOD (perturbed) samples  : label 0 (LIME/SHAP synthetic data)
    """
    # 1. Generate the combined SHAP + LIME fake data
    X_perturb = create_ood_samples(X_train)

    # 2. Stack real (1) and fake (0) data
    X_combined = pd.concat([X_train, X_perturb], axis=0, ignore_index=True)
    y_combined = np.array([1] * len(X_train) + [0] * len(X_perturb), dtype=np.int32)

    # 3. Shuffle so the real and fake data are interleaved
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

    # Build and verify the OOD samples
    X_ood = create_ood_samples(X_train)
    print("Original X_train shape :", X_train.shape)
    print("OOD X_perturb shape    :", X_ood.shape)  # Should be exactly 2 * len(X_train)

    # Combine into the training dataset
    X_comb, y_comb = build_ood_dataset(X_train)
    print("Combined dataset shape :", X_comb.shape) # Should be 3 * len(X_train)
    print("Label distribution     :", dict(zip(*np.unique(y_comb, return_counts=True))))
    
    # Run the t-SNE visualization
    # Increase or decrease sample_size depending on your CPU power
    visualize_tsne(X_train, X_ood, sample_size=2000)
    