# project/inference.py
import pandas as pd
import joblib
from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def load_artifacts():
    """Loads the pre-trained model and preprocessors."""
    fair_model = joblib.load('artifacts/fair_model.joblib')
    scaler = joblib.load('artifacts/scaler.joblib')
    label_encoders = joblib.load('artifacts/label_encoders.joblib')
    return fair_model, scaler, label_encoders

def predict_new_data(new_data_df, sensitive_features_series):
    """
    Loads artifacts and infers on new data using the pre-trained fair pipeline.
    """
    # 1. Load everything
    fair_model, scaler, label_encoders = load_artifacts()
    
    X_infer = new_data_df.copy()
    
    # 2. Encode categoricals using the loaded dictionary
    for col in CATEGORICAL_FEATURES:
        if col in X_infer.columns:
            le = label_encoders[col]
            # Handle unseen labels by mapping them to a known class or handling gracefully
            # (Assuming clean data for this basic example)
            X_infer[col] = le.transform(X_infer[col])
            
    # 3. Scale numericals using the loaded scaler
    X_infer[NUMERICAL_FEATURES] = scaler.transform(X_infer[NUMERICAL_FEATURES])
    
    # 4. Predict using the Fairlearn ThresholdOptimizer
    predictions = fair_model.predict(X_infer, sensitive_features=sensitive_features_series)
    
    return predictions

# Example Usage:
if __name__ == "__main__":
    # Fake new incoming data for inference
    dummy_data = pd.DataFrame({
        'c_charge_degree': ['F', 'M'],
        'sex': ['Male', 'Female'],
        'age': [25, 45],
        'juv_fel_count': [0, 0],
        'juv_misd_count': [1, 0],
        'juv_other_count': [0, 0],
        'priors_count': [3, 0]
    })
    
    # The sensitive feature data corresponding to these two individuals
    dummy_sensitive = pd.DataFrame({
        'race': ['African-American', 'Caucasian']
    })
    
    preds = predict_new_data(dummy_data, dummy_sensitive)
    print("Predictions:", preds)