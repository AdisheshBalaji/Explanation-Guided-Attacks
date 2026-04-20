from data import load_and_filter_data, preprocess_data
from visualization import plot_overview
from model import train_base_mlp, train_fair_model
from fairness import evaluate_fairness
from explainability import run_shap_explanations
import joblib
import os

def main():
    print("1. Loading and filtering data...")
    df = load_and_filter_data()
    
    print("2. Generating Exploratory Data Visualizations...")
    plot_overview(df)
    
    print("3. Preprocessing (non-sensitive features only)...")
    X_train, X_test, y_train, y_test, A_train, A_test, scaler, le_dict = preprocess_data(df)
    
    print("4. Training Base MLP Classifier...")
    base_model = train_base_mlp(X_train, y_train)
    y_pred_base = base_model.predict(X_test)
    
    print("5. Training Fair ThresholdOptimizer...")
    fair_model = train_fair_model(base_model, X_train, y_train, A_train)
    y_pred_fair = fair_model.predict(X_test, sensitive_features=A_test)
    
    print("\n--- Fairness Verification ---")
    evaluate_fairness(y_test, y_pred_base, A_test, model_name="Base MLP")
    evaluate_fairness(y_test, y_pred_fair, A_test, model_name="Fair Classifier")
    
    print("\n6. Running baseline explainability (SHAP)...")
    print("Note: Explanations describe base model behavior.")
    shap_values = run_shap_explanations(base_model, X_train, X_test)
    print("Pipeline complete. Ground truth rankings ready for attack experiments.")

    

    os.makedirs('checkpoints', exist_ok=True)

    print("7. Saving models and preprocessors...")
    # Save the models
    joblib.dump(base_model, 'checkpoints/base_mlp.joblib')
    joblib.dump(fair_model, 'checkpoints/fair_model.joblib')

    # Save the preprocessors
    joblib.dump(scaler, 'checkpoints/scaler.joblib')
    joblib.dump(le_dict, 'checkpoints/label_encoders.joblib')

    print("Saved successfully to the 'checkpoints/' directory.")

if __name__ == "__main__":
    main()