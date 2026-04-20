import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import URL, SEED, TARGET_COL, SENSITIVE_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

def load_and_filter_data():
    df_raw = pd.read_csv(URL)
    df = df_raw.copy()
    
    # ProPublica standard filters
    df = df[df['days_b_screening_arrest'].between(-30, 30)]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != 'O']   # exclude ordinary traffic
    df = df[df['score_text'] != 'N/A']
    df = df[df['race'].isin(['African-American', 'Caucasian'])] # binary race for fairness eval
    
    return df.reset_index(drop=True)

def preprocess_data(df):
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COL].values
    A = df[SENSITIVE_FEATURES].copy()
    
    # Encode categoricals
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, A, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # Scale numericals
    scaler = StandardScaler()
    X_train[NUMERICAL_FEATURES] = scaler.fit_transform(X_train[NUMERICAL_FEATURES])
    X_test[NUMERICAL_FEATURES] = scaler.transform(X_test[NUMERICAL_FEATURES])
    
    return X_train, X_test, y_train, y_test, A_train, A_test, scaler, label_encoders