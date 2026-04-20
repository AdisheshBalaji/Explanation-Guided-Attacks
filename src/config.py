import numpy as np

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Data URL
URL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'

# Feature definitions
TARGET_COL = 'two_year_recid'
SENSITIVE_FEATURES = ['race'] 

# Based on typical ProPublica preprocessing
CATEGORICAL_FEATURES = ['c_charge_degree', 'sex']
NUMERICAL_FEATURES = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']

# Hyperparameters
MLP_PARAMS = {
    'hidden_layer_sizes': (64, 32),
    'max_iter': 300,
    'random_state': SEED
}