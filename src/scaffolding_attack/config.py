# =============================================================================
# config.py  —  Central configuration for COMPAS Scaffolding Attack project
# =============================================================================

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------
TARGET_COL = "two_year_recid"

# ---------------------------------------------------------------------------
# Sensitive / protected attributes
# Included in X so that f_biased can produce race/sex-dependent outputs
# on OOD samples, causing LIME/SHAP to report false bias
# ---------------------------------------------------------------------------
SENSITIVE_FEATURES = ["race", "sex"]

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
NUMERICAL_FEATURES = [
    "age",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "days_b_screening_arrest",
]

CATEGORICAL_FEATURES = [
    "c_charge_degree",
    "race",   # African-American=0 / Caucasian=1 after encoding
    "sex",    # Female=0 / Male=1 after encoding
]

# ---------------------------------------------------------------------------
# All model features
# ---------------------------------------------------------------------------
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# OOD classifier settings
# ---------------------------------------------------------------------------
OOD_NOISE_STD  = 1.0
OOD_TEST_SIZE  = 0.20
OOD_MODEL_PATH = "./models/ood_classifier.pkl"



# ---------------------------------------------------------------------------
# Scaffolding attack settings
# ---------------------------------------------------------------------------
# Path to serialised fair and biased sub-models
FAIR_MODEL_PATH   = "./models/f_fair.pkl"
BIASED_MODEL_PATH = "./models/f_biased.pkl"

# OOD decision threshold: d(x) = 1 if P(in-dist) >= threshold, else 0
OOD_THRESHOLD = 0.5

