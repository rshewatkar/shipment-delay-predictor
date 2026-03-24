import os
import joblib
import pandas as pd

# ── Paths ───────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW    = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROC   = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# ── Loaders ─────────────────────────────────────────────────
def load_model():
    path = os.path.join(MODELS_DIR, 'best_model.pkl')
    return joblib.load(path)

def load_encoders():
    path = os.path.join(MODELS_DIR, 'label_encoders.pkl')
    return joblib.load(path)

def load_scaler():
    path = os.path.join(MODELS_DIR, 'scaler.pkl')
    return joblib.load(path)

def load_raw_data(filename='shipment-data.csv'):
    path = os.path.join(DATA_RAW, filename)
    return pd.read_csv(path)

def load_processed_data():
    X_train = pd.read_csv(os.path.join(DATA_PROC, 'X_train.csv'))
    X_test  = pd.read_csv(os.path.join(DATA_PROC, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(DATA_PROC, 'y_train.csv')).values.ravel()
    y_test  = pd.read_csv(os.path.join(DATA_PROC, 'y_test.csv')).values.ravel()
    return X_train, X_test, y_train, y_test

# ── Savers ──────────────────────────────────────────────────
def save_model(model, filename='best_model.pkl'):
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"Model saved: {path}")

def save_artifact(obj, filename):
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(obj, path)
    print(f"Artifact saved: {path}")

# ── Display ─────────────────────────────────────────────────
def print_separator(title=''):
    print("=" * 55)
    if title:
        print(f"   {title}")
        print("=" * 55)