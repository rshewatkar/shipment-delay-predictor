import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from utils import (load_raw_data, save_artifact,
                   DATA_PROC, print_separator)
import os

# ── Constants ───────────────────────────────────────────────
ENCODE_COLS = ['Warehouse_block', 'Mode_of_Shipment',
               'Product_importance', 'Gender']

SCALE_COLS  = ['Customer_care_calls', 'Customer_rating',
               'Cost_of_the_Product', 'Prior_purchases',
               'Discount_offered', 'Weight_in_gms']

TARGET      = 'Reached.on.Time_Y.N'

# ── Step 1: Clean ───────────────────────────────────────────
def clean(df):
    df = df.copy()
    df.drop(columns=['ID'], inplace=True, errors='ignore')
    return df

# ── Step 2: Encode ──────────────────────────────────────────
def encode(df):
    df = df.copy()
    label_encoders = {}
    for col in ENCODE_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

# ── Step 3: Feature Engineering ─────────────────────────────
def engineer_features(df):
    df = df.copy()
    df['high_discount']  = (df['Discount_offered'] > 10).astype(int)
    df['weight_bucket']  = pd.cut(df['Weight_in_gms'],
                                   bins=[0,1000,2000,3000,4000,5000,7000],
                                   labels=[0,1,2,3,4,5]).astype(int)
    df['high_call_risk'] = (df['Customer_care_calls'] >= 4).astype(int)
    return df

# ── Step 4: Scale ───────────────────────────────────────────
def scale(X_train, X_test):
    scaler = StandardScaler()
    X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
    X_test[SCALE_COLS]  = scaler.transform(X_test[SCALE_COLS])
    return X_train, X_test, scaler

# ── Step 5: Split ───────────────────────────────────────────
def split(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2,
                            random_state=42, stratify=y)

# ── Full Pipeline ───────────────────────────────────────────
def run_pipeline():
    print_separator("PREPROCESSING PIPELINE")

    # Load
    df = load_raw_data()
    print(f"Loaded data: {df.shape}")

    # Clean
    df = clean(df)
    print("Cleaned: ID column dropped")

    # Encode
    df, label_encoders = encode(df)
    print("Encoded: categorical columns transformed")

    # Engineer
    df = engineer_features(df)
    print("Engineered: 3 new features added")

    # Split
    X_train, X_test, y_train, y_test = split(df)
    print(f"Split: train={X_train.shape}, test={X_test.shape}")

    # Scale
    X_train, X_test, scaler = scale(X_train, X_test)
    print("Scaled: numerical features standardized")

    # Save processed data
    os.makedirs(DATA_PROC, exist_ok=True)
    X_train.to_csv(f'{DATA_PROC}/X_train.csv', index=False)
    X_test.to_csv(f'{DATA_PROC}/X_test.csv',  index=False)
    y_train.to_csv(f'{DATA_PROC}/y_train.csv', index=False)
    y_test.to_csv(f'{DATA_PROC}/y_test.csv',  index=False)
    print("Saved: processed data to data/processed/")

    # Save artifacts
    save_artifact(label_encoders, 'label_encoders.pkl')
    save_artifact(scaler, 'scaler.pkl')

    print_separator("PREPROCESSING COMPLETE")
    return X_train, X_test, y_train, y_test, label_encoders, scaler

if __name__ == '__main__':
    run_pipeline()