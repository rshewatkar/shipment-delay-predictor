import pandas as pd
import numpy as np
from utils import load_model, load_encoders, load_scaler, print_separator

# ── Constants ───────────────────────────────────────────────
ENCODE_COLS = ['Warehouse_block', 'Mode_of_Shipment',
               'Product_importance', 'Gender']

SCALE_COLS  = ['Customer_care_calls', 'Customer_rating',
               'Cost_of_the_Product', 'Prior_purchases',
               'Discount_offered', 'Weight_in_gms']

# ── Preprocess single input ──────────────────────────────────
def preprocess_input(input_dict, label_encoders, scaler):
    df = pd.DataFrame([input_dict])

    # Encode
    for col in ENCODE_COLS:
        le = label_encoders[col]
        df[col] = le.transform(df[col])

    # Engineer features
    df['high_discount']  = (df['Discount_offered'] > 10).astype(int)
    df['weight_bucket']  = pd.cut(df['Weight_in_gms'],
                                   bins=[0,1000,2000,3000,4000,5000,7000],
                                   labels=[0,1,2,3,4,5]).astype(int)
    df['high_call_risk'] = (df['Customer_care_calls'] >= 4).astype(int)

    # Scale
    df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])

    return df

# ── Predict ─────────────────────────────────────────────────
def predict(input_dict):
    model          = load_model()
    label_encoders = load_encoders()
    scaler         = load_scaler()

    processed = preprocess_input(input_dict, label_encoders, scaler)

    prediction   = model.predict(processed)[0]
    probability  = model.predict_proba(processed)[0]
    delay_prob   = round(float(probability[1]) * 100, 1)
    ontime_prob  = round(float(probability[0]) * 100, 1)

    result = {
        'prediction' : int(prediction),
        'status'     : 'DELAYED' if prediction == 1 else 'ON TIME',
        'delay_prob' : delay_prob,
        'ontime_prob': ontime_prob,
        'risk_level' : get_risk_level(delay_prob)
    }
    return result

# ── Risk level ───────────────────────────────────────────────
def get_risk_level(delay_prob):
    if delay_prob >= 70:
        return 'HIGH'
    elif delay_prob >= 40:
        return 'MEDIUM'
    else:
        return 'LOW'

# ── CLI Test ─────────────────────────────────────────────────
if __name__ == '__main__':
    sample = {
        'Warehouse_block'    : 'D',
        'Mode_of_Shipment'   : 'Ship',
        'Customer_care_calls': 4,
        'Customer_rating'    : 2,
        'Cost_of_the_Product': 200,
        'Prior_purchases'    : 3,
        'Product_importance' : 'low',
        'Gender'             : 'M',
        'Discount_offered'   : 15,
        'Weight_in_gms'      : 3000
    }

    print_separator("PREDICTION TEST")
    result = predict(sample)
    print(f"Status     : {result['status']}")
    print(f"Delay Risk : {result['delay_prob']}%")
    print(f"On Time    : {result['ontime_prob']}%")
    print(f"Risk Level : {result['risk_level']}")
    print_separator()