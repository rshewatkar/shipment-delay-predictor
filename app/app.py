import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config 
st.set_page_config(
    page_title="Shipment Delay Predictor",
    page_icon="🚚",
    layout="wide"
)

#  Load model and encoders 
@st.cache_resource
def load_artifacts():
    model    = joblib.load('../models/best_model.pkl')
    encoders = joblib.load('../models/label_encoders.pkl')
    scaler   = joblib.load('../models/scaler.pkl')
    return model, encoders, scaler

model, label_encoders, scaler = load_artifacts()

#  Header 
st.title("🚚 Shipment Delay Predictor")
st.markdown("Enter shipment details below to predict whether the delivery will be **on time or delayed**.")
st.divider()

#  Layout: 2 columns 
col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Shipment Details")

    warehouse_block = st.selectbox(
        "Warehouse Block",
        options=['A', 'B', 'C', 'D', 'F'],
        help="Which warehouse block is the shipment coming from?"
    )

    mode_of_shipment = st.selectbox(
        "Mode of Shipment",
        options=['Ship', 'Flight', 'Road'],
        help="How will the shipment be transported?"
    )

    product_importance = st.selectbox(
        "Product Importance",
        options=['low', 'medium', 'high'],
        help="Priority level of the product"
    )

    gender = st.selectbox(
        "Customer Gender",
        options=['F', 'M']
    )

    weight_in_gms = st.slider(
        "Product Weight (grams)",
        min_value=1000, max_value=7000,
        value=3000, step=100
    )

with col2:
    st.subheader("👤 Customer Details")

    customer_care_calls = st.slider(
        "Customer Care Calls",
        min_value=1, max_value=7,
        value=3,
        help="Number of calls made to customer care before delivery"
    )

    customer_rating = st.slider(
        "Customer Rating",
        min_value=1, max_value=5,
        value=3,
        help="Customer's satisfaction rating (1=Low, 5=High)"
    )

    cost_of_the_product = st.slider(
        "Cost of Product (₹)",
        min_value=96, max_value=308,
        value=200
    )

    prior_purchases = st.slider(
        "Prior Purchases",
        min_value=2, max_value=7,
        value=3,
        help="Number of previous orders by this customer"
    )

    discount_offered = st.slider(
        "Discount Offered (%)",
        min_value=0, max_value=65,
        value=5
    )

st.divider()

#  Predict button 
predict_btn = st.button("🔍 Predict Delivery Status", type="primary", use_container_width=True)

if predict_btn:

    #  Build input dataframe 
    input_dict = {
        'Warehouse_block'    : warehouse_block,
        'Mode_of_Shipment'   : mode_of_shipment,
        'Customer_care_calls': customer_care_calls,
        'Customer_rating'    : customer_rating,
        'Cost_of_the_Product': cost_of_the_product,
        'Prior_purchases'    : prior_purchases,
        'Product_importance' : product_importance,
        'Gender'             : gender,
        'Discount_offered'   : discount_offered,
        'Weight_in_gms'      : weight_in_gms
    }

    input_df = pd.DataFrame([input_dict])

    #  Encode categorical columns 
    encode_cols = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
    for col in encode_cols:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

    #  Engineer same features as training 
    input_df['high_discount']  = (input_df['Discount_offered'] > 10).astype(int)
    input_df['weight_bucket']  = pd.cut(input_df['Weight_in_gms'],
                                         bins=[0,1000,2000,3000,4000,5000,7000],
                                         labels=[0,1,2,3,4,5]).astype(int)
    input_df['high_call_risk'] = (input_df['Customer_care_calls'] >= 4).astype(int)

    #  Scale numerical columns 
    scale_cols = ['Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
                  'Prior_purchases', 'Discount_offered', 'Weight_in_gms']
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])

    #  Predict 
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    delay_prob  = round(float(probability[1]) * 100, 1)
    ontime_prob = round(float(probability[0]) * 100, 1)

    st.divider()

    #  Result display 
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if prediction == 1:
            st.error("### ⚠️ Likely DELAYED")
        else:
            st.success("### ✅ Likely ON TIME")

    with res_col2:
        st.metric(label="Delay Risk",   value=f"{delay_prob:.1f}%")

    with res_col3:
        st.metric(label="On-Time Chance", value=f"{ontime_prob:.1f}%")

    st.divider()

    # Risk factors 
    st.subheader("📊 Risk Factor Analysis")

    risk_factors = []

    if discount_offered > 10:
        risk_factors.append(("🔴 High discount (>10%)",
                              "Heavily discounted orders are almost always delayed"))
    if customer_care_calls >= 4:
        risk_factors.append(("🔴 High customer calls (4+)",
                              "Many calls before delivery strongly indicates delay"))
    if 2000 <= weight_in_gms <= 4000:
        risk_factors.append(("🟡 Medium weight range (2–4 kg)",
                              "This weight range has highest historical delay rate"))
    if mode_of_shipment == 'Ship':
        risk_factors.append(("🟡 Ship mode selected",
                              "Ship has higher delay rate than Flight or Road"))
    if customer_rating <= 2:
        risk_factors.append(("🟡 Low customer rating",
                              "Low-rated customers tend to have more delivery issues"))

    if risk_factors:
        for factor, explanation in risk_factors:
            st.warning(f"**{factor}** — {explanation}")
    else:
        st.info("✅ No major risk factors detected for this shipment.")

    st.divider()

    #  Business recommendation 
    st.subheader("💡 Recommendation")

    if delay_prob >= 70:
        st.error("""
        **High Risk — Immediate Action Required**
        - Prioritize this order in warehouse packing queue
        - Consider upgrading shipment mode if possible
        - Send proactive delay notification to customer
        - Assign dedicated tracking to this shipment
        """)
    elif delay_prob >= 40:
        st.warning("""
        **Medium Risk — Monitor Closely**
        - Flag this order for daily status check
        - Ensure carrier pickup is confirmed on time
        - Have customer service ready for follow-up
        """)
    else:
        st.success("""
        **Low Risk — Standard Processing**
        - No special action needed
        - Follow standard delivery protocol
        """)

# Footer 
st.divider()
st.markdown("""
<div style='text-align: center; color: grey; font-size: 13px;'>
    Built by <strong>Rahul Shewatkar</strong> — 
    Logistics Operations Expert + Data Scientist |
    <a href='https://github.com/rshewatkar/shipment-delay-predictor'>GitHub</a>
</div>
""", unsafe_allow_html=True)