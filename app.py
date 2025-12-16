import streamlit as st
import pandas as pd
import joblib

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("ckd_xgb_model.pkl")

# ===============================
# PREPROCESS (SAMA PERSIS DENGAN TRAINING)
# ===============================
def preprocess_input(df):
    df["bp_systolic"] = df["bp_systolic"].clip(80, 180)
    df["bp_diastolic"] = df["bp_diastolic"].clip(50, 120)
    df["serum_creatinine"] = df["serum_creatinine"].clip(lower=0)
    return df

# ===============================
# UI
# ===============================
st.title("CKD Risk Prediction System")

patient_age = st.number_input("Patient Age", 18, 100, 50)
bp_systolic = st.number_input("Systolic BP", 80, 200, 120)
bp_diastolic = st.number_input("Diastolic BP", 50, 150, 80)
blood_urea = st.number_input("Blood Urea", 0.0, 5.0, 1.0)
serum_creatinine = st.number_input("Serum Creatinine", 0.0, 5.0, 1.0)
albumin = st.selectbox("Albumin Level", [1, 2, 3])
diabetes = st.selectbox("Diabetes", [0, 1])
hypertension = st.selectbox("Hypertension", [0, 1])
nephrotoxic_label = st.selectbox("Nephrotoxic Drug", [0, 1])
toxicity_score = st.slider("Toxicity Score", 0.0, 1.0, 0.5)
pk_score = st.slider("PK Toxic Interaction Score", 0.0, 1.0, 0.5)

if st.button("Predict CKD Risk"):
    input_df = pd.DataFrame([{
        "patient_age": patient_age,
        "bp_systolic": bp_systolic,
        "bp_diastolic": bp_diastolic,
        "blood_urea": blood_urea,
        "serum_creatinine": serum_creatinine,
        "albumin": albumin,
        "diabetes": diabetes,
        "hypertension": hypertension,
        "nephrotoxic_label": nephrotoxic_label,
        "toxicity_score_composite": toxicity_score,
        "pk_toxic_interaction_score": pk_score
    }])

    input_df = preprocess_input(input_df)
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted CKD Risk Level: {prediction}")
