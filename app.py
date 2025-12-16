import streamlit as st
import pandas as pd
import joblib

# ===============================
# LOAD TRAINED MODEL
# ===============================
model = joblib.load("ckd_xgb_model.pkl")

# ===============================
# PREPROCESSING (MATCH TRAINING)
# ===============================

def preprocess_input(df):
    df["bp_systolic"] = df["bp_systolic"].clip(80, 180)
    df["bp_diastolic"] = df["bp_diastolic"].clip(50, 120)
    df["serum_creatinine"] = df["serum_creatinine"].clip(lower=0)
    return df

# ===============================
# MAPPING (CLINICAL-FRIENDLY)
# ===============================

binary_map = {
    "No history": 0,
    "Diagnosed": 1,
    "Not used": 0,
    "Used": 1
}

albumin_map = {
    "Normal": 1,
    "Mild Hypoalbuminemia": 2,
    "Severe Hypoalbuminemia": 3
}

ckd_risk_map = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk"
}

# ===============================
# STREAMLIT UI
# ===============================

st.set_page_config(page_title="CKD Risk Prediction", layout="centered")

st.title("🩺 Chronic Kidney Disease Risk Prediction")
st.write("This system predicts CKD risk based on clinical biomarkers and drug toxicity profiles.")

st.subheader("Patient Clinical Information")

patient_age = st.number_input("Patient Age (years)", 18, 100, 50)

bp_systolic = st.number_input("Systolic Blood Pressure (mmHg)", 80, 200, 120)
bp_diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", 50, 150, 80)

blood_urea = st.number_input("Blood Urea Level", 0.0, 5.0, 1.0)
serum_creatinine = st.number_input("Serum Creatinine Level", 0.0, 5.0, 1.0)

albumin_text = st.selectbox(
    "Serum Albumin Status",
    ["Normal", "Mild Hypoalbuminemia", "Severe Hypoalbuminemia"]
)

diabetes_text = st.selectbox(
    "Diabetes Status",
    ["No history", "Diagnosed"]
)

hypertension_text = st.selectbox(
    "Hypertension Status",
    ["No history", "Diagnosed"]
)

st.subheader("Drug Toxicity Profile")

nephrotoxic_text = st.selectbox(
    "Nephrotoxic Drug Usage",
    ["Not used", "Used"]
)

toxicity_score = st.slider(
    "Drug Toxicity Score",
    0.0, 1.0, 0.5,
    help="Higher score indicates greater nephrotoxic potential"
)

pk_score = st.slider(
    "PK Toxic Interaction Score",
    0.0, 1.0, 0.5,
    help="Represents pharmacokinetic interaction-related toxicity"
)

# ===============================
# PREDICTION
# ===============================

if st.button("🔍 Predict CKD Risk"):
    input_df = pd.DataFrame([{
        "patient_age": patient_age,
        "bp_systolic": bp_systolic,
        "bp_diastolic": bp_diastolic,
        "blood_urea": blood_urea,
        "serum_creatinine": serum_creatinine,
        "albumin": albumin_map[albumin_text],
        "diabetes": binary_map[diabetes_text],
        "hypertension": binary_map[hypertension_text],
        "nephrotoxic_label": binary_map[nephrotoxic_text],
        "toxicity_score_composite": toxicity_score,
        "pk_toxic_interaction_score": pk_score
    }])

    input_df = preprocess_input(input_df)
    prediction = model.predict(input_df)[0]

    risk_label = ckd_risk_map[prediction]

    # ===============================
    # RISK COLOR CODING OUTPUT
    # ===============================
    if prediction == 0:
        st.markdown("<div style='background-color:#E8F5E9;padding:15px;border-radius:10px'>"
                    "<h3 style='color:#2E7D32'>🟢 CKD Risk Level: Low Risk</h3>"
                    "<p><b>Clinical Recommendation:</b></p>"
                    "<ul>"
                    "<li>Maintain healthy blood pressure and blood glucose levels</li>"
                    "<li>Avoid unnecessary nephrotoxic medications</li>"
                    "<li>Routine kidney function monitoring annually</li>"
                    "</ul></div>", unsafe_allow_html=True)

    elif prediction == 1:
        st.markdown("<div style='background-color:#FFFDE7;padding:15px;border-radius:10px'>"
                    "<h3 style='color:#F9A825'>🟡 CKD Risk Level: Moderate Risk</h3>"
                    "<p><b>Clinical Recommendation:</b></p>"
                    "<ul>"
                    "<li>Schedule regular renal function monitoring</li>"
                    "<li>Review current medications for nephrotoxic potential</li>"
                    "<li>Implement lifestyle modifications to reduce CKD progression</li>"
                    "</ul></div>", unsafe_allow_html=True)

    else:
        st.markdown("<div style='background-color:#FFEBEE;padding:15px;border-radius:10px'>"
                    "<h3 style='color:#C62828'>🔴 CKD Risk Level: High Risk</h3>"
                    "<p><b>Clinical Recommendation:</b></p>"
                    "<ul>"
                    "<li>Immediate referral to a nephrologist is recommended</li>"
                    "<li>Consider discontinuation or adjustment of nephrotoxic drugs</li>"
                    "<li>Initiate intensive renal monitoring and management</li>"
                    "</ul></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("This tool is intended for research and educational purposes only.")
