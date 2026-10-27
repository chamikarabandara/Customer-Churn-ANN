from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Predict Customer Churn",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_ann_model.keras"
PREPROCESSOR_PATH = PROJECT_DIR / "models" / "preprocessor.joblib"

# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown("""
<style>
/* Main page */
.stApp {
    background-color: #F4F7FB;
    color: #1E293B;
}

[data-testid="stHeader"] {
    background-color: #F4F7FB;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #14213D;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Headings and labels */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #14213D !important;
}

.stApp p {
    color: #475569;
}

.stApp [data-testid="stWidgetLabel"] p,
.stApp label {
    color: #334155 !important;
    font-weight: 600;
}

/* DARK dropdown boxes with WHITE text */
.stApp [data-baseweb="select"] > div {
    background-color: #262730 !important;
    border: 1px solid #262730 !important;
}

.stApp [data-baseweb="select"] span,
.stApp [data-baseweb="select"] input {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

.stApp [data-baseweb="select"] svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}

/* Dropdown options menu */
ul[role="listbox"] {
    background-color: #FFFFFF !important;
}

li[role="option"],
li[role="option"] * {
    color: #1E293B !important;
    -webkit-text-fill-color: #1E293B !important;
}

/* DARK number inputs with WHITE text */
.stApp [data-baseweb="input"] > div {
    background-color: #262730 !important;
    border: 1px solid #262730 !important;
}

.stApp [data-baseweb="input"] input {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

.stApp [data-baseweb="input"] button,
.stApp [data-baseweb="input"] svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

/* Form card */
.stForm {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 20px;
}

/* Predict button */
.stFormSubmitButton > button {
    background-color: #1D4ED8 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px;
    font-weight: 700 !important;
    min-height: 48px;
}

.stFormSubmitButton > button p,
.stFormSubmitButton > button span {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

.stFormSubmitButton > button:hover {
    background-color: #1E40AF !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.05);
}

div[data-testid="stMetric"] label {
    color: #64748B !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #14213D !important;
    font-weight: 700;
}

/* Slider and progress */
.stApp [data-testid="stSlider"] {
    color: #1E293B !important;
}

div[data-testid="stProgress"] > div > div {
    background-color: #2A9D8F;
}

/* Result alerts */
.stApp [data-testid="stAlert"] p {
    color: #1E293B !important;
}

hr {
    border-color: #DCE3ED;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD TRAINED MODEL AND PREPROCESSOR
# ==================================================
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


# ==================================================
# PAGE HEADER
# ==================================================
st.title("🔮 Customer Churn Prediction")

st.markdown(
    "Estimate customer churn risk using the trained "
    "Artificial Neural Network (ANN) model."
)

st.info(
    "Enter the customer's service and billing information below, "
    "then select Predict Churn."
)


# ==================================================
# CHECK MODEL FILES
# ==================================================
if not MODEL_PATH.exists():
    st.error(f"ANN model not found: {MODEL_PATH}")
    st.stop()

if not PREPROCESSOR_PATH.exists():
    st.error(f"Preprocessor not found: {PREPROCESSOR_PATH}")
    st.stop()

try:
    model, preprocessor = load_artifacts()
    st.success(
        "ANN model and preprocessing pipeline loaded successfully."
    )
except Exception as exc:
    st.error(f"Could not load model artifacts: {exc}")
    st.stop()


# ==================================================
# CUSTOMER INPUT FORM
# ==================================================
st.subheader("Customer Information")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    # PERSONAL INFORMATION
    with col1:
        st.markdown("#### Personal Details")

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Has Partner?",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Has Dependents?",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    # INTERNET AND SERVICES
    with col2:
        st.markdown("#### Internet & Services")

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    # CONTRACT AND BILLING
    with col3:
        st.markdown("#### Contract & Billing")

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=840.0,
            step=10.0
        )

    submitted = st.form_submit_button(
        "🔮 Predict Churn",
        type="primary",
        use_container_width=True
    )


# ==================================================
# PREDICTION
# ==================================================
if submitted:

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    try:
        # Apply the same fitted preprocessing used during training
        transformed = preprocessor.transform(customer)
        transformed = np.asarray(transformed, dtype=np.float32)

        expected_features = model.input_shape[-1]

        if transformed.shape[1] != expected_features:
            st.error(
                f"Feature mismatch: received {transformed.shape[1]} "
                f"features but the model expects {expected_features}."
            )
            st.stop()

        raw_prediction = model.predict(
            transformed,
            verbose=0
        )

        probability = float(
            np.asarray(raw_prediction).reshape(-1)[0]
        )

        probability = float(
            np.clip(probability, 0.0, 1.0)
        )

        predicted_class = int(probability >= 0.5)

        # RESULTS
        st.divider()
        st.subheader("Prediction Results")

        result1, result2 = st.columns([1, 1])

        with result1:
            st.metric(
                "Estimated Churn Probability",
                f"{probability * 100:.1f}%"
            )

            st.progress(probability)

            st.caption(
                "Probability estimated by the trained ANN model."
            )

        with result2:
            if predicted_class == 1:
                st.error("⚠️ Prediction: Likely to Churn")
                st.markdown(
                    "**Suggested action:** Review customer concerns, "
                    "service quality and suitable retention options."
                )
            else:
                st.success("✅ Prediction: Likely to Stay")
                st.markdown(
                    "**Suggested action:** Continue providing good "
                    "service and monitor customer satisfaction."
                )

        # CUSTOMER SUMMARY
        st.subheader("Customer Summary")

        summary1, summary2, summary3, summary4 = st.columns(4)

        summary1.markdown("**Contract**")
        summary1.markdown(
           f"<div style='background:#FFFFFF; padding:12px; "
           f"border-radius:8px; color:#14213D; font-size:20px; "
           f"font-weight:700; overflow-wrap:anywhere;'>{contract}</div>",
           unsafe_allow_html=True
        )
        summary2.metric("Tenure", f"{tenure} months")
        summary3.metric(
            "Monthly Charges",
            f"${monthly_charges:.2f}"
        )
        summary4.metric(
            "Total Charges",
            f"${total_charges:.2f}"
        )

        st.caption(
            "This prediction is an estimate, not a guarantee of future "
            "customer behaviour. Classification uses a 0.50 threshold. "
            "Actual outcomes may differ."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")