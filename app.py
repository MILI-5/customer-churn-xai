import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import numpy as np

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load("rf_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# =====================================
# TITLE
# =====================================

st.title("📊 Customer Churn Prediction System")

st.write("""
This dashboard predicts telecom customer churn
using Machine Learning + Explainable AI.
""")

# =====================================
# SIDEBAR INPUTS
# =====================================

st.sidebar.header("Customer Inputs")

tenure = st.sidebar.slider(
    "Tenure",
    0,
    72,
    12
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges",
    0.0,
    150.0,
    70.0
)

total_charges = st.sidebar.slider(
    "Total Charges",
    0.0,
    10000.0,
    1000.0
)

# =====================================
# CREATE INPUT DATA
# =====================================

input_data = {}

# Initialize all columns with 0
for col in model_columns:
    input_data[col] = 0

# Fill numerical features
if "tenure" in input_data:
    input_data["tenure"] = tenure

if "MonthlyCharges" in input_data:
    input_data["MonthlyCharges"] = monthly_charges

if "TotalCharges" in input_data:
    input_data["TotalCharges"] = total_charges

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# =====================================
# PREDICTION BUTTON
# =====================================

if st.button("Predict Churn"):

    # =====================================
    # MODEL PREDICTION
    # =====================================

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.success("Prediction Complete")

    st.write(f"## Churn Probability: {probability:.2%}")

    # =====================================
    # RISK LEVEL
    # =====================================

    if probability > 0.75:
        st.error("🔴 High Churn Risk")

    elif probability > 0.45:
        st.warning("🟠 Medium Churn Risk")

    else:
        st.success("🟢 Low Churn Risk")

    # =====================================
    # FINAL PREDICTION
    # =====================================

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    # =====================================
    # FEATURE IMPORTANCE
    # =====================================

    st.subheader("Top Feature Importance")

    importance_df = pd.DataFrame({
        "feature": model_columns,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    ).head(10)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        importance_df["feature"],
        importance_df["importance"]
    )

    ax.invert_yaxis()

    st.pyplot(fig)

    # =====================================
    # SHAP EXPLAINABILITY
    # =====================================

    st.subheader("SHAP Explainability")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_df)

    # ---------------------------------
    # FIX SHAP DIMENSIONS
    # ---------------------------------

    if isinstance(shap_values, list):

        # Older SHAP versions
        customer_shap = shap_values[1][0]

    else:

        # Newer SHAP versions
        if hasattr(shap_values, "values"):
            customer_shap = shap_values.values
        else:
            customer_shap = shap_values

        customer_shap = np.array(customer_shap)

        if customer_shap.ndim == 3:

            # shape: (1, features, classes)
            customer_shap = customer_shap[0, :, 1]

        elif customer_shap.ndim == 2:

            # shape: (1, features)
            customer_shap = customer_shap[0]

    # Final safety conversion
    customer_shap = np.ravel(customer_shap)

    # =====================================
    # CREATE SHAP DATAFRAME
    # =====================================

    shap_df = pd.DataFrame({
        "Feature": list(input_df.columns),
        "SHAP Value": customer_shap.tolist()
    })

    shap_df["Impact"] = shap_df["SHAP Value"].abs()

    shap_df = shap_df.sort_values(
        by="Impact",
        ascending=False
    ).head(10)

    # =====================================
    # SHAP PLOT
    # =====================================

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    ax2.barh(
        shap_df["Feature"],
        shap_df["SHAP Value"]
    )

    ax2.invert_yaxis()

    st.pyplot(fig2)

    # =====================================
    # BUSINESS RECOMMENDATION
    # =====================================

    st.subheader("Business Recommendation")

    if probability > 0.75:

        st.write("""
        Offer:
        - loyalty discounts
        - premium support
        - long-term contract plans
        - personalized retention campaigns
        """)

    elif probability > 0.45:

        st.write("""
        Offer:
        - customer engagement offers
        - moderate discounts
        - service upgrades
        """)

    else:

        st.write("""
        Customer is likely satisfied.
        Maintain current service quality.
        """)

    # =====================================
    # WHY THIS PREDICTION
    # =====================================

    st.subheader("Why This Prediction Happened")

    for _, row in shap_df.head(5).iterrows():

        direction = (
            "increases"
            if row["SHAP Value"] > 0
            else "decreases"
        )

        st.write(
            f"• {row['Feature']} {direction} churn risk "
            f"(impact: {row['SHAP Value']:.3f})"
        )

# =====================================
# CSV UPLOAD
# =====================================

st.subheader("Bulk Customer Prediction")

uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)

if uploaded_file is not None:

    uploaded_df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data")
    st.write(uploaded_df.head())

    # =====================================
    # MATCH MODEL COLUMNS
    # =====================================

    for col in model_columns:
        if col not in uploaded_df.columns:
            uploaded_df[col] = 0

    uploaded_df = uploaded_df[model_columns]

    # =====================================
    # PREDICTIONS
    # =====================================

    predictions = model.predict(uploaded_df)

    probabilities = model.predict_proba(uploaded_df)[:, 1]

    uploaded_df["Prediction"] = predictions

    uploaded_df["Churn Probability"] = probabilities

    uploaded_df["Risk"] = uploaded_df["Prediction"].map({
        1: "High Risk",
        0: "Low Risk"
    })

    st.write("Prediction Results")
    st.write(uploaded_df.head())

    # =====================================
    # DOWNLOAD CSV
    # =====================================

    csv = uploaded_df.to_csv(index=False)

    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )