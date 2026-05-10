import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# =====================================
# BASIC EDA
# =====================================

print(df.head())
print(df.info())
print(df.describe())

sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

# =====================================
# DATA CLEANING
# =====================================

# Remove useless column
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)

# Remove missing rows
df.dropna(inplace=True)

# Convert target column
df['Churn'] = df['Churn'].map({
    'Yes': 1,
    'No': 0
})

# =====================================
# ENCODING
# =====================================

df = pd.get_dummies(df, drop_first=True)

# Convert ALL columns to float
df = df.astype(float)

# =====================================
# SPLIT DATA
# =====================================

X = df.drop('Churn', axis=1)
y = df['Churn']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# SCALING (ONLY FOR LR)
# =====================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================
# LOGISTIC REGRESSION
# =====================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

lr_model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced'
)

lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_acc = accuracy_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_prob)

print("\n===== LOGISTIC REGRESSION =====")
print("Accuracy:", lr_acc)
print("AUC:", lr_auc)
print(classification_report(y_test, lr_pred))

# =====================================
# LR CONFUSION MATRIX
# =====================================

cm_lr = confusion_matrix(y_test, lr_pred)

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =====================================
# RANDOM FOREST
# =====================================

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)

print("\n===== RANDOM FOREST =====")
print("Accuracy:", rf_acc)
print("AUC:", rf_auc)
print(classification_report(y_test, rf_pred))

# =====================================
# RF CONFUSION MATRIX
# =====================================

cm_rf = confusion_matrix(y_test, rf_pred)

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =====================================
# MODEL COMPARISON
# =====================================

print("\n===== MODEL COMPARISON =====")

print(f"LR Accuracy : {lr_acc:.3f}")
print(f"RF Accuracy : {rf_acc:.3f}")

print(f"LR AUC : {lr_auc:.3f}")
print(f"RF AUC : {rf_auc:.3f}")

# =====================================
# FEATURE IMPORTANCE
# =====================================

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='importance',
    ascending=False
)

print("\n===== TOP 10 FEATURES =====")
print(feature_importance.head(10))

# =====================================
# SHAP
# =====================================

import shap

# Smaller samples for speed
X_train_sample = X_train.sample(
    100,
    random_state=42
)

X_test_sample = X_test.sample(
    50,
    random_state=42
)

# Convert to float arrays
X_test_array = np.array(
    X_test_sample,
    dtype=np.float64
)

# =====================================
# SHAP EXPLAINER
# =====================================

explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(X_test_array)

# Handle SHAP versions safely
if isinstance(shap_values, list):
    shap_values_plot = shap_values[1]
else:
    shap_values_plot = shap_values

# =====================================
# SHAP SUMMARY PLOT
# =====================================

plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_values_plot,
    X_test_sample,
    max_display=10,
    show=False
)

plt.tight_layout()
plt.show()

# =====================================
# SINGLE CUSTOMER
# =====================================

customer_index = 0

rf_pred_sample = rf_model.predict(X_test_sample)
rf_prob_sample = rf_model.predict_proba(X_test_sample)[:, 1]

print(f"\nCustomer {customer_index}")
print(f"Prediction: {rf_pred_sample[customer_index]}")
print(f"Probability: {rf_prob_sample[customer_index]:.2f}")

# =====================================
# LOCAL SHAP PLOT
# =====================================

shap.summary_plot(
    shap_values_plot[
        customer_index:customer_index+1
    ],
    X_test_sample.iloc[
        customer_index:customer_index+1
    ],
    show=False
)

plt.show()

# =====================================
# TOP SHAP FEATURES
# =====================================

customer_index = 0

# Handle SHAP output safely
if isinstance(shap_values, list):

    # Older SHAP versions
    customer_shap = shap_values[1][customer_index]

else:

    # Newer SHAP versions
    customer_shap = shap_values[customer_index]

    # If 2D, select churn class
    if len(customer_shap.shape) > 1:
        customer_shap = customer_shap[:, 1]

# Convert to 1D
customer_shap = np.array(customer_shap).flatten()

print("Feature count:", len(X_test_sample.columns))
print("SHAP count:", len(customer_shap))

# Create dataframe
shap_df = pd.DataFrame({
    "feature": X_test_sample.columns,
    "shap_value": customer_shap
})

# Absolute importance
shap_df["abs_value"] = shap_df["shap_value"].abs()

# Sort top features
top_features = shap_df.sort_values(
    by="abs_value",
    ascending=False
).head(5)

print("\n===== TOP SHAP FEATURES =====\n")
print(top_features)

# ===========================
# BUSINESS INSIGHT GENERATOR
# ===========================

print("\n===== AI BUSINESS INSIGHT =====\n")

risk = rf_prob_sample[customer_index]

# Risk level
if risk >= 0.75:
    risk_level = "HIGH"
elif risk >= 0.50:
    risk_level = "MEDIUM"
else:
    risk_level = "LOW"

print(f"Overall Churn Risk: {risk_level}")
print(f"Predicted Churn Probability: {risk:.2%}\n")

print("Main Reasons:")

for _, row in top_features.iterrows():

    direction = (
        "increases churn risk"
        if row["shap_value"] > 0
        else "decreases churn risk"
    )

    print(
        f"- {row['feature']} {direction} "
        f"(impact: {row['shap_value']:.3f})"
    )

print("\nRetention Recommendation:")

if risk >= 0.75:

    print("""
Offer:
- loyalty discounts
- long-term contract benefits
- premium support
- personalized retention campaigns
""")

elif risk >= 0.50:

    print("""
Offer:
- moderate discounts
- service upgrades
- customer engagement programs
""")

else:

    print("""
Customer risk is low.
Focus on maintaining service quality
and customer satisfaction.
""")
    
import joblib

# Save trained Random Forest model
joblib.dump(rf_model, "rf_model.pkl")

# Save feature column names
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("✅ Model files saved successfully")