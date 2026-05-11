import os
import time
import joblib
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

# =====================================
# CREATE FIGURE FOLDER
# =====================================

os.makedirs("research_figures", exist_ok=True)

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

plt.figure(figsize=(6, 4))

sns.countplot(x='Churn', data=df)

plt.title("Churn Distribution")

plt.savefig("research_figures/churn_distribution.png")

plt.show()

# =====================================
# DATA CLEANING
# =====================================

# Remove customerID column
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)

# Remove missing values
df.dropna(inplace=True)

# Encode target variable
df['Churn'] = df['Churn'].map({
    'Yes': 1,
    'No': 0
})

# =====================================
# ENCODING
# =====================================

df = pd.get_dummies(df, drop_first=True)

# Convert all columns to float
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
# SCALING
# =====================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# =====================================
# METRICS
# =====================================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    roc_curve
)

# =====================================
# LOGISTIC REGRESSION
# =====================================

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced'
)

lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)

lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_acc = accuracy_score(y_test, lr_pred)

lr_auc = roc_auc_score(y_test, lr_prob)

lr_f1 = f1_score(y_test, lr_pred)

print("\n===== LOGISTIC REGRESSION =====")

print("Accuracy:", lr_acc)

print("AUC:", lr_auc)

print("F1 Score:", lr_f1)

print(classification_report(y_test, lr_pred))

# =====================================
# LR CONFUSION MATRIX
# =====================================

cm_lr = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_lr,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Logistic Regression Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("research_figures/lr_confusion_matrix.png")

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

rf_f1 = f1_score(y_test, rf_pred)

print("\n===== RANDOM FOREST =====")

print("Accuracy:", rf_acc)

print("AUC:", rf_auc)

print("F1 Score:", rf_f1)

print(classification_report(y_test, rf_pred))

# =====================================
# RF CONFUSION MATRIX
# =====================================

cm_rf = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("research_figures/rf_confusion_matrix.png")

plt.show()

# =====================================
# XGBOOST
# =====================================

try:

    from xgboost import XGBClassifier

except ImportError:

    print("\n❌ XGBoost not installed")

    print("Run this command:")

    print("pip install xgboost")

    exit()

start_time = time.time()

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

xgb_training_time = time.time() - start_time

xgb_pred = xgb_model.predict(X_test)

xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

xgb_acc = accuracy_score(y_test, xgb_pred)

xgb_auc = roc_auc_score(y_test, xgb_prob)

xgb_f1 = f1_score(y_test, xgb_pred)

print("\n===== XGBOOST =====")

print("Accuracy:", xgb_acc)

print("AUC:", xgb_auc)

print("F1 Score:", xgb_f1)

print("Training Time:", xgb_training_time)

print(classification_report(y_test, xgb_pred))

# =====================================
# ANN / MLP CLASSIFIER
# =====================================

from sklearn.neural_network import MLPClassifier

ann_model = MLPClassifier(

    hidden_layer_sizes=(64, 32),

    activation='relu',

    solver='adam',

    max_iter=1000,

    early_stopping=True,

    random_state=42
)

ann_model.fit(X_train_scaled, y_train)

ann_pred = ann_model.predict(X_test_scaled)

ann_prob = ann_model.predict_proba(X_test_scaled)[:, 1]

ann_acc = accuracy_score(y_test, ann_pred)

ann_auc = roc_auc_score(y_test, ann_prob)

ann_f1 = f1_score(y_test, ann_pred)

print("\n===== ANN (MLP CLASSIFIER) =====")

print("Accuracy:", ann_acc)

print("AUC:", ann_auc)

print("F1 Score:", ann_f1)

print(classification_report(y_test, ann_pred))

# =====================================
# ANN CONFUSION MATRIX
# =====================================

cm_ann = confusion_matrix(y_test, ann_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_ann,
    annot=True,
    fmt='d',
    cmap='Oranges'
)

plt.title("ANN Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("research_figures/ann_confusion_matrix.png")

plt.show()

# =====================================
# MODEL COMPARISON TABLE
# =====================================

results_df = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "ANN"
    ],

    "Accuracy": [
        lr_acc,
        rf_acc,
        xgb_acc,
        ann_acc
    ],

    "AUC": [
        lr_auc,
        rf_auc,
        xgb_auc,
        ann_auc
    ],

    "F1 Score": [
        lr_f1,
        rf_f1,
        xgb_f1,
        ann_f1
    ]
})

print("\n===== MODEL COMPARISON TABLE =====")

print(results_df)

# =====================================
# FEATURE IMPORTANCE
# =====================================

feature_importance = pd.DataFrame({

    "feature": X.columns,

    "importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n===== TOP 10 FEATURES =====")

print(feature_importance.head(10))

# =====================================
# SHAP ANALYSIS
# =====================================

X_test_sample = X_test.sample(
    50,
    random_state=42
)

explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(X_test_sample)

# =====================================
# HANDLE SHAP VERSIONS
# =====================================

if isinstance(shap_values, list):

    shap_values_plot = shap_values[1]

else:

    if hasattr(shap_values, "values"):

        shap_values_plot = shap_values.values

    else:

        shap_values_plot = shap_values

    shap_values_plot = np.array(shap_values_plot)

    if shap_values_plot.ndim == 3:

        shap_values_plot = shap_values_plot[:, :, 1]

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

plt.savefig("research_figures/shap_summary.png")

plt.show()

# =====================================
# SINGLE CUSTOMER ANALYSIS
# =====================================

customer_index = 0

rf_pred_sample = rf_model.predict(X_test_sample)

rf_prob_sample = rf_model.predict_proba(X_test_sample)[:, 1]

print(f"\nCustomer {customer_index}")

print(f"Prediction: {rf_pred_sample[customer_index]}")

print(f"Probability: {rf_prob_sample[customer_index]:.2f}")

# =====================================
# CUSTOMER SHAP VALUES
# =====================================

customer_shap = shap_values_plot[customer_index]

customer_shap = np.array(customer_shap).flatten()

print("Feature count:", len(X_test_sample.columns))

print("SHAP count:", len(customer_shap))

# =====================================
# SHAP WATERFALL PLOT
# =====================================

try:

    # Fix expected value safely
    if isinstance(explainer.expected_value, list):
        base_value = explainer.expected_value[1]
    else:
        base_value = explainer.expected_value

    # Convert ndarray to scalar
    if isinstance(base_value, np.ndarray):
        base_value = base_value.flatten()[0]

    shap_explanation = shap.Explanation(
        values=customer_shap,
        base_values=base_value,
        data=X_test_sample.iloc[customer_index],
        feature_names=X_test_sample.columns.tolist()
    )

    plt.figure(figsize=(10, 6))

    shap.plots.waterfall(
        shap_explanation,
        max_display=10,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        "research_figures/shap_waterfall.png"
    )

    plt.show()

except Exception as e:

    print("\nWaterfall plot skipped:")

    print(e)

# =====================================
# SHAP DATAFRAME
# =====================================

shap_df = pd.DataFrame({

    "feature": X_test_sample.columns,

    "shap_value": customer_shap
})

shap_df["abs_value"] = shap_df["shap_value"].abs()

top_features = shap_df.sort_values(
    by="abs_value",
    ascending=False
).head(5)

print("\n===== TOP SHAP FEATURES =====")

print(top_features)

# =====================================
# BUSINESS INSIGHT GENERATOR
# =====================================

print("\n===== AI BUSINESS INSIGHT =====")

risk = rf_prob_sample[customer_index]

if risk >= 0.75:

    risk_level = "HIGH"

elif risk >= 0.50:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"

print(f"\nOverall Churn Risk: {risk_level}")

print(f"Predicted Churn Probability: {risk:.2%}")

print("\nMain Reasons:")

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
- premium support
- personalized retention campaigns
- long-term contract benefits
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
Maintain service quality
and customer satisfaction.
""")

# =====================================
# SAVE MODEL FILES
# =====================================

joblib.dump(rf_model, "rf_model.pkl")

joblib.dump(
    X.columns.tolist(),
    "model_columns.pkl"
)

print("\n✅ Model files saved successfully")

# =====================================
# ROC CURVE COMPARISON
# =====================================

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_prob)

ann_fpr, ann_tpr, _ = roc_curve(y_test, ann_prob)

plt.figure(figsize=(8, 6))

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"LR AUC = {lr_auc:.3f}"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"RF AUC = {rf_auc:.3f}"
)

plt.plot(
    xgb_fpr,
    xgb_tpr,
    label=f"XGB AUC = {xgb_auc:.3f}"
)

plt.plot(
    ann_fpr,
    ann_tpr,
    label=f"ANN AUC = {ann_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.savefig("research_figures/roc_curve.png")

plt.show()

# =====================================
# LIME EXPLAINABILITY
# =====================================

print("\n===== LIME EXPLAINABILITY =====")

explainer_lime = LimeTabularExplainer(

    training_data=X_train.values,

    feature_names=X.columns.tolist(),

    class_names=['No Churn', 'Churn'],

    mode='classification'
)

lime_exp = explainer_lime.explain_instance(

    data_row=X_test.iloc[customer_index].values,

    predict_fn=rf_model.predict_proba,

    num_features=5
)

lime_features = lime_exp.as_list()

print("\nTop LIME Features:\n")

for feature, weight in lime_features:

    print(f"{feature} --> {weight:.4f}")

print("\n✅ PROJECT COMPLETED SUCCESSFULLY")