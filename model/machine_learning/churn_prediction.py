import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("processed_data/customer_segmented.csv")

print("\nDataset Loaded Successfully\n")
print(df.head())

# =====================================================
# CREATE BUSINESS-BASED CHURN SCORE
# =====================================================

df["Churn_Score"] = 0

# Long purchase gap
df.loc[df["purchase_frequency_days"] > 180, "Churn_Score"] += 3

# Few previous purchases
df.loc[df["previous_purchases"] < 10, "Churn_Score"] += 2

# Low purchase amount
df.loc[df["purchase_amount"] < 50, "Churn_Score"] += 1

# Poor review rating
df.loc[df["review_rating"] < 3.5, "Churn_Score"] += 1

# No subscription
df.loc[df["subscription_status"] == "No", "Churn_Score"] += 2

# Already at-risk customer
df.loc[
    df["Customer_Segment"] == "At-Risk Customer",
    "Churn_Score"
] += 3

# Final Churn Label
df["Churn"] = (df["Churn_Score"] >= 6).astype(int)

print("\nChurn Distribution\n")
print(df["Churn"].value_counts())

# =====================================================
# SAVE CHURN DATASET
# =====================================================

os.makedirs("processed_data", exist_ok=True)

df.to_csv(
    "processed_data/customer_churn.csv",
    index=False
)

print("\nCustomer churn dataset saved.\n")

# =====================================================
# FEATURE SELECTION
# =====================================================

features = [
    "age",
    "purchase_amount",
    "review_rating",
    "previous_purchases",
    "purchase_frequency_days",
    "Cluster"
]

X = df[features]
y = df["Churn"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# =====================================================
# RANDOM FOREST MODEL
# =====================================================

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    random_state=42
)

rf.fit(X_train, y_train)

print("\nRandom Forest Trained Successfully\n")

# =====================================================
# PREDICTION
# =====================================================

predictions = rf.predict(X_test)

# =====================================================
# EVALUATION
# =====================================================

accuracy = accuracy_score(y_test, predictions)

print("Accuracy :", round(accuracy, 4))

print("\nClassification Report\n")

print(classification_report(y_test, predictions))

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Confusion Matrix")

plt.show()

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance\n")

print(importance)

plt.figure(figsize=(8,5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.xticks(rotation=30)

plt.tight_layout()

plt.show()

# =====================================================
# SAVE MODEL
# =====================================================

os.makedirs("model", exist_ok=True)

joblib.dump(
    rf,
    "model/churn_prediction_model.pkl"
)

print("\nRandom Forest Model Saved")

# =====================================================
# SAVE FEATURE IMPORTANCE
# =====================================================

importance.to_csv(
    "processed_data/feature_importance.csv",
    index=False
)

print("Feature Importance Saved")

print("\nProject Completed Successfully")