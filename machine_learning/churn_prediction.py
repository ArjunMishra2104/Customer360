import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, precision_score, recall_score, f1_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

df = pd.read_csv(os.path.join(PROCESSED_DIR, "customer_segmented.csv"))

# Business rule: add points for each warning sign, mark as churn if points >= 5
df["Churn_Score"] = 0
df.loc[df["purchase_frequency_days"] > 180, "Churn_Score"] += 3
df.loc[df["previous_purchases"] < 10, "Churn_Score"] += 2
df.loc[df["purchase_amount"] < 50, "Churn_Score"] += 1
df.loc[df["review_rating"] < 3.5, "Churn_Score"] += 1
df.loc[df["subscription_status"] == "No", "Churn_Score"] += 2
df.loc[df["Customer_Segment"] == "At-Risk Customer", "Churn_Score"] += 3
df["Churn"] = (df["Churn_Score"] >= 5).astype(int)
print("Churn counts:\n", df["Churn"].value_counts())

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"   # keeps the model from ignoring the smaller churn group
)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

print(f"Accuracy  : {accuracy_score(y_test, predictions):.4f}")
print(f"Precision : {precision_score(y_test, predictions):.4f}")
print(f"Recall    : {recall_score(y_test, predictions):.4f}")
print(f"F1 Score  : {f1_score(y_test, predictions):.4f}")
print("\n", classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Customer Churn Confusion Matrix")
plt.show()

cv_scores = cross_val_score(rf, X, y, cv=5)
print("Cross-validation average accuracy:", cv_scores.mean())

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)
print("\nFeature Importance:\n", importance)

plt.figure(figsize=(8, 5))
plt.bar(importance["Feature"], importance["Importance"])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(rf, os.path.join(MODEL_DIR, "churn_prediction_model.pkl"))
importance.to_csv(os.path.join(PROCESSED_DIR, "feature_importance.csv"), index=False)
df.to_csv(os.path.join(PROCESSED_DIR, "customer_churn.csv"), index=False)