import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

df = pd.read_csv(os.path.join(PROCESSED_DIR, "customer_cleaned.csv"))

features = [
    "age",
    "purchase_amount",
    "review_rating",
    "previous_purchases",
    "purchase_frequency_days"
]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow chart - helps you visually check if 6 clusters is a reasonable choice
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

cluster_names = {
    0: "Premium Customer",
    1: "Loyal Customer",
    2: "High-Value New Customer",
    3: "At-Risk Customer",
    4: "Regular Customer",
    5: "Budget Happy Customer"
}
df["Customer_Segment"] = df["Cluster"].map(cluster_names)

df.to_csv(os.path.join(PROCESSED_DIR, "customer_segmented.csv"), index=False)

cluster_summary = df.groupby("Cluster")[
    ["purchase_amount", "previous_purchases", "review_rating", "purchase_frequency_days"]
].mean()
print("Cluster Summary:\n", cluster_summary)
# Check this against cluster_names above - update names if the numbers don't match anymore

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(kmeans, os.path.join(MODEL_DIR, "customer_segmentation_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))