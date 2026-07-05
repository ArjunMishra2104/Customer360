import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("processed_data/customer_cleaned.csv")
print(df.head())
features = [
    "age",
    "purchase_amount",
    "review_rating",
    "previous_purchases",
    "purchase_frequency_days"
]

X = df[features]

print(X.head())
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(X_scaled)
wcss = []

for k in range(1,11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)
    plt.figure(figsize=(8,5))

plt.plot(range(1,11), wcss, marker="o")

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.grid(True)
plt.show()
# Final KMeans Model
kmeans = KMeans(
    n_clusters=6,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)

print(clusters)
df["Cluster"] = clusters
cluster_names = {
    0: "Premium Customer",
    1: "Loyal Customer",
    2: "High-Value New Customer",
    3: "At-Risk Customer",
    4: "Regular Customer",
    5: "Budget Happy Customer"
}

df["Customer_Segment"] = df["Cluster"].map(cluster_names)
print(df[["customer_id", "Cluster"]].head(10))
df.to_csv(
    "processed_data/customer_segmented.csv",
    index=False
)

print("Customer segmentation completed successfully!")
print("\n================ Cluster Summary ================\n")

cluster_summary = df.groupby("Cluster")[
    [
        "purchase_amount",
        "previous_purchases",
        "review_rating",
        "purchase_frequency_days"
    ]
].mean()

print(cluster_summary)
os.makedirs("model", exist_ok=True)

joblib.dump(kmeans, "model/customer_segmentation_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model and scaler saved successfully!")