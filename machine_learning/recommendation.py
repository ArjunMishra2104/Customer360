"""
personalized_recommendation.py
===============================
A recommendation model that looks at THREE things together for each
customer, not just demographics:
    1. Their segment       (from segmentation.py)
    2. Their churn risk    (from churn_prediction.py)
    3. Their purchase history (category, product, spend, rating)

WHY THIS IS DIFFERENT FROM THE EARLIER recommendation.py:
The earlier version only used demographics + spend/rating tier to find
"similar" customers. This version pulls from customer_churn.csv (which
already has Cluster, Customer_Segment, and Churn from the earlier
scripts) so a customer's risk status and segment directly shape both
WHO they're matched against and WHAT gets recommended to them.

WHAT'S NEW:
- The profile text used for matching now includes the customer's churn
  status and segment, not just demographics. So two customers who
  bought from the same category but have different churn risk will be
  treated as less similar - which is the point, because what you'd
  recommend to a loyal customer vs. an at-risk one should differ.
- The exact product name is deliberately LEFT OUT of the matching
  profile (only category is used). If the exact product were included,
  TF-IDF would match customers almost entirely by "bought the identical
  item," which then leaves nothing left to recommend once that same
  item gets excluded from the results. The product name is still used
  to know what NOT to recommend (their own past purchase), just not to
  decide who counts as "similar."
- Each customer also gets a Recommended_Action, not just a product
  list. At-risk customers get a retention-style action; low-risk,
  high-value customers get an upsell-style action.

INPUT:  processed_data/customer_churn.csv (already has Cluster,
        Customer_Segment, and Churn columns from earlier scripts)
OUTPUT: processed_data/customer_personalized_recommendations.csv
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

df = pd.read_csv(os.path.join(PROCESSED_DIR, "customer_churn.csv"))


def purchase_level(amount):
    if amount >= 80:
        return "Premium"
    elif amount >= 50:
        return "Medium"
    else:
        return "Budget"


def rating_level(rating):
    if rating >= 4:
        return "High"
    elif rating >= 3:
        return "Medium"
    else:
        return "Low"


df["Purchase_Level"] = df["purchase_amount"].apply(purchase_level)
df["Rating_Level"] = df["review_rating"].apply(rating_level)
df["Churn_Label"] = df["Churn"].map({1: "ChurnRisk", 0: "Retained"})

# ---------------------------------------------------------------
# Build one profile per customer that blends segment + churn risk +
# purchase behavior into a single text string. This is what makes the
# similarity matching "see" all three things at once instead of just
# demographics.
# ---------------------------------------------------------------
df["Customer_Profile"] = (
    df["Customer_Segment"] + " "
    + df["Churn_Label"] + " "
    + df["category"] + " "
    + df["Purchase_Level"] + " "
    + df["Rating_Level"]
)

vectorizer = TfidfVectorizer()
profile_matrix = vectorizer.fit_transform(df["Customer_Profile"])
similarity_matrix = cosine_similarity(profile_matrix)

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "personalized_vectorizer.pkl"))
joblib.dump(similarity_matrix, os.path.join(MODEL_DIR, "personalized_similarity.pkl"))


def recommend_for_customer(customer_id, top_n=5):
    customer = df[df["customer_id"] == customer_id]
    if customer.empty:
        return []

    idx = customer.index[0]
    purchased_item = customer.iloc[0]["item_purchased"]

    scores = sorted(list(enumerate(similarity_matrix[idx])), key=lambda x: x[1], reverse=True)
    scores = scores[1:31]  # look at a wider pool of similar customers, skipping the customer themself

    recommendations = []
    for i, score in scores:
        product = df.iloc[i]["item_purchased"]
        if product != purchased_item and product not in recommendations:
            recommendations.append(product)
        if len(recommendations) == top_n:
            break
    return recommendations


def recommended_action(row):
    """
    Turns churn + segment into a plain-language next step, so the
    output isn't just a product list - it tells you what to actually
    DO for this customer.
    """
    if row["Churn"] == 1 and row["Customer_Segment"] in ("Premium Customer", "Loyal Customer"):
        return "Priority retention offer (high-value customer at risk)"
    elif row["Churn"] == 1:
        return "Send win-back discount"
    elif row["Customer_Segment"] in ("Premium Customer", "Loyal Customer"):
        return "Upsell premium / new arrivals"
    else:
        return "Standard product recommendation"


# ---------------------------------------------------------------
# Build the final per-customer table: segment + churn risk + purchase
# history all feed into both the recommended products AND the action.
# ---------------------------------------------------------------
rows = []
for customer_id in df["customer_id"]:
    row = df[df["customer_id"] == customer_id].iloc[0]
    recs = recommend_for_customer(customer_id)
    rows.append({
        "Customer_ID": customer_id,
        "Customer_Segment": row["Customer_Segment"],
        "Churn_Risk": "Yes" if row["Churn"] == 1 else "No",
        "Purchased_Product": row["item_purchased"],
        "Recommendation_1": recs[0] if len(recs) > 0 else "",
        "Recommendation_2": recs[1] if len(recs) > 1 else "",
        "Recommendation_3": recs[2] if len(recs) > 2 else "",
        "Recommendation_4": recs[3] if len(recs) > 3 else "",
        "Recommendation_5": recs[4] if len(recs) > 4 else "",
        "Recommended_Action": recommended_action(row),
    })

result_df = pd.DataFrame(rows)
result_df.to_csv(os.path.join(PROCESSED_DIR, "customer_personalized_recommendations.csv"), index=False)

print(f"Personalized recommendations built for {len(result_df)} customers")
print(result_df["Recommended_Action"].value_counts())