# ============================================================
# CUSTOMER360 AI
# Personalized Customer Recommendation System
# ============================================================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("="*60)
print("CUSTOMER360 PERSONALIZED RECOMMENDATION SYSTEM")
print("="*60)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("processed_data/customer_segmented.csv")

print("\nDataset Loaded Successfully")

print(df.head())

print("\nDataset Shape :", df.shape)

# ============================================================
# PURCHASE LEVEL
# ============================================================

def purchase_level(amount):

    if amount >= 80:
        return "Premium"

    elif amount >= 50:
        return "Medium"

    else:
        return "Budget"

df["Purchase_Level"] = df["purchase_amount"].apply(purchase_level)

# ============================================================
# RATING LEVEL
# ============================================================

def rating_level(rating):

    if rating >= 4:
        return "High"

    elif rating >= 3:
        return "Medium"

    else:
        return "Low"

df["Rating_Level"] = df["review_rating"].apply(rating_level)

# ============================================================
# CUSTOMER PROFILE
# ============================================================

df["Customer_Profile"] = (

    df["Customer_Segment"]

    + " "

    + df["gender"]

    + " "

    + df["age_group"]

    + " "

    + df["category"]

    + " "

    + df["Purchase_Level"]

    + " "

    + df["Rating_Level"]

)

print("\nCustomer Profiles Created")

print(df["Customer_Profile"].head())

# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer()

profile_matrix = vectorizer.fit_transform(
    df["Customer_Profile"]
)

print("\nTF-IDF Matrix Shape")

print(profile_matrix.shape)

# ============================================================
# COSINE SIMILARITY
# ============================================================

similarity_matrix = cosine_similarity(
    profile_matrix
)

print("\nSimilarity Matrix Shape")

print(similarity_matrix.shape)

# ============================================================
# SAVE MODELS
# ============================================================

os.makedirs("model", exist_ok=True)

joblib.dump(
    vectorizer,
    "model/recommendation_vectorizer.pkl"
)

joblib.dump(
    similarity_matrix,
    "model/customer_similarity.pkl"
)

print("\nVectorizer Saved")

print("Similarity Matrix Saved")
# =====================================================
# RECOMMENDATION FUNCTION
# =====================================================

def recommend_products(product_name, top_n=5):

    product_name = product_name.lower()

    # Find all rows with the product
    matches = df[
        df["item_purchased"].str.lower() == product_name
    ]

    if matches.empty:
        print("\nProduct Not Found!")
        return []

    # -----------------------------
    # Average similarity scores
    # -----------------------------

    similarity_scores = np.zeros(len(df))

    for idx in matches.index:
        similarity_scores += similarity_matrix[idx]

    similarity_scores = similarity_scores / len(matches)

    scores = list(enumerate(similarity_scores))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in scores:

        item = df.iloc[index]["item_purchased"]

        if item.lower() != product_name:

            if item not in recommendations:

                recommendations.append(item)

        if len(recommendations) == top_n:
            break

    return recommendations


# =====================================================
# TEST THE MODEL
# =====================================================

print("\n" + "="*60)
print("TESTING RECOMMENDATION SYSTEM")
print("="*60)

sample_products = [
    "Jacket",
    "Jeans",
    "Shoes",
    "Dress",
    "Sweater"
]

for product in sample_products:

    print("\n--------------------------------")

    print("Product :", product)

    recommendations = recommend_products(product)

    if recommendations:

        print("Recommended Products:")

        for i, item in enumerate(recommendations, start=1):

            print(f"{i}. {item}")

    else:

        print("No Recommendation Found")


# =====================================================
# CREATE RECOMMENDATION TABLE
# =====================================================

print("\nCreating Recommendation Dataset...\n")

recommendation_rows = []

unique_products = sorted(
    df["item_purchased"].unique()
)

for product in unique_products:

    recommendations = recommend_products(product)

    row = {

        "Product": product,

        "Recommendation_1": recommendations[0] if len(recommendations) > 0 else "",

        "Recommendation_2": recommendations[1] if len(recommendations) > 1 else "",

        "Recommendation_3": recommendations[2] if len(recommendations) > 2 else "",

        "Recommendation_4": recommendations[3] if len(recommendations) > 3 else "",

        "Recommendation_5": recommendations[4] if len(recommendations) > 4 else ""

    }

    recommendation_rows.append(row)

recommendation_df = pd.DataFrame(
    recommendation_rows
)

# =====================================================
# SAVE RECOMMENDATION CSV
# =====================================================

os.makedirs(
    "processed_data",
    exist_ok=True
)

recommendation_df.to_csv(

    "processed_data/product_recommendations.csv",

    index=False

)

print("Recommendation CSV Saved Successfully")

print(recommendation_df.head())


# =====================================================
# SAVE COMPLETE DATASET
# =====================================================

df.to_csv(

    "processed_data/customer_recommendation_dataset.csv",

    index=False

)

print("\nRecommendation Dataset Saved")


# =====================================================
# SAVE VECTOR FEATURES
# =====================================================

feature_names = vectorizer.get_feature_names_out()

feature_df = pd.DataFrame({

    "Feature": feature_names

})

feature_df.to_csv(

    "processed_data/tfidf_features.csv",

    index=False

)

print("TF-IDF Features Saved")


# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)

print(f"Total Customers : {len(df)}")

print(f"Unique Products : {df['item_purchased'].nunique()}")

print(f"Customer Segments : {df['Customer_Segment'].nunique()}")

print(f"Categories : {df['category'].nunique()}")

print(f"Recommendation Profiles : {len(df['Profile'])}")

print("\nFiles Generated")

print("--------------------------------")

print("✓ recommendation_vectorizer.pkl")

print("✓ similarity_matrix.pkl")

print("✓ product_recommendations.csv")

print("✓ customer_recommendation_dataset.csv")

print("✓ tfidf_features.csv")

print("\nRecommendation Engine Completed Successfully!")
# ============================================================
# PERSONALIZED RECOMMENDATION FUNCTION
# ============================================================

def recommend_products(customer_id, top_n=5):

    # ----------------------------------------
    # Find the customer
    # ----------------------------------------

    customer = df[df["customer_id"] == customer_id]

    if customer.empty:

        print("Customer Not Found")

        return []

    customer_index = customer.index[0]

    purchased_item = customer.iloc[0]["item_purchased"]

    # ----------------------------------------
    # Find Similar Customers
    # ----------------------------------------

    similarity_scores = list(
        enumerate(similarity_matrix[customer_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Skip the customer himself
    similarity_scores = similarity_scores[1:11]

    recommendations = []

    for index, score in similarity_scores:

        product = df.iloc[index]["item_purchased"]

        if product != purchased_item:

            if product not in recommendations:

                recommendations.append(product)

        if len(recommendations) == top_n:

            break

    return recommendations
# ============================================================
# TEST
# ============================================================

print("\n")
print("="*60)
print("TESTING RECOMMENDATION SYSTEM")
print("="*60)

customer = 1

recommendations = recommend_products(customer)

print(f"\nCustomer ID : {customer}")

print(
    "Purchased :",
    df[df["customer_id"]==customer]["item_purchased"].values[0]
)

print("\nRecommended Products")

for i,item in enumerate(recommendations,start=1):

    print(f"{i}. {item}")
    # ============================================================
# CREATE RECOMMENDATION DATASET
# ============================================================

recommendation_rows = []

for customer in df["customer_id"]:

    recommendations = recommend_products(customer)

    purchased = df[
        df["customer_id"]==customer
    ]["item_purchased"].values[0]

    segment = df[
        df["customer_id"]==customer
    ]["Customer_Segment"].values[0]

    recommendation_rows.append({

        "Customer_ID": customer,

        "Customer_Segment": segment,

        "Purchased_Product": purchased,

        "Recommendation_1":
        recommendations[0] if len(recommendations)>0 else "",

        "Recommendation_2":
        recommendations[1] if len(recommendations)>1 else "",

        "Recommendation_3":
        recommendations[2] if len(recommendations)>2 else "",

        "Recommendation_4":
        recommendations[3] if len(recommendations)>3 else "",

        "Recommendation_5":
        recommendations[4] if len(recommendations)>4 else ""

    })

recommendation_df = pd.DataFrame(
    recommendation_rows
)
# ============================================================
# SAVE FILE
# ============================================================

os.makedirs(
    "processed_data",
    exist_ok=True
)

recommendation_df.to_csv(

    "processed_data/customer_recommendations.csv",

    index=False

)

print("\nRecommendations Saved Successfully")

print("\n")

print(recommendation_df.head())