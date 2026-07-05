"""
streamlit/app.py
=================
Dashboard for Customer360. It just reads the CSV files your scripts
already made - it doesn't train or change anything.

The SQL panel connects to the same local database Power BI can also
use, so both tools show the same numbers instead of two different
copies.

Run with:  python -m streamlit run streamlit/app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DB_PATH = os.path.join(BASE_DIR, "customer360.db")

st.set_page_config(page_title="Customer360 Dashboard", layout="wide")
st.title("Customer360 Dashboard")

segmented_path = os.path.join(PROCESSED_DIR, "customer_segmented.csv")
churn_path = os.path.join(PROCESSED_DIR, "customer_churn.csv")
recs_path = os.path.join(PROCESSED_DIR, "customer_recommendations.csv")
personalized_path = os.path.join(PROCESSED_DIR, "customer_personalized_recommendations.csv")
rfm_path = os.path.join(PROCESSED_DIR, "customer_rfm.csv")
churn_model_path = os.path.join(MODEL_DIR, "churn_prediction_model.pkl")

if not os.path.exists(segmented_path):
    st.error("customer_segmented.csv not found. Run segmentation.py first.")
    st.stop()

df = pd.read_csv(segmented_path)

# ---- Overview ----
st.header("Overview")
col1, col2 = st.columns(2)
col1.metric("Total Customers", len(df))
col2.metric("Segments", df["Customer_Segment"].nunique())
st.bar_chart(df["Customer_Segment"].value_counts())

# ---- Segmentation ----
st.header("Segments (K-Means)")
segment = st.selectbox("Filter by segment", ["All"] + sorted(df["Customer_Segment"].unique()))
if segment != "All":
    st.dataframe(df[df["Customer_Segment"] == segment])
else:
    st.dataframe(df)

# ---- RFM ----
st.header("RFM Segments")
if os.path.exists(rfm_path):
    rfm_df = pd.read_csv(rfm_path)
    st.caption("A simpler, easier-to-explain way of grouping customers, "
               "based on Recency, Frequency, and Monetary value.")
    st.bar_chart(rfm_df["RFM_Segment"].value_counts())
else:
    st.info("Run rfm_analysis.py to see RFM segments here.")

# ---- Churn ----
st.header("Churn")
if os.path.exists(churn_path):
    churn_df = pd.read_csv(churn_path)
    st.write(f"Churn rate: {churn_df['Churn'].mean() * 100:.1f}%")
    st.bar_chart(churn_df["Churn"].map({0: "Retained", 1: "Churned"}).value_counts())

    if os.path.exists(churn_model_path):
        st.subheader("Predict churn for a customer")
        model = joblib.load(churn_model_path)

        age = st.number_input("Age", 18, 100, 35)
        purchase_amount = st.number_input("Purchase Amount", 0.0, 1000.0, 60.0)
        review_rating = st.slider("Review Rating", 1.0, 5.0, 3.5)
        previous_purchases = st.number_input("Previous Purchases", 0, 200, 10)
        purchase_frequency_days = st.number_input("Purchase Frequency (days)", 0, 400, 90)
        cluster = st.selectbox("Cluster", sorted(df["Cluster"].unique()))

        if st.button("Predict"):
            row = pd.DataFrame([{
                "age": age,
                "purchase_amount": purchase_amount,
                "review_rating": review_rating,
                "previous_purchases": previous_purchases,
                "purchase_frequency_days": purchase_frequency_days,
                "Cluster": cluster,
            }])
            prediction = model.predict(row)[0]
            st.write("Prediction:", "Will Churn" if prediction == 1 else "Will Stay")
else:
    st.info("Run churn_prediction.py to see churn data here.")

# ---- Recommendations ----
st.header("Recommendations")
if os.path.exists(recs_path):
    recs_df = pd.read_csv(recs_path)
    customer_id = st.selectbox("Customer ID", recs_df["Customer_ID"].unique())
    row = recs_df[recs_df["Customer_ID"] == customer_id].iloc[0]
    st.write("Purchased:", row["Purchased_Product"])
    for i in range(1, 6):
        col = f"Recommendation_{i}"
        if row.get(col):
            st.write(f"{i}. {row[col]}")
else:
    st.info("Run recommendation.py to see recommendations here.")

# ---- Personalized Recommendations ----
st.header("Personalized Recommendations")
st.caption("Looks at each customer's segment, churn risk, and purchase history together - "
           "gives a recommended action, not just a list of products.")
if os.path.exists(personalized_path):
    p_df = pd.read_csv(personalized_path)

    c1, c2 = st.columns(2)
    with c1:
        risk_filter = st.selectbox("Filter by churn risk", ["All", "Yes", "No"])
    with c2:
        seg_filter = st.selectbox("Filter by segment", ["All"] + sorted(p_df["Customer_Segment"].unique()), key="p_seg")

    filtered = p_df.copy()
    if risk_filter != "All":
        filtered = filtered[filtered["Churn_Risk"] == risk_filter]
    if seg_filter != "All":
        filtered = filtered[filtered["Customer_Segment"] == seg_filter]

    p_customer_id = st.selectbox("Customer ID", filtered["Customer_ID"].unique())
    p_row = filtered[filtered["Customer_ID"] == p_customer_id].iloc[0]

    st.write("Segment:", p_row["Customer_Segment"])
    st.write("Churn Risk:", p_row["Churn_Risk"])
    st.write("Purchased:", p_row["Purchased_Product"])
    st.info(f"Recommended Action: {p_row['Recommended_Action']}")
    for i in range(1, 6):
        col = f"Recommendation_{i}"
        if p_row.get(col):
            st.write(f"{i}. {p_row[col]}")

    st.divider()
    st.dataframe(filtered, use_container_width=True, height=300)
else:
    st.info("Run personalized_recommendation.py to see this section.")

# ---- SQL panel - connects to the shared database, same one Power BI can use ----
st.header("SQL Query")
db_exists = os.path.exists(DB_PATH)

if db_exists:
    st.success(f"Connected to local database: customer360.db")
    default_conn = f"sqlite:///{DB_PATH}"
else:
    st.warning(
        "No local database found yet. Run `python models/load_to_sql.py` first "
        "to build customer360.db from your processed CSVs, then reload this page."
    )
    default_conn = ""

st.caption("Power BI should point at this same database, so both tools show the "
           "same numbers instead of separate copies.")

conn_string = st.text_input("Connection string", value=default_conn,
                             help="Filled in automatically if the local database exists. "
                                  "Replace with a Postgres/MySQL/SQL Server string if you're "
                                  "using a shared database instead.")
query = st.text_area("SQL query", "SELECT * FROM customer LIMIT 100;")

if st.button("Run Query"):
    if not conn_string:
        st.warning("Add a connection string first (or run load_to_sql.py).")
    else:
        try:
            from sqlalchemy import create_engine
            engine = create_engine(conn_string)
            result = pd.read_sql(query, engine)
            st.dataframe(result, use_container_width=True)
            st.download_button("Download as CSV", result.to_csv(index=False), file_name="query_result.csv")
        except Exception as e:
            st.error(f"Query failed: {e}")