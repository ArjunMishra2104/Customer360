"""
load_to_sql.py
===============
Loads your processed CSVs into a real, local SQL database (SQLite) so
the SQL query file and the Streamlit "SQL Query" panel can actually run
against something, instead of sitting empty.

Why SQLite: it needs no server, no installation, and no password - it's
just one file (customer360.db) sitting in your project folder. It's the
fastest way to test your 19 SQL queries locally. When you're ready for
Power BI or a shared/production setup, swap this for Postgres, MySQL,
or SQL Server - the SQL queries themselves barely need to change.

TABLES CREATED (matching what your SQL file expects):
    customer        <- from customer_churn.csv (has every column
                        Q1-Q18 need: gender, purchase_amount, category,
                        Customer_Segment, Churn, etc.)
    recommendations <- from customer_personalized_recommendations.csv
                        (has Recommendation_1, needed for Q19)

Run this any time after your pipeline scripts finish, to refresh the
database with the latest data:
    python models/load_to_sql.py
"""

import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
DB_PATH = os.path.join(BASE_DIR, "customer360.db")

customer_csv = os.path.join(PROCESSED_DIR, "customer_churn.csv")
recs_csv = os.path.join(PROCESSED_DIR, "customer_personalized_recommendations.csv")

if not os.path.exists(customer_csv):
    raise FileNotFoundError(
        f"Could not find {customer_csv}. Run segmentation.py then "
        f"churn_prediction.py first - this script loads their output."
    )

conn = sqlite3.connect(DB_PATH)

customer_df = pd.read_csv(customer_csv)
customer_df.to_sql("customer", conn, if_exists="replace", index=False)
print(f"Loaded {len(customer_df)} rows into table 'customer'")

if os.path.exists(recs_csv):
    recs_df = pd.read_csv(recs_csv)
    recs_df.to_sql("recommendations", conn, if_exists="replace", index=False)
    print(f"Loaded {len(recs_df)} rows into table 'recommendations'")
else:
    print("customer_personalized_recommendations.csv not found - skipping "
          "'recommendations' table. Run personalized_recommendation.py first if you need Q19.")

conn.close()
print(f"\nDatabase ready at: {DB_PATH}")
print(f"Connection string for the Streamlit SQL panel:\n  sqlite:///{DB_PATH}")