import os
import numpy as np
import pandas as pd

RAW_PATH = "data/raw/online_retail.csv"
CLEAN_PATH = "data/cleaned/online_retail_cleaned.csv"
RISK_PATH = "data/processed/risk_features.csv"

os.makedirs("data/cleaned", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# 1. Load raw data
df = pd.read_csv(RAW_PATH, encoding="ISO-8859-1")

# Standardize column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Expected columns:
# invoiceNo, stockCode, description, quantity,
# invoiceDate, unitPrice, customerID, country

# 2. Convert data types
df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df["unitprice"] = pd.to_numeric(df["unitprice"], errors="coerce")

# Preserve missing customer IDs as anonymous customers
df["customerid"] = df["customerid"].fillna("UNKNOWN").astype(str)

# 3. Remove unusable records
df = df.dropna(
    subset=["invoiceno", "stockcode", "invoicedate", "quantity", "unitprice"]
)

df = df.drop_duplicates()

# Remove impossible price records
df = df[df["unitprice"] >= 0].copy()

# 4. Create basic fields
df["is_return"] = (
    df["invoiceno"].astype(str).str.startswith("C")
    | (df["quantity"] < 0)
).astype(int)

df["absolute_quantity"] = df["quantity"].abs()
df["line_amount"] = df["absolute_quantity"] * df["unitprice"]

df["transaction_hour"] = df["invoicedate"].dt.hour
df["transaction_date"] = df["invoicedate"].dt.date

# Save the cleaned line-item dataset
df.to_csv(CLEAN_PATH, index=False)

# 5. Convert line items into invoice-level transactions
risk = (
    df.groupby("invoiceno", as_index=False)
    .agg(
        customer_id=("customerid", "first"),
        transaction_time=("invoicedate", "min"),
        country=("country", "first"),
        transaction_amount=("line_amount", "sum"),
        total_quantity=("absolute_quantity", "sum"),
        unique_products=("stockcode", "nunique"),
        line_count=("stockcode", "size"),
        is_return=("is_return", "max"),
    )
)

risk["transaction_time"] = pd.to_datetime(risk["transaction_time"])
risk["transaction_hour"] = risk["transaction_time"].dt.hour
risk["is_night_transaction"] = (
    (risk["transaction_hour"] < 6)
    | (risk["transaction_hour"] >= 23)
).astype(int)

risk["is_unknown_customer"] = (
    risk["customer_id"] == "UNKNOWN"
).astype(int)

# 6. Customer behaviour features
risk = risk.sort_values(["customer_id", "transaction_time"])

risk["customer_avg_amount"] = (
    risk.groupby("customer_id")["transaction_amount"]
    .transform("mean")
)

risk["customer_std_amount"] = (
    risk.groupby("customer_id")["transaction_amount"]
    .transform("std")
    .fillna(0)
)

risk["customer_transaction_count"] = (
    risk.groupby("customer_id")["invoiceno"]
    .transform("count")
)

risk["previous_transaction_time"] = (
    risk.groupby("customer_id")["transaction_time"].shift(1)
)

risk["minutes_since_previous_transaction"] = (
    risk["transaction_time"] - risk["previous_transaction_time"]
).dt.total_seconds() / 60

risk["minutes_since_previous_transaction"] = (
    risk["minutes_since_previous_transaction"].fillna(999999)
)

# 7. Risk indicators
amount_threshold = risk["transaction_amount"].quantile(0.99)
quantity_threshold = risk["total_quantity"].quantile(0.99)

risk["high_amount_flag"] = (
    risk["transaction_amount"] >= amount_threshold
).astype(int)

risk["high_quantity_flag"] = (
    risk["total_quantity"] >= quantity_threshold
).astype(int)

risk["rapid_transaction_flag"] = (
    risk["minutes_since_previous_transaction"] <= 10
).astype(int)

risk["amount_deviation_flag"] = (
    risk["transaction_amount"]
    > risk["customer_avg_amount"] + 3 * risk["customer_std_amount"]
).astype(int)

# 8. Explainable rule-based risk score
risk["risk_score"] = (
    risk["high_amount_flag"] * 30
    + risk["high_quantity_flag"] * 15
    + risk["rapid_transaction_flag"] * 25
    + risk["amount_deviation_flag"] * 20
    + risk["is_unknown_customer"] * 5
    + risk["is_night_transaction"] * 5
).clip(0, 100)

risk["risk_level"] = pd.cut(
    risk["risk_score"],
    bins=[-1, 24, 49, 100],
    labels=["low", "medium", "high"]
)

# Proxy label—not confirmed real-world fraud
risk["is_suspicious"] = (risk["risk_score"] >= 50).astype(int)

# 9. Explain why each transaction was flagged
def create_reason(row):
    reasons = []

    if row["high_amount_flag"]:
        reasons.append("unusually high amount")
    if row["high_quantity_flag"]:
        reasons.append("unusually high quantity")
    if row["rapid_transaction_flag"]:
        reasons.append("rapid repeat transaction")
    if row["amount_deviation_flag"]:
        reasons.append("amount differs from customer behaviour")
    if row["is_unknown_customer"]:
        reasons.append("unknown customer")
    if row["is_night_transaction"]:
        reasons.append("unusual transaction time")

    return "; ".join(reasons) if reasons else "no major risk indicator"

risk["risk_reason"] = risk.apply(create_reason, axis=1)

risk.to_csv(RISK_PATH, index=False)

print(f"Cleaned rows: {len(df):,}")
print(f"Transactions created: {len(risk):,}")
print(f"Suspicious transactions: {risk['is_suspicious'].sum():,}")
print(f"Saved cleaned data to: {CLEAN_PATH}")
print(f"Saved risk features to: {RISK_PATH}")
print("\nRisk-level distribution:")
print(risk["risk_level"].value_counts())