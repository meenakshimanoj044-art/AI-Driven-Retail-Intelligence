import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "invoiceno",
    "customer_id",
    "transaction_time",
    "transaction_amount",
    "total_quantity",
    "is_return",
    "customer_transaction_count",
    "rapid_transaction_flag",
    "high_amount_flag",
    "high_quantity_flag",
    "risk_score",
    "risk_level",
    "risk_reason",
]


def prepare_risk_data(df):
    """Validate and prepare the existing transaction-level risk dataset."""

    df = df.copy()

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    df["transaction_time"] = pd.to_datetime(
        df["transaction_time"],
        errors="coerce"
    )

    df["transaction_amount"] = pd.to_numeric(
        df["transaction_amount"],
        errors="coerce"
    ).fillna(0)

    df["total_quantity"] = pd.to_numeric(
        df["total_quantity"],
        errors="coerce"
    ).fillna(0)

    df["risk_score"] = pd.to_numeric(
        df["risk_score"],
        errors="coerce"
    ).fillna(0).clip(0, 100)

    # Convert low/medium/high into Low/Medium/High.
    df["risk_level"] = (
        df["risk_level"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "low": "Low",
            "medium": "Medium",
            "high": "High"
        })
    )

    # If a risk level is missing, derive it from the risk score.
    missing_level = df["risk_level"].isna()

    df.loc[missing_level, "risk_level"] = pd.cut(
        df.loc[missing_level, "risk_score"],
        bins=[-1, 35, 65, 100],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    df["risk_reason"] = (
        df["risk_reason"]
        .fillna("No major risk indicator")
        .astype(str)
    )

    # Aliases used by the Streamlit dashboard.
    df["InvoiceNo"] = df["invoiceno"]
    df["CustomerID"] = df["customer_id"]
    df["InvoiceDate"] = df["transaction_time"]
    df["transaction_value"] = df["transaction_amount"]
    df["risk_reasons"] = df["risk_reason"]

    return df


def calculate_transaction_risk(df):
    """
    Uses the risk scores already created in risk_features.csv.

    Returns:
    - risk_score from 0–100
    - risk_level: Low, Medium or High
    - risk_reasons: explainable risk indicators
    """

    df = prepare_risk_data(df)

    # Recalculate missing or zero scores from existing risk indicators.
    calculated_score = (
        df["high_amount_flag"].fillna(0) * 25
        + df["high_quantity_flag"].fillna(0) * 20
        + df["rapid_transaction_flag"].fillna(0) * 25
        + df["is_return"].fillna(0) * 15
        + df["is_night_transaction"].fillna(0) * 10
        + df["is_unknown_customer"].fillna(0) * 5
    ).clip(0, 100)

    missing_score = df["risk_score"].isna()

    df.loc[missing_score, "risk_score"] = calculated_score[
        missing_score
    ]

    return df


def detect_fraud_spikes(df):
    """
    Detect abnormal daily increases in suspicious or high-risk
    transactions using a seven-day rolling baseline.
    """

    df = df.copy()

    df["risk_date"] = pd.to_datetime(
        df["transaction_time"],
        errors="coerce"
    ).dt.date

    df["high_risk_indicator"] = (
        (df["risk_level"] == "High")
        | (df["is_suspicious"] == 1)
    ).astype(int)

    daily = df.groupby("risk_date").agg(
        total_transactions=("invoiceno", "count"),
        high_risk_transactions=("high_risk_indicator", "sum"),
        total_amount=("transaction_amount", "sum")
    ).reset_index()

    risky_amount = (
        df[df["high_risk_indicator"] == 1]
        .groupby("risk_date")["transaction_amount"]
        .sum()
    )

    daily["risky_amount"] = (
        daily["risk_date"]
        .map(risky_amount)
        .fillna(0)
    )

    daily = daily.sort_values("risk_date").reset_index(drop=True)

    daily["rolling_mean"] = (
        daily["high_risk_transactions"]
        .shift(1)
        .rolling(window=7, min_periods=2)
        .mean()
    )

    daily["rolling_std"] = (
        daily["high_risk_transactions"]
        .shift(1)
        .rolling(window=7, min_periods=2)
        .std()
        .fillna(0)
    )

    daily["spike_threshold"] = (
        daily["rolling_mean"]
        + 2 * daily["rolling_std"]
    )

    daily["is_fraud_spike"] = (
        daily["rolling_mean"].notna()
        & (
            daily["high_risk_transactions"]
            > daily["spike_threshold"]
        )
        & (daily["high_risk_transactions"] >= 2)
    )

    daily["merchant_alert"] = np.where(
        daily["is_fraud_spike"],
        "ALERT: Abnormal increase in suspicious transactions",
        "Normal activity"
    )

    return daily


def detect_abuse_rings(df):
    """
    Detect suspicious customer groups using repeated transactions,
    returns, rapid transactions, unusually high amounts and quantities.

    The dataset does not contain device, IP or payment-instrument IDs,
    so these results are labelled abuse-ring candidates rather than
    confirmed fraud rings.
    """

    df = df.copy()

    customer_patterns = df.groupby(
        "customer_id",
        dropna=False
    ).agg(
        transaction_count=("invoiceno", "nunique"),
        return_count=("is_return", "sum"),
        rapid_transaction_count=("rapid_transaction_flag", "sum"),
        high_amount_count=("high_amount_flag", "sum"),
        high_quantity_count=("high_quantity_flag", "sum"),
        night_transaction_count=("is_night_transaction", "sum"),
        suspicious_transaction_count=("is_suspicious", "sum"),
        total_value=("transaction_amount", "sum"),
        average_risk=("risk_score", "mean"),
        first_transaction=("transaction_time", "min"),
        last_transaction=("transaction_time", "max")
    ).reset_index()

    customer_patterns["activity_hours"] = (
        customer_patterns["last_transaction"]
        - customer_patterns["first_transaction"]
    ).dt.total_seconds().div(3600).fillna(0)

    customer_patterns["ring_score"] = (
        np.minimum(
            customer_patterns["transaction_count"] / 10,
            1
        ) * 15
        + np.minimum(
            customer_patterns["return_count"] / 3,
            1
        ) * 20
        + np.minimum(
            customer_patterns["rapid_transaction_count"] / 3,
            1
        ) * 25
        + np.minimum(
            customer_patterns["high_amount_count"] / 3,
            1
        ) * 15
        + np.minimum(
            customer_patterns["high_quantity_count"] / 3,
            1
        ) * 10
        + (
            customer_patterns["average_risk"] / 100
        ) * 15
    ).clip(0, 100).round(2)

    def create_reason(row):
        reasons = []

        if row["return_count"] >= 2:
            reasons.append("repeated returns")

        if row["rapid_transaction_count"] >= 2:
            reasons.append("rapid repeat transactions")

        if row["high_amount_count"] >= 2:
            reasons.append("repeated high-value transactions")

        if row["high_quantity_count"] >= 2:
            reasons.append("repeated high-quantity transactions")

        if row["night_transaction_count"] >= 2:
            reasons.append("repeated night activity")

        return (
            ", ".join(reasons)
            if reasons
            else "multiple suspicious indicators"
        )

    customer_patterns["ring_reason"] = customer_patterns.apply(
        create_reason,
        axis=1
    )

    suspected_rings = customer_patterns[
        (customer_patterns["transaction_count"] >= 3)
        & (
            (customer_patterns["return_count"] >= 2)
            | (
                customer_patterns["rapid_transaction_count"]
                >= 2
            )
            | (
                customer_patterns["high_amount_count"]
                >= 2
            )
            | (
                customer_patterns["high_quantity_count"]
                >= 2
            )
            | (
                customer_patterns[
                    "suspicious_transaction_count"
                ] >= 2
            )
            | (customer_patterns["ring_score"] >= 50)
        )
    ]

    return suspected_rings.sort_values(
        "ring_score",
        ascending=False
    )