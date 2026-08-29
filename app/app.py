from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))


from src.pipelines import pipeline
from src.clustering import cluster_summary
from src.llm_insights import generate_insights
import streamlit as st
import pandas as pd
import joblib
from src.risk_engine import (
    calculate_transaction_risk,
    detect_fraud_spikes,
    detect_abuse_rings
)




regression_data_path = BASE_DIR / "data/processed/regression_features.csv"
clustering_data_path = BASE_DIR / "data/processed/customer_churn_features.csv"
risk_data_path = BASE_DIR / "data/processed/risk_features.csv"
model_path = BASE_DIR / "models/sales_model.pkl"

st.title("📊 Smart Product Insights Hub")

reg_model = joblib.load(model_path)

@st.cache_resource
def load_pipeline():
    return pipeline(clustering_data_path)


tab1, tab2, tab3 = st.tabs([
    "Sales Prediction",
    "Customer Segments Overview",
    "AI Risk Manager"
])

with tab1:

    st.subheader("Select Product")

    df = pd.read_csv(regression_data_path)
    df["YearMonth"] = pd.to_datetime(df["YearMonth"])

    stock_codes = df["StockCode"].unique()
    selected_product = st.selectbox("Select Product", stock_codes)

    product_df = df[df["StockCode"] == selected_product].sort_values("YearMonth")

    if not product_df.empty:

        latest = product_df.iloc[-1]

        # Avoid Streamlit Arrow serialization issues with mixed scalar types.
        st.dataframe(latest.to_frame().T.astype(str))

        features = df.drop(columns=["target", "YearMonth", "StockCode"], errors="ignore").columns
        input_data = latest[features].to_frame().T

        if st.button("Predict for Selected Product"):
            pred = reg_model.predict(input_data)
            st.success(f"Predicted Sales: {pred[0]:.2f}")


with tab2:

    st.subheader("👥 Customer Segments Overview")

    try:
        st.markdown("### Cluster Summary")
        df_clusters, _, _ = load_pipeline()

        summary = cluster_summary(df_clusters)
        st.dataframe(summary)

        st.markdown("### Segment Distribution")
        st.bar_chart(df_clusters["segment"].value_counts())

        st.markdown("### Explore Customers")

        selected_segment = st.selectbox(
            "Select Segment",
            df_clusters["segment"].dropna().unique()
        )

        filtered_df = df_clusters[df_clusters["segment"] == selected_segment]

        st.write(f"Showing customers in: **{selected_segment}**")
        st.dataframe(filtered_df.head(20))

        st.markdown("### Customer Profile")

        if not filtered_df.empty:
            sample = filtered_df.sample(1, random_state=42)
            st.text(sample["Customer_Text"].values[0])
        else:
            st.info("No customers found in this segment.")

        st.markdown("### AI Insight Segment")

        if st.button("Generate Segment Insight", key="customer_button"):
            if filtered_df.empty:
                st.warning("Please choose a segment with at least one customer.")
            else:
                segment_summary = filtered_df[[
                    "Recency",
                    "TotalOrders",
                    "PurchaseFrequency",
                    "AvgOrderValue"
                ]].mean().to_frame().T

                text = f"""
Segment: {selected_segment}
Average Recency: {segment_summary['Recency'].values[0]:.2f}
Average Orders: {segment_summary['TotalOrders'].values[0]:.2f}
Average Frequency: {segment_summary['PurchaseFrequency'].values[0]:.2f}
Average Order Value: {segment_summary['AvgOrderValue'].values[0]:.2f}
"""

                temp_df = pd.DataFrame({"Customer_Text": [text]})

                with st.spinner("Generating insights..."):
                    temp_df = generate_insights(temp_df)
                    st.success(temp_df["insights"].values[0])
    except Exception as err:
        st.error(f"Unable to load customer segmentation tab: {err}")
with tab3:

    st.header("🛡️ RazorShield AI Risk Manager")
    st.caption(
        "Explainable transaction risk scoring, fraud-spike monitoring "
        "and abuse-ring detection"
    )

    try:
        # Load risk dataset
        risk_df = pd.read_csv(risk_data_path)

        # Run all three risk components
        scored_df = calculate_transaction_risk(risk_df)
        spike_df = detect_fraud_spikes(scored_df)
        ring_df = detect_abuse_rings(scored_df)

        # Ensure risk level can be compared safely
        scored_df["risk_level"] = scored_df["risk_level"].astype(str)

        high_risk_df = scored_df[
            scored_df["risk_level"] == "High"
        ]

        high_risk_count = len(high_risk_df)
        high_risk_value = high_risk_df["transaction_value"].sum()

        spike_count = (
            int(spike_df["is_fraud_spike"].sum())
            if not spike_df.empty
            else 0
        )

        # --------------------------
        # Summary metrics
        # --------------------------

        st.subheader("Risk Overview")

        metric1, metric2, metric3, metric4 = st.columns(4)

        metric1.metric(
            "Total Transactions",
            f"{len(scored_df):,}"
        )

        metric2.metric(
            "High-Risk Transactions",
            f"{high_risk_count:,}"
        )

        metric3.metric(
            "High-Risk Value",
            f"£{high_risk_value:,.2f}"
        )

        metric4.metric(
            "Fraud Spikes",
            spike_count
        )

        # --------------------------
        # Transaction Risk Scorer
        # --------------------------

        st.divider()
        st.subheader("1. Transaction Risk Scorer")

        st.write(
            "Each transaction receives a score from 0–100 and an "
            "explanation of the detected risk factors."
        )

        risk_distribution = (
            scored_df["risk_level"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
        )

        st.bar_chart(risk_distribution)

        selected_risk_level = st.selectbox(
            "Filter by risk level",
            ["All", "High", "Medium", "Low"]
        )

        if selected_risk_level == "All":
            displayed_transactions = scored_df
        else:
            displayed_transactions = scored_df[
                scored_df["risk_level"] == selected_risk_level
            ]

        preferred_columns = [
            "InvoiceNo",
            "CustomerID",
            "StockCode",
            "InvoiceDate",
            "transaction_value",
            "risk_score",
            "risk_level",
            "risk_reasons"
        ]

        # Only display columns that exist
        available_columns = [
            column
            for column in preferred_columns
            if column in displayed_transactions.columns
        ]

        st.dataframe(
            displayed_transactions[available_columns]
            .sort_values("risk_score", ascending=False)
            .head(200),
            use_container_width=True
        )

        # --------------------------
        # Fraud-Spike Detector
        # --------------------------

        st.divider()
        st.subheader("2. Fraud-Spike Detector")

        st.write(
            "A spike is generated when high-risk activity exceeds "
            "the recent rolling baseline."
        )

        if spike_df.empty:
            st.info("There is not enough data to calculate fraud spikes.")

        else:
            spike_chart = spike_df.copy()
            spike_chart["risk_date"] = pd.to_datetime(
                spike_chart["risk_date"]
            )

            spike_chart = spike_chart.set_index("risk_date")

            st.line_chart(
                spike_chart[
                    [
                        "high_risk_transactions",
                        "spike_threshold"
                    ]
                ]
            )

            detected_spikes = spike_df[
                spike_df["is_fraud_spike"]
            ]

            if detected_spikes.empty:
                st.success(
                    "No abnormal fraud spike was detected."
                )
            else:
                st.error(
                    f"Merchant alert: {len(detected_spikes)} abnormal "
                    "fraud spike(s) detected."
                )

                st.dataframe(
                    detected_spikes,
                    use_container_width=True
                )

        # --------------------------
        # Abuse-Ring Detector
        # --------------------------

        st.divider()
        st.subheader("3. Abuse-Ring Detector")

        st.write(
            "Suspicious customer-product groups are detected using "
            "repeated transactions, cancellations, large quantities "
            "and transaction risk."
        )

        if ring_df.empty:
            st.success("No suspected abuse rings were detected.")

        else:
            st.warning(
                f"{len(ring_df)} suspicious customer-product "
                "group(s) detected."
            )

            st.dataframe(
                ring_df.head(100),
                use_container_width=True
            )

        # --------------------------
        # Download evidence
        # --------------------------

        st.divider()
        st.subheader("Investigation Export")

        download1, download2, download3 = st.columns(3)

        with download1:
            st.download_button(
                "Download Risk Scores",
                scored_df.to_csv(index=False),
                file_name="scored_transactions.csv",
                mime="text/csv"
            )

        with download2:
            st.download_button(
                "Download Fraud Spikes",
                spike_df.to_csv(index=False),
                file_name="fraud_spikes.csv",
                mime="text/csv"
            )

        with download3:
            st.download_button(
                "Download Abuse Rings",
                ring_df.to_csv(index=False),
                file_name="abuse_rings.csv",
                mime="text/csv"
            )

    except FileNotFoundError:
        st.error(
            "Risk dataset not found. Expected file: "
            "data/processed/risk_features.csv"
        )

    except KeyError as err:
        st.error(
            f"Required column is missing from the risk dataset: {err}"
        )

    except Exception as err:
        st.error(f"Unable to load AI Risk Manager: {err}")
