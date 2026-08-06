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




regression_data_path = BASE_DIR / "data/processed/regression_features.csv"
clustering_data_path = BASE_DIR / "data/processed/customer_churn_features.csv"
model_path = BASE_DIR / "models/sales_model.pkl"

st.title("📊 Smart Product Insights Hub")

reg_model = joblib.load(model_path)

@st.cache_resource
def load_pipeline():
    return pipeline(clustering_data_path)


tab1, tab2 = st.tabs(["Sales Prediction", "Customer Segments Overview"])

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