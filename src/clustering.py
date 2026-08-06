
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


REQUIRED_CLUSTER_FEATURES = [
    "TotalOrders",
    "ActiveMonth",
    "AvgOrderValue",
    "Recency",
    "PurchaseFrequency",
]



def load_path(path):
    df = pd.read_csv(path)
    return df


def validate_columns(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns for clustering: {', '.join(missing_columns)}"
        )
def create_training_data(df):
    validate_columns(df, REQUIRED_CLUSTER_FEATURES)
    X = df[REQUIRED_CLUSTER_FEATURES].copy()
    X = X.fillna(0)
    return X

def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def implement_k_means(X_scaled, df):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df = df.copy()
    df["clusters"] = kmeans.fit_predict(X_scaled)
    return df, kmeans


# Churn represents inactivity.
def add_churn_label(df):
    validate_columns(df, ["Recency"])
    df = df.copy()
    df["Churn"] = (df["Recency"] > 180).astype(int)
    return df


def label_customers(df):
    clusters_map = {
        0: "Dormant Customers",
        1: "Occasional Customers",
        2: "Frequent Customers",
    }
    df = df.copy()
    df["segment"] = df["clusters"].map(clusters_map).fillna("Unassigned")
    return df


def cluster_summary(df):
    validate_columns(
        df,
        ["clusters", "Recency", "TotalOrders", "PurchaseFrequency", "AvgOrderValue"],
    )
    return df.groupby("clusters")[
        ["Recency", "TotalOrders", "PurchaseFrequency", "AvgOrderValue"]
    ].mean()

def create_customer_text(df):
    def to_text(row):
        behavior = (
            "low engagement" if row["PurchaseFrequency"] < 0.5 else "high engagement"
        )
        recency_status = "inactive" if row["Recency"] > 180 else "recently active"
        return f"""
       Customer Profile:
- Orders: {row['TotalOrders']}
- Active Months: {row['ActiveMonth']}
- Avg Order Value: {row['AvgOrderValue']:.2f}
- Recency: {row['Recency']} ({recency_status})
- Frequency: {row['PurchaseFrequency']} ({behavior})
- Cluster: {row['clusters']}
- Churn Risk: {"Yes" if row['Churn'] == 1 else "No"}
"""

    validate_columns(df, REQUIRED_CLUSTER_FEATURES + ["clusters", "Churn"])
    df = df.copy()
    df["Customer_Text"] = df.apply(to_text, axis=1)
    return df

"""if __name__ == "__main__":

    base_dir = Path().resolve()
    data_path = base_dir/"data/processed/customer_churn_features.csv"

    df, kmeans, scaler = pipelines(data_path)

    # save output
    output_path = base_dir/"data/processed/customer_segments.csv"
    df.to_csv(output_path, index=False)

    # save models
    joblib.dump(kmeans,base_dir/"models/kmeans_model.pkl")
    joblib.dump(scaler,base_dir/"models/scaler.pkl")"""












"""def pipeline(data_path):

    df = load_path(data_path)

    X = create_training_data(df)

    X_scaled, scaler = scale_data(X)

    df, kmeans = implement_k_means(X_scaled, df)

    df = add_churn_label(df)

    df = label_customers(df)

    df = create_customer_text(df)
    df_samples=df.head(10).copy()
    df_samples= generate_insights(df_samples)
    df.loc[df_samples.index, "insights"] = df_samples["insights"]

    return df, kmeans, scaler"""









    


















