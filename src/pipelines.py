from src.clustering import (
    load_path,
    validate_columns,
    create_training_data,
    scale_data,
    implement_k_means,
    add_churn_label,
    label_customers,
    create_customer_text
)

def pipeline(data_path):

    df = load_path(data_path)
    validate_columns(
        df,
        ["TotalOrders", "ActiveMonth", "AvgOrderValue", "Recency", "PurchaseFrequency"],
    )

    X = create_training_data(df)

    X_scaled, scaler = scale_data(X)

    df, kmeans = implement_k_means(X_scaled, df)

    df = add_churn_label(df)
    df = label_customers(df)
    df = create_customer_text(df)

    return df, kmeans, scaler