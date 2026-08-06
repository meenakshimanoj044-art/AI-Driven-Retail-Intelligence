from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score


def load_data(path):
    df = pd.read_csv(path)
    df["YearMonth"] = pd.to_datetime(df["YearMonth"])

    # SAME target logic as train.py
    df["target"] = df.groupby("StockCode")["MonthlyQuantity"].shift(-1)
    df = df.dropna(subset=["target"])

    return df


def prepare_test_data(df, date_col, target_col, cut_off):
    df = df.sort_values(date_col)

    test = df[df[date_col] > cut_off]

    X_test = test.drop(columns=[target_col, date_col, "StockCode"])
    y_test = test[target_col]

    return X_test, y_test, test


def predict_pipeline():
    BASE_DIR = Path().resolve()

    data_path = BASE_DIR / "data/processed/regression_features.csv"
    model_path = BASE_DIR / "models/sales_model.pkl"
    output_path = BASE_DIR / "data/predictions.csv"

    # Load data
    df = load_data(data_path)

    # Prepare test data (same cutoff!)
    X_test, y_test, test_df = prepare_test_data(
        df,
        date_col="YearMonth",
        target_col="target",
        cut_off="2011-10"
    )

    # Load trained model
    model = joblib.load(model_path)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")

    # Save predictions
    test_df["Predicted"] = y_pred
    test_df.to_csv(output_path, index=False)

    print("Predictions saved at:", output_path)


if __name__ == "__main__":
    predict_pipeline()