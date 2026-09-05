from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
BASE_DIR = Path(__file__).resolve().parent.parent



def load_data(path):
    df = pd.read_csv(path)
    df["YearMonth"] = pd.to_datetime(df["YearMonth"])
    latest_month = df["YearMonth"].max()
    df = df[df["YearMonth"] < latest_month].copy()

    df = df.sort_values(
        ["StockCode", "YearMonth"]
    ).reset_index(drop=True)    

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

    data_path = BASE_DIR / "data/processed/regression_features.csv"
    model_path = BASE_DIR / "models/sales_model.pkl"
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "predictions.csv"

    # Load data
    df = load_data(data_path)

    # Prepare test data (same cutoff!)
    X_test, y_test, test_df = prepare_test_data(
        df,
        date_col="YearMonth",
        target_col="target",
        cut_off="2011-08"
    )

    # Load trained model
    model = joblib.load(model_path)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

# Baseline: assume next month's quantity equals this month's quantity.
       
    baseline_pred = X_test["MonthlyQuantity"]

    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_pred)
    )

    baseline_r2 = r2_score(y_test, baseline_pred)

    print(f"Model MAE: {mae:.2f}")
    print(f"Model RMSE: {rmse:.2f}")
    print(f"Model R2: {r2:.4f}")

    print(f"Baseline MAE: {baseline_mae:.2f}")
    print(f"Baseline RMSE: {baseline_rmse:.2f}")
    print(f"Baseline R2: {baseline_r2:.4f}")

    # Save predictions
    test_df["Predicted"] = y_pred
    test_df.to_csv(output_path, index=False)

    print("Predictions saved at:", output_path)


if __name__ == "__main__":
    predict_pipeline()