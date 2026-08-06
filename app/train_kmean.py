from pathlib import Path
import joblib
from src.pipelines import pipeline

if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data/processed/customer_churn_features.csv"

    df, kmeans, scaler = pipeline(data_path)

    # save dataset
    output_path = base_dir / "data/processed/customer_segments.csv"
    df.to_csv(output_path, index=False)

    # save models
    joblib.dump(kmeans, base_dir / "models/kmeans_model.pkl")
    joblib.dump(scaler, base_dir / "models/scaler.pkl")

