import joblib
import pandas as pd
from pathlib import Path
import shap
import matplotlib.pyplot as plt




BASE_DIR = Path().resolve()

data_path = BASE_DIR / "data/processed/regression_features.csv"
model_path = BASE_DIR / "models/sales_model.pkl"

df = pd.read_csv(data_path)


df["YearMonth"] = pd.to_datetime(df["YearMonth"])

# Create target (same as train.py and predict.py)
df["target"] = df.groupby("StockCode")["MonthlyQuantity"].shift(-1)

# Remove last month for each product
df = df.dropna(subset=["target"])



cut_off = "2011-10"

df = df.sort_values("YearMonth")

test = df[df["YearMonth"] > cut_off]

X_test = test.drop(
    columns=[
        "target",
        "YearMonth",
        "StockCode"
    ]
)

y_test = test["target"]



model=joblib.load(model_path)

output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)
explainer = shap.Explainer(model, X_test)
shap_values=explainer(X_test)


# Summary Plot
plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()

plt.savefig(
    output_dir / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP Summary Plot saved successfully!")