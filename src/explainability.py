import joblib
import pandas as pd
from pathlib import Path
import numpy as np
import shap
import matplotlib.pyplot as plt




BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data/processed/regression_features.csv"
model_path = BASE_DIR / "models/sales_model.pkl"

df = pd.read_csv(data_path)
df["YearMonth"] = pd.to_datetime(df["YearMonth"])

latest_month = df["YearMonth"].max()
df = df[df["YearMonth"] < latest_month].copy()

df = df.sort_values(
    ["StockCode", "YearMonth"]
).reset_index(drop=True)

df["target"] = (
    df.groupby("StockCode")["MonthlyQuantity"]
    .shift(-1)
)

df = df.dropna(subset=["target"])

cut_off = "2011-08"
df = df.sort_values("YearMonth")

train = df[df["YearMonth"] <= cut_off].copy()
test = df[df["YearMonth"] > cut_off].copy()

columns_to_remove = [
    "target",
    "YearMonth",
    "StockCode"
]

X_train = train.drop(columns=columns_to_remove)
X_test = test.drop(columns=columns_to_remove)

y_train = train["target"]
y_test = test["target"]

if X_train.empty or X_test.empty:
    raise ValueError("Training or test data is empty.")

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and test feature columns do not match.")



model = joblib.load(model_path)

# Verify that the saved model has the expected pipeline structure.
if not hasattr(model, "named_steps"):
    raise TypeError("Expected a scikit-learn Pipeline.")

if "scaler" not in model.named_steps or "ridge" not in model.named_steps:
    raise KeyError("Pipeline must contain 'scaler' and 'ridge' steps.")

scaler = model.named_steps["scaler"]
ridge_model = model.named_steps["ridge"]

# Transform the data exactly as the forecasting pipeline does.
X_train_scaled = pd.DataFrame(
    scaler.transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# Use a reproducible sample of training data as the SHAP background.
background_size = min(500, len(X_train_scaled))
background_data = shap.sample(
    X_train_scaled,
    background_size,
    random_state=42
)

# LinearExplainer is appropriate for the Ridge model.
explainer = shap.LinearExplainer(
    ridge_model,
    background_data
)

raw_shap_values = explainer(X_test_scaled)

# Retain original feature values for readable plot labels.
shap_values = shap.Explanation(
    values=raw_shap_values.values,
    base_values=raw_shap_values.base_values,
    data=X_test.to_numpy(),
    feature_names=X_test.columns.tolist()
)

output_dir = BASE_DIR / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# Calculate and save numerical feature importance.
importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "MeanAbsoluteSHAP": np.abs(shap_values.values).mean(axis=0),
    "StandardizedRidgeCoefficient": ridge_model.coef_
})

importance_df = importance_df.sort_values(
    "MeanAbsoluteSHAP",
    ascending=False
).reset_index(drop=True)

importance_df.insert(
    0,
    "Rank",
    range(1, len(importance_df) + 1)
)

importance_df.to_csv(
    output_dir / "shap_feature_importance.csv",
    index=False
)

# Global SHAP importance bar chart.
plt.figure(figsize=(9, 6))

shap.summary_plot(
    shap_values.values,
    X_test,
    plot_type="bar",
    max_display=10,
    show=False
)

plt.xlabel("Mean absolute SHAP value")
plt.tight_layout()
plt.savefig(
    output_dir / "shap_global_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# SHAP beeswarm: importance and direction of influence.
plt.figure(figsize=(9, 6))

shap.summary_plot(
    shap_values.values,
    X_test,
    max_display=10,
    show=False
)

plt.tight_layout()
plt.savefig(
    output_dir / "shap_beeswarm.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Select a representative prediction with median absolute error.
predictions = model.predict(X_test)
absolute_errors = np.abs(y_test.to_numpy() - predictions)

sorted_error_positions = np.argsort(absolute_errors)
representative_position = int(
    sorted_error_positions[len(sorted_error_positions) // 2]
)

# Local explanation for the representative prediction.
shap.plots.waterfall(
    shap_values[representative_position],
    max_display=10,
    show=False
)

plt.tight_layout()
plt.savefig(
    output_dir / "shap_local_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Save information about the representative prediction.
representative_result = test.iloc[
    [representative_position]
][["StockCode", "YearMonth", "target"]].copy()

representative_result = representative_result.rename(
    columns={"target": "ActualNextMonthQuantity"}
)

representative_result["PredictedNextMonthQuantity"] = (
    predictions[representative_position]
)

representative_result["AbsoluteError"] = (
    absolute_errors[representative_position]
)

representative_result.to_csv(
    output_dir / "shap_representative_prediction.csv",
    index=False
)

print("Research-grade SHAP analysis completed successfully.")
print("\nTop forecasting features:")
print(importance_df.head(10).to_string(index=False))
print("\nOutputs saved in:", output_dir)