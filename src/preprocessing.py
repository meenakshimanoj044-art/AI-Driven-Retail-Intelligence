import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent
def load_data():
    retail_path=BASE_DIR/"data/raw/online_retail.csv"
    holidays_path=BASE_DIR/"data/external/publicHolidays.csv"
    raw_online_retail = pd.read_csv(retail_path)
    raw_public_holidays = pd.read_csv(holidays_path)
    return raw_online_retail, raw_public_holidays



def initial_cleaning(df):
    df = df.copy()

    # Drop missing CustomerID
    df = df.dropna(subset=["CustomerID"])

    # Remove duplicates
    df = df.drop_duplicates(subset=["InvoiceNo", "StockCode", "Quantity", "UnitPrice"])

    # Convert datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    return df


def split_datasets(df):
    purchased_dataset = df[df["Quantity"] > 0].copy()
    invalid_purchase_dataset = df[df["Quantity"] == 0].copy()
    return_dataset = df[df["Quantity"] < 0].copy()

    return purchased_dataset, invalid_purchase_dataset, return_dataset



def standardize_text(df):
    df = df.copy()

    df = df[df["StockCode"].notnull()]
    df["Description"] = df["Description"].astype(str).str.strip().str.lower()

    return df



def handle_outliers(df):
    df = df.copy()

    # ---- Quantity Outliers ----
    Q1_q = df["Quantity"].quantile(0.25)
    Q3_q = df["Quantity"].quantile(0.75)
    IQR_q = Q3_q - Q1_q
    upper_q = Q3_q + 1.5 * IQR_q

    df["Quantity"] = df["Quantity"].clip(upper=upper_q)

    # ---- UnitPrice Outliers ----
    Q1_u = df["UnitPrice"].quantile(0.25)
    Q3_u = df["UnitPrice"].quantile(0.75)
    IQR_u = Q3_u - Q1_u
    lower_u = Q1_u - 1.5 * IQR_u
    upper_u = Q3_u + 1.5 * IQR_u

    df = df[(df["UnitPrice"] > 0) & (df["UnitPrice"] < upper_u)]

    return df



def feature_engineering(df):
    df = df.copy()

    # Revenue
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Time features
    df = df.sort_values(by=["CustomerID", "InvoiceDate"])
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["day_of_week"] = df["InvoiceDate"].dt.weekday
    df["week"] = df["InvoiceDate"].dt.isocalendar().week.astype(int)

    # Cohort
    df["CohortMonth"] = (
        df.groupby("CustomerID")["InvoiceDate"]
        .transform("min")
        .dt.to_period("M")
    )

    return df



def process_holidays(df):
    df = df.copy()

    df.columns = df.columns.str.lower().str.strip()
    df = df[["date", "countryorregion", "holidayname"]]

    df["date"] = pd.to_datetime(df["date"])

    df = df.dropna(subset=["date", "countryorregion"])
    df = df.drop_duplicates(subset=["date", "countryorregion"])

    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["date"].dt.weekday > 5
    df["month_end"] = df["date"].dt.is_month_end
    df["month_start"] = df["date"].dt.is_month_start

    df["holidayname"] = df["holidayname"].astype(str).str.strip()
    df["isholiday"] = df["holidayname"].notna().astype(int)

    return df


def merge_data(retail_df, holiday_df):

    # FIX: align dates properly (IMPORTANT from your notebook issue)
    retail_df["InvoiceDate"] = pd.to_datetime(retail_df["InvoiceDate"]).dt.date
    holiday_df["date"] = pd.to_datetime(holiday_df["date"]).dt.date

    df = pd.merge(
        retail_df,
        holiday_df,
        left_on="InvoiceDate",
        right_on="date",
        how="left"
    )

    # Fill missing holiday info
    df["isholiday"] = df["isholiday"].fillna(0).astype(int)
    df["holidayname"] = df["holidayname"].fillna("Non-holiday")

    # Holiday impact feature
    df["HolidayQuantity"] = df["Quantity"] * df["isholiday"]

    # Drop unnecessary nulls
    df = df.dropna(subset=["InvoiceNo", "StockCode", "Quantity", "UnitPrice", "InvoiceDate", "Revenue"])

    return df



def final_clean(df):
    df = df.copy()

    # Rename Month
    if "Month" in df.columns:
        df = df.rename(columns={"Month": "Month_num"})

    # YearMonth feature
    df["YearMonth"] = pd.to_datetime(df["InvoiceDate"]).dt.to_period("M")

    return df



def save(df):
    output_path=BASE_DIR/"data"/"cleaned"/"online_retail_cleaned.csv"
    output_path.parent.mkdir(parents=True,exist_ok=True)

    df.to_csv(output_path, index=False)
    df.to_csv(BASE_DIR / "online_retail_clean_backup.csv", index=False)



def run_pipeline():
    retail, holidays = load_data()

    retail = initial_cleaning(retail)

    purchased, invalid, returns = split_datasets(retail)

    purchased = standardize_text(purchased)
    purchased = handle_outliers(purchased)
    purchased = feature_engineering(purchased)

    holidays = process_holidays(holidays)

    final_df = merge_data(purchased, holidays)
    final_df = final_clean(final_df)

    save(final_df)

    print("Preprocessing completed")


if __name__ == "__main__":
    run_pipeline()