import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent

def load_data(clean_path: str):
    df = pd.read_csv(clean_path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
    return df


def monthly_features(df: pd.DataFrame):
    monthly = df.groupby(['StockCode', 'YearMonth']).agg(
        MonthlyQuantity=('Quantity', 'sum'),
        AvgUnitPrice=('UnitPrice', 'mean'),
        HolidayCount=('isholiday', 'sum'),
        MonthlyRevenue=('Revenue', 'sum')
    ).reset_index()

    monthly = monthly.sort_values(['StockCode', 'YearMonth'])

    # Lag features
    for lag in [1, 2, 3]:
        monthly[f'lag{lag}_quantity'] = monthly.groupby('StockCode')['MonthlyQuantity'].shift(lag)

    # Rolling mean
    monthly["rolling_mean_3"] = (monthly.groupby("StockCode")["MonthlyQuantity"]
                                 .transform(
                                     lambda values: values.shift(1).rolling(
                                    window=3,
                                    min_periods=3
                                    ).mean()
                                    )
                                )

    # Growth rate
    monthly['GrowthRate'] = monthly.groupby('StockCode')['MonthlyQuantity'].pct_change(fill_method=None)
    monthly["GrowthRate"] = monthly["GrowthRate"].replace(
    [np.inf, -np.inf],
    np.nan
    )

    # Month feature
    
    monthly['Month'] = monthly['YearMonth'].dt.month
    

    # Lag revenue
    for lag in [1, 2, 3]:
        monthly[f'LagRevenue{lag}'] = monthly.groupby('StockCode')['MonthlyRevenue'].shift(lag)

    monthly = monthly.dropna()
    return monthly



def product_features(df: pd.DataFrame):

    monthly_product = df.groupby(['StockCode', 'YearMonth']).agg(
        MonthlyQuantity=('Quantity', 'sum'),
        MonthlyRevenue=('Revenue', 'sum'),
        TotalHolidayQuantity=('HolidayQuantity', 'sum')
    ).reset_index()

    product_level = monthly_product.groupby('StockCode').agg(
        TotalQuantitySold=('MonthlyQuantity', 'sum'),
        TotalRevenue=('MonthlyRevenue', 'sum'),
        AvgMonthlyQuantity=('MonthlyQuantity', 'mean'),
        AvgMonthlyRevenue=('MonthlyRevenue', 'mean'),
        ActiveMonthCount=('YearMonth', 'nunique'),
        StdMonthlyQuantity=('MonthlyQuantity', 'std'),
        PeakMonthSale=('MonthlyQuantity', 'max'),
        TotalHolidayQuantity=('TotalHolidayQuantity', 'sum')
    ).reset_index()

    product_level['HolidayRatio'] = (
        product_level['TotalHolidayQuantity'] /
        product_level['TotalQuantitySold']
    )

    pricing = df.groupby('StockCode').agg(
        AvgUnitPrice=('UnitPrice', 'mean'),
        PriceVariance=('UnitPrice', 'var')
    ).reset_index()

    customer_eng = df.groupby('StockCode').agg(
        UniqueCustomersCount=('CustomerID', 'nunique')
    ).reset_index()

    product_kpis = (
        product_level
        .merge(pricing, on='StockCode', how='left')
        .merge(customer_eng, on='StockCode', how='left')
    )

    product_kpis['HolidayRatio'] = product_kpis['HolidayRatio'].fillna(0)

    return product_kpis


def customer_features(df: pd.DataFrame):

    df['OrderValue'] = df['Quantity'] * df['UnitPrice']

    # Total orders
    total_orders = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    total_orders.columns = ['CustomerID', 'TotalOrders']

    # Recency
    reference_date = df['InvoiceDate'].max()

    recency = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    recency['Recency'] = (reference_date - recency['InvoiceDate']).dt.days
    recency = recency[['CustomerID', 'Recency']]

    # Avg Order Value
    order_revenue = df.groupby(['InvoiceNo', 'CustomerID'])['OrderValue'].sum().reset_index()
    avg_order_value = order_revenue.groupby('CustomerID')['OrderValue'].mean().reset_index()
    avg_order_value.columns = ['CustomerID', 'AvgOrderValue']

    # Active months
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
    active_months = df.groupby('CustomerID')['YearMonth'].nunique().reset_index()
    active_months.columns = ['CustomerID', 'ActiveMonth']

    # Merge all
    customer_kpis = total_orders.merge(active_months, on='CustomerID') \
                                .merge(avg_order_value, on='CustomerID') \
                                .merge(recency, on='CustomerID')

    customer_kpis['PurchaseFrequency'] = (
        customer_kpis['TotalOrders'] / customer_kpis['ActiveMonth']
    )

    # Outlier clipping
    for col in ['AvgOrderValue', 'PurchaseFrequency', 'Recency']:
        lower = customer_kpis[col].quantile(0.01)
        upper = customer_kpis[col].quantile(0.99)
        customer_kpis[col] = customer_kpis[col].clip(lower, upper)

    return customer_kpis

def run_feature_engineering(input_path: str,
                            output_dir: str = "../data/processed/"):

    df = load_data(input_path)

    output_dir = BASE_DIR / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Product regression features
    monthly_df = monthly_features(df)
    monthly_df.to_csv(output_dir / "regression_features.csv", index=False)

    # Product KPIs
    product_df = product_features(df)
    product_df.to_csv(output_dir / "product_classification_features.csv", index=False)

    # Customer churn features
    customer_df= customer_features(df)
    customer_df.to_csv(output_dir / "customer_churn_features.csv", index=False)

    
    
    
    
    print("Feature Engineering Completed Successfully!")


if __name__ == "__main__":
    cleaned_online_retail_path=BASE_DIR/"data/cleaned/online_retail_cleaned.csv"
    run_feature_engineering(cleaned_online_retail_path)
