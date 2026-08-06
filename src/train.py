from pathlib import Path
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV
from sklearn.linear_model import Ridge
import joblib


def load_data(path):
    df=pd.read_csv(path)
    df["YearMonth"]=pd.to_datetime(df["YearMonth"])
    df["target"] = df.groupby("StockCode")["MonthlyQuantity"].shift(-1)
    df = df.dropna(subset=["target"])
    return df

def time_split(df, date_col, target_col, cut_off):

    df = df.sort_values(date_col)

    train = df[df[date_col] <= cut_off]
    test = df[df[date_col] > cut_off]
    
    X_train = train.drop(columns=[target_col, date_col, "StockCode"])
    y_train = train[target_col]

    X_test = test.drop(columns=[target_col, date_col, "StockCode"])
    y_test = test[target_col]

    return X_train, X_test, y_train, y_test

def train_model(X_train,y_train):
    tscv=TimeSeriesSplit(n_splits=5)
    model=Ridge()
    param_grid={
        "alpha": [0.1,1,10,50,100]
    }
    grid=GridSearchCV(
        model,
        param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error"

    )
    grid.fit(X_train,y_train)
    return grid.best_estimator_,grid.best_params_

def train_pipeline():
    BASE_DIR = Path().resolve()

    data_path = BASE_DIR / "data/processed/regression_features.csv"
    model_path = BASE_DIR / "models/sales_model.pkl"

    # ✅ FIXED HERE
    df = load_data(data_path)

    X_train, X_test, y_train, y_test = time_split(
        df,
        date_col="YearMonth",
        target_col="target",
        cut_off="2011-10"
    )

    model, best_params = train_model(X_train, y_train)

    print("Best Params:", best_params)

    joblib.dump(model, model_path)
    print("Model saved at:", model_path)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    train_pipeline()



