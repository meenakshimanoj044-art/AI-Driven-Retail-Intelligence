# Smart Product Insights Hub

An end-to-end retail analytics and machine-learning project that transforms online retail transaction data into product-demand forecasts, customer segments, churn indicators, and AI-generated business insights.

## Project Overview

The project processes 541,909 raw retail transactions and produces a cleaned dataset containing 357,993 valid purchase records. It combines retail data with country-aware public-holiday information and provides two main capabilities:

- Next-month product demand forecasting
- Customer segmentation and profile analysis
- Rule-based customer inactivity indicators
- Gemini-powered customer segment insights
- Interactive exploration through a Streamlit dashboard

The project demonstrates data preprocessing, feature engineering, time-series-aware model training, customer clustering, evaluation, and application development.
## How It Works

1. **Data preprocessing**
   - Removes records with missing customer IDs
   - Removes duplicate transactions
   - Separates purchases, invalid purchases, and returns
   - Filters invalid prices and handles quantity outliers
   - Creates revenue and time-based features

2. **Holiday integration**
   - Standardizes retail and holiday country names
   - Matches holidays using both transaction date and country
   - Creates holiday indicators and holiday-related sales features

3. **Feature engineering**
   - Creates monthly product-level sales features
   - Generates quantity and revenue lag features
   - Calculates rolling averages and growth rates
   - Builds product and customer KPI datasets

4. **Sales forecasting**
   - Predicts the following month’s product quantity
   - Uses a StandardScaler and Ridge regression pipeline
   - Uses time-series cross-validation for hyperparameter selection

5. **Customer analytics**
   - Groups customers using K-Means clustering
   - Uses order count, active months, average order value, recency, and purchase frequency
   - Labels customer groups as Dormant, Occasional, or Frequent
   - Generates concise business insights using the Gemini API

6. **Interactive dashboard**
   - Provides sales prediction and customer segment views through Streamlit
   ## Forecasting Evaluation

The forecasting data is divided chronologically, with records through August 2011 used for training and later records used for testing. The incomplete final month is excluded before target construction to prevent misleading evaluation.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge regression | 74.54 | 160.66 | 0.6971 |
| Previous-month baseline | 66.16 | 145.32 | 0.7522 |

The Ridge model explains approximately 69.7% of the variation in next-month product quantity. However, the simple previous-month baseline currently performs better. This comparison is retained transparently and identifies a clear opportunity for future model improvement using additional seasonal features or tree-based forecasting models.
## Technology Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Forecasting model:** StandardScaler and Ridge regression
- **Customer segmentation:** K-Means clustering
- **Model persistence:** Joblib
- **Generative AI:** Google Gemini API using `google-genai`
- **Dashboard:** Streamlit
- **Visualization:** Matplotlib and Seaborn
- **Configuration:** Python Dotenv
- **Version control:** Git and GitHub
## Project Structure

```text
AI-Driven-Retail-Intelligence/
├── app/
│   ├── app.py
│   └── train_kmean.py
├── data/
│   ├── raw/
│   ├── external/
│   ├── cleaned/
│   └── processed/
├── models/
│   └── sales_model.pkl
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   ├── llm_insights.py
│   └── explainability.py
├── .gitignore
├── README.md
└── requirements.txt

## Installation

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/meenakshimanoj044-art/AI-Driven-Retail-Intelligence.git
cd AI-Driven-Retail-Intelligence
```

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root and add your Gemini API key:

```text
GOOGLE_API_KEY=your_api_key_here
```

Never commit the `.env` file or expose the API key publicly.

## Running the Project

Run the data-processing pipeline:

```powershell
python -m src.preprocessing
python -m src.feature_engineering
```

Train and evaluate the forecasting model:

```powershell
python -m src.train
python -m src.predict
```

Start the Streamlit dashboard:

```powershell
python -m streamlit run app/app.py
```
## Current Limitations

- The forecasting model is trained on a historical dataset covering a limited time period.
- The Ridge model currently performs below the previous-month baseline.
- Customer churn is a rule-based inactivity indicator, not a supervised churn prediction model.
- AI-generated insights require an internet connection and a valid Gemini API key.
- Gemini requests can occasionally fail temporarily because of API demand or usage limits.

## Future Improvements

- Compare Ridge regression with Random Forest, LightGBM, and other forecasting approaches
- Add stronger seasonal, trend, and product-category features
- Evaluate customer clustering using silhouette score
- Build and evaluate a supervised churn model when labelled churn data is available
- Add automated tests and continuous integration
- Deploy the Streamlit application for public access
## Author

**Meenakshi Manoj**

- GitHub: [meenakshimanoj044-art](https://github.com/meenakshimanoj044-art)
- LinkedIn: [Meenakshi Manoj](https://www.linkedin.com/in/meenakshi-manoj-619929284/)