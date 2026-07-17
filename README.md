# AI-Powered Supermarket Sales Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn)

An interactive Streamlit dashboard for exploring supermarket sales performance, customer behavior, product trends, branch performance, financial metrics, and sales forecasting from a single dataset.

## Overview

This project turns transactional supermarket data into a multi-page business intelligence app. It combines filtering, KPI tracking, Plotly visualizations, export tools, and lightweight machine learning so you can move from raw sales records to operational insights quickly.

The repository ships with a sample dataset at `data/supermarket_sales.csv`, and each analytics page also supports CSV or Excel uploads directly from the Streamlit sidebar.

## Key Features

- Executive dashboard with KPI cards, dataset health checks, previews, and business insights
- Sales analytics with daily, weekly, monthly, rolling-average, and forecast-style trend views
- Branch analytics for revenue, orders, ratings, gross income, and branch comparison
- Product analytics for assortment performance, brand mix, and category-level trends
- Customer analytics with segmentation and clustering workflows
- Finance analytics focused on revenue, tax, gross income, and profit margin behavior
- Ratings and satisfaction analysis
- Reporting and export support for CSV, Excel, and PDF outputs
- Prediction page with `Linear Regression` and `Random Forest` models for what-if revenue forecasting
- Sidebar filters and per-page dataset upload support across the app

## App Pages

- `app.py`: Main landing dashboard with dataset preview, validation, missing-value reporting, insights, and exports
- `pages/dashboard.py`: Rich executive BI experience with charts, health metrics, recent transactions, and summary cards
- `pages/sales.py`: Sales trend analysis across dates, weekdays, hours, rolling averages, and simple forecasting views
- `pages/branches.py`: Branch-level revenue, order volume, rating, customer, and gross-income comparisons
- `pages/products.py`: Product and brand performance analysis
- `pages/customers.py`: Customer behavior and segmentation, including clustering support
- `pages/finance.py`: Financial performance, tax analysis, and profitability monitoring
- `pages/ratings.py`: Customer rating and satisfaction insights
- `pages/reports.py`: Export-oriented reporting workflows
- `pages/prediction.py`: ML-based sales prediction and scenario simulation

## Project Structure

```text
Supermarket-Sales-Dashboard/
|-- app.py
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- assets/
|   `-- styles.css
|-- data/
|   `-- supermarket_sales.csv
|-- pages/
|   |-- branches.py
|   |-- customers.py
|   |-- dashboard.py
|   |-- finance.py
|   |-- prediction.py
|   |-- products.py
|   |-- ratings.py
|   |-- reports.py
|   `-- sales.py
`-- utils/
    |-- analytics.py
    |-- charts.py
    |-- config.py
    |-- data_loader.py
    |-- exports.py
    |-- helpers.py
    |-- insights.py
    |-- ml_models.py
    `-- preprocessing.py
```

## Dataset

The bundled sample dataset includes fields such as:

- `Invoice ID`
- `Customer ID`
- `Customer Name`
- `Branch`
- `City`
- `State`
- `Customer Type`
- `Gender`
- `Brand`
- `Product`
- `Product Line`
- `Unit Price`
- `Quantity`
- `Discount %`
- `Discount Amount`
- `Tax`
- `Total`
- `Date`
- `Time`
- `Payment`
- `COGS`
- `Gross Margin %`
- `Gross Income`
- `Rating`
- `Salesperson`
- `Stock Available`

The app is tolerant of some schema differences and derives several helper fields during preprocessing, but pages work best when the uploaded dataset preserves the core transactional columns above.

## Installation

```bash
git clone https://github.com/akash098p/AI-Supermarket-Sales-Analytics.git
cd AI-Supermarket-Sales-Analytics
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib
- Scikit-learn
- SciPy
- OpenPyXL
- ReportLab
- Joblib
- `streamlit-option-menu`
- `streamlit-extras`

## Machine Learning

The prediction workflow currently supports:

- `Linear Regression`
- `RandomForestRegressor`
- Feature preprocessing via utilities in `utils/ml_models.py`
- Evaluation metrics including `R2`, `MAE`, and `RMSE`
- Single-row what-if scenario prediction from user-selected inputs

## Exports

Multiple pages support exporting the current filtered view as:

- CSV
- Excel
- PDF

## Running With Your Own Data

1. Start the Streamlit app.
2. Open the page you want to analyze.
3. Use the sidebar uploader to load a `.csv`, `.xlsx`, or `.xls` file.
4. Apply filters to narrow the view by branch, city, product, customer type, payment mode, date range, rating, or quantity depending on the page.

## License

This project is released under the [MIT License](LICENSE).

## Author

Akash Pramanik
