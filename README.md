# FORESIGHT-Demand-Inventory-Intelligence
FORESIGHT-Demand-Inventory-Intelligence
# FORESIGHT — Demand & Inventory Intelligence

## Project
End-to-end demand forecasting and inventory risk analysis for the supplied synthetic retail dataset.

## What is included
- `src/` — data preparation, feature engineering, forecasting, rolling evaluation
- `dashboard/app.py` — Streamlit planning dashboard
- `data/processed/` — processed project tables (raw 10M transaction archive intentionally excluded)
- `outputs/` — EDA tables, charts, forecasts, backtest and risk scoring outputs
- `models/` — trained model artifact
- `reports/` — executive readout template

## Key result
The evaluated gradient-boosted candidate did **not** beat the seasonal-naive baseline. The recorded holdout metrics are:
- Seasonal-naive WAPE: **39.29%**
- Candidate model WAPE: **50.64%**
- Seasonal-naive MAE: **5.997**
- Candidate model MAE: **7.729**

Therefore the seasonal-naive forecast is used for the current production-style forecast output. No fabricated accuracy claim is made.

## Data mapping assumption
The supplied retail dataset did not match the Zidio four-table schema one-for-one. The transaction table was aggregated to daily SKU demand; SKU master was used as the product dimension; calendar fields were derived from the date range/promotional fields; and inventory was represented by the available SKU-level snapshot data. These assumptions are documented in the report.

## Run
From the repository root:
```bash
pip install -r requirements.txt
python src/model_project.py
streamlit run dashboard/app.py
```

The raw 10M transaction archive should not be committed to a public repository.
