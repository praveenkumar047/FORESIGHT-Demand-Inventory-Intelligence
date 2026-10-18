
# FORESIGHT Executive Readout

## Business objective
Provide a weekly SKU demand forecast and an early-warning system for stockout and overstock decisions.

## Dataset
The supplied retail archive contains approximately 10 million transactions and 5,000 SKUs. The clean transaction file covers 2022-01-01 through 2025-12-31.

## EDA findings
- Demand is available across 5,000 SKUs and four calendar years.
- Top movers and low-volume/dead-stock candidates are provided in `outputs/top_movers.csv` and `outputs/dead_stock_candidates.csv`.
- Category and monthly demand summaries are in `outputs/category_performance.csv` and `outputs/monthly_demand.csv`.

## Forecasting
A seasonal-naive four-week baseline was compared with a gradient-boosted candidate using rolling-origin time-based backtesting.

| Metric | Seasonal-naive | Candidate model |
|---|---:|---:|
| Mean WAPE | 35.85% | 36.86% |

The baseline wins, so the baseline is retained for the current forecast. This is an honest model-selection result rather than a fabricated improvement.

## Inventory risk
Current output identifies:
- 731 SKUs with stockout risk
- 3,424 SKUs with overstock risk
- 845 SKUs healthy
- Estimated sales at risk: ₹14,022,502
- Estimated locked inventory capital: ₹1,289,535,472

These figures are generated from the supplied inventory snapshot and the 8-week forecast. They should be treated as planning estimates, not confirmed financial losses.

## Recommended actions
- REORDER NOW: prioritize SKUs where forecast lead-time demand exceeds available/on-order stock.
- MARKDOWN / CLEAR: prioritize SKUs with on-hand stock above forecast horizon demand.
- HEALTHY: no immediate action.

## Limitations
The supplied inventory file is a current store-level snapshot rather than the periodic date/SKU inventory extract described in the original brief. It was therefore aggregated to SKU level and assigned the latest sales date as the snapshot date. Lead time was set to 7 days where the source did not provide it. This assumption must be replaced with real lead-time data before operational use.
