# Time Series Analysis and Forecasting of Daily Website Visitors (Level 2)

An intermediate time-series project that takes a real website-traffic dataset from a raw CSV file to a 13-model comparison, rolling-origin validation, a significance test and a 28-day forecast with uncertainty. Every step is explained in a *Concept → Why → Code → Interpretation* format.

This is **Level 2** of a series. [Level 1](https://github.com/mdfokrulakon/Delhi_Temperature_Time_Series_Forecasting) forecasted daily temperature (one yearly seasonal cycle). Level 2 adds:

| Topic | Level 1 | Level 2 |
|---|---|---|
| Seasonality | yearly | **weekly + yearly** |
| Seasonal ARIMA | not practical (period 365) | **true SARIMA with period 7** |
| Differencing | regular only | **regular and seasonal**, chosen by stated rules |
| Evaluation | one hold-out | **one-shot + rolling-origin (walk-forward)** |
| Comparing models | table of errors | **Diebold-Mariano significance test** |
| Variance | additive only | **log-scale model** |
| Intervals | model-based only | **model-based or empirical, plus coverage check** |
| Data cleaning | clean numeric CSV | **numbers stored as text, renamed columns** |

## Dataset

- **Name:** Daily website visitors (time series regression)
- **Source:** https://www.kaggle.com/datasets/bobnau/daily-website-visitors
- **What it is:** daily traffic of a statistical-forecasting teaching-notes website. According to the dataset page: 2,167 rows from 2014-09-14 to 2020-08-19, with page loads, unique visitors, first-time visitors and returning visitors.
- **Target:** unique visitors per day (`unique_visits`)
- By the author's definition, unique visitors = first-time + returning visitors. The notebook uses this as a data-quality check.

## Objectives

- Clean a real-world CSV and build a proper datetime index
- Find weekly and yearly seasonality and a trend
- Test stationarity and choose regular and seasonal differencing by documented rules
- Compare baseline, classical, transformed and machine-learning models
- Validate with a hold-out **and** with rolling-origin forecasts
- Test whether differences between models are statistically meaningful
- Diagnose residuals and interval calibration, then forecast with uncertainty

## Technologies Used

Python, pandas, NumPy, SciPy, Matplotlib, Seaborn, statsmodels, scikit-learn, Jupyter

## Methodology

1. Load and inspect the raw file
2. Clean: standardise column names, convert text counts to numbers, parse dates, check duplicates, gaps, negative and zero counts, and the unique = first-time + returning identity
3. **Chronological split first:** the last 90 days are the test set, and all exploration uses the training data only
4. Explore: rolling statistics, distribution, visitor types, weekday/month/year patterns, resampling
5. Decompose weekly (period 7) and yearly (period 52 on weekly data)
6. Stationarity: ADF test, first and seasonal differencing
7. ACF/PACF before and after differencing
8. Fit 13 models; ARIMA/SARIMA orders chosen by a small AIC search on training data only
9. Evaluate: one-shot 90-day forecast and rolling-origin 14-day forecasts with refitting
10. Diebold-Mariano test of the selected model against every competitor
11. Residual analysis and interval coverage
12. Refit on all data and forecast the next 28 days

## Models Used (13)

| Model | Family |
|---|---|
| Naive | baseline |
| Seasonal Naive (7 days) | baseline |
| Seasonal Mean (8 weeks, same weekday) | baseline |
| Moving Average (7 days) | smoothing |
| ARIMA | classical, non-seasonal |
| SARIMA (weekly) | classical, seasonal |
| SARIMA (log scale) | transformed |
| SARIMAX + weekday dummies + yearly Fourier terms | regression with ARIMA errors |
| Holt-Winters | exponential smoothing |
| STL + ARIMA | decomposition |
| Theta | classical |
| Random Forest (lags + calendar, recursive) | machine learning |
| Gradient Boosting (lags + calendar, recursive) | machine learning |

The model-selection rule is stated before the results are seen: the final model is the one with the lowest **rolling-origin RMSE**.

## Evaluation Metrics

MAE, RMSE, MAPE and bias, on both the one-shot and the rolling-origin evaluation, plus Diebold-Mariano p-values and 95% interval coverage.

## Key Findings

Results from running the notebook on the real data. Training data: 2,077 days (2014-09-14 to 2020-05-21). Test data: the last 90 days (2020-05-22 to 2020-08-19).

**Weekly pattern (the dominant structure)**

- **Tuesday is the busiest weekday** (about 3,555 unique visitors on average) and **Saturday the quietest** (about 1,775).
- Weekends are about **37.8% lower** than weekdays.
- Weekly seasonal strength is **0.90**, so most of the short-term variation is a repeating 7-day pattern.

**Yearly pattern and trend**

- Yearly seasonal strength (measured on weekly averages) is **0.89**. On average **April is the busiest month and July the quietest**.
- Average daily visitors per complete year: 2015: 2,908 | 2016: 3,200 | 2017: 2,462 | 2018: 3,055 | 2019: 3,056. These move up and down; they do not show a steady rise.
- The decomposition's trend strength is high (**0.92**), but that trend is a 7-day moving average, so it includes the slow yearly movements. It reflects a changing level, not a one-way trend.

**Stationarity and differencing**

- ADF p-values: original **0.0003**, first difference 0.0000, seasonal difference 0.0000, both 0.0000.
- Chosen by the stated rules: ARIMA `d = 0`; SARIMA `d = 0, D = 1` (weekly seasonal strength 0.90 is above the 0.64 threshold).
- Selected SARIMA: **(2, 0, 1) x (1, 1, 1, 7)**.

**Model performance, sorted by rolling-origin RMSE (14-day forecasts, refitted at each origin)**

| Model | Rolling RMSE | Rolling MAE | Rolling MAPE (%) | One-shot 90-day RMSE |
|---|---:|---:|---:|---:|
| **Random Forest (lags)** | **192.89** | 164.78 | 5.92 | 257.03 |
| STL + ARIMA | 213.73 | 162.66 | 5.86 | 518.88 |
| Gradient Boosting (lags) | 215.02 | 176.32 | 6.27 | 385.08 |
| SARIMA (log scale) | 256.21 | 200.99 | 7.07 | 490.76 |
| SARIMA (weekly) | 266.51 | 215.95 | 7.81 | 453.61 |
| Seasonal Naive (7d) | 279.54 | 221.60 | 7.95 | 1049.10 |
| Theta | 332.48 | 259.78 | 9.68 | 908.53 |
| Holt-Winters | 337.17 | 262.01 | 9.84 | 899.69 |
| SARIMAX + dummies + Fourier | 406.91 | 322.69 | 12.27 | 319.44 |
| Moving Average (7d) | 490.42 | 378.34 | 15.28 | 1122.93 |
| ARIMA | 572.65 | 453.32 | 17.26 | 612.16 |
| Naive | 639.73 | 476.74 | 19.36 | 1290.59 |
| Seasonal Mean (8 weeks) | 656.70 | 609.79 | 21.85 | 1188.17 |

- **Selected model (lowest rolling-origin RMSE): Random Forest with lag and calendar features.** Its rolling MAE of about 165 visitors is roughly 6% of a typical day, and its mean rolling bias is only about -10 visitors.
- **Significance (Diebold-Mariano test):** Random Forest was **not** significantly different from **SARIMA (log scale)**, **Gradient Boosting** and **STL + ARIMA** (p >= 0.05). It was significantly better than the other nine models (p < 0.05), keeping in mind that several tests were run against one reference model.
- **The metrics do not fully agree.** STL + ARIMA has a slightly lower rolling MAE and MAPE than Random Forest, while Random Forest has the lowest RMSE. Given the tie in the Diebold-Mariano test, the choice of Random Forest follows the stated rule rather than proving it is clearly the best.
- **Short and long horizons rank models differently.** STL + ARIMA is second at 14 days but sixth over 90 days. SARIMAX + dummies + Fourier is second over 90 days (RMSE 319) but ninth at 14 days. Models that adapt quickly to the recent level do well on short horizons, while the model with explicit yearly terms does better on the long one.
- **Most models over-forecast the 90-day test window.** Naive, Seasonal Naive, Seasonal Mean and Moving Average all have large negative bias (about -950 to -1,140 visitors), so traffic was lower in the test window than in the weeks before it. The yearly pattern shows July as the quietest month, which is a likely contributor, but this project cannot separate a seasonal decline from other causes (for example the unusual conditions of 2020).
- **Baselines matter.** A plain seasonal naive forecast (same weekday last week) had a rolling RMSE of 279.54, better than ARIMA, Holt-Winters, Theta and SARIMAX in the short-horizon test. Models without any weekly component (Naive, Moving Average, ARIMA) were clearly the weakest.

**Forecast**

- The selected model is refitted on all data and forecasts the **next 28 days**. Random Forest has no model-based prediction interval, so the notebook uses an **empirical 95% interval** built from the model's rolling-origin errors. This assumes errors do not grow with the horizon, which is an approximation.
- Errors grow with the horizon, so expect the 28-day forecast to be less accurate than the 14-day scores above.

## Figures

After running the notebook, upload the PNG files to `figures/`. Suggested ones:

![Model grid](figures/16_model_grid.png)
![Top models](figures/17_top_models.png)
![Future forecast](figures/19_future_forecast.png)

## Limitations

- One website and one series, so results may not transfer to other sites
- The 90-day test window covers one part of the year (late May to mid-August 2020). Most models over-forecast it, and the dataset does not document why traffic was lower
- Model orders are fixed during rolling-origin refits (only parameters are re-estimated)
- Interval approximations; empirical intervals assume errors do not grow with the horizon
- The selected Random Forest is untuned, has no native prediction intervals, and recursive forecasting accumulates errors over long horizons
- Several Diebold-Mariano tests are run against one reference model, so p-values are guidance, not proof
- "Unique visit" depends on IP addresses and cookies

## Repository Structure

```text
Daily_Website_Visitors_Time_Series_Forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── website_visitors_forecasting.ipynb
├── data/
│   └── README.md
├── src/
│   └── README.md
└── figures/
    └── README.md
```

## How to Run

```bash
git clone https://github.com/<your-username>/Daily_Website_Visitors_Time_Series_Forecasting.git
cd Daily_Website_Visitors_Time_Series_Forecasting
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# download the CSV into data/ (see data/README.md)
jupyter notebook notebooks/website_visitors_forecasting.ipynb
```

**Google Colab:** upload the CSV to `/content/` and run the notebook. It looks there first. The full run takes several minutes because every model is refitted at each rolling origin.

## Dataset / License Note

The dataset belongs to its original author and is not redistributed here. Check the licence and terms on the Kaggle page before any reuse.

## Future Improvements

- Tune the machine-learning models with time-series cross-validation
- Re-select model orders at each rolling origin
- Add holiday and academic-calendar features
- Try a multiplicative seasonal model and Box-Cox transformations
- Compare with Prophet and other libraries
- Move helper functions into `src/` and add tests
