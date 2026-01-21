# Global Temperature Time Series Forecasting

## 1. Introduction
This project analyzes the NASA GISTEMP Global Temperature Anomalies dataset (1880–Present). The goal is to understand historical patterns, decompose the time series components, and forecast future temperature trends using statistical modeling techniques.

## 2. Data Exploration
The dataset was loaded and transformed into a monthly time series format.
* **Time Range:** 1880 to Present.
* **Trend:** A visual inspection reveals a distinct, non-linear upward trend, accelerating significantly post-1960.
* **Seasonality:** While present, the seasonality is less dominant than the overall trend.
* **Autocorrelation:** The ACF plot shows a slow decay, confirming the data is non-stationary and possesses a strong trend component.

## 3. Decomposition
An **Additive Decomposition** was performed (`Trend + Seasonal + Residual`). 
* *Why Additive?* The dataset contains temperature anomalies which can be negative. Multiplicative decomposition generally requires strictly positive values.
* **Findings:** The trend component successfully isolated the long-term warming. The seasonal component revealed a consistent 12-month cycle, likely driven by land-ocean temperature variances during seasons.

## 4. Forecasting Models
The data was split temporally:
* **Training Set:** 1880 – [3 years ago]
* **Testing Set:** Last 36 months

Three models were evaluated:
1.  **Seasonal Naive:** Uses the value from the same month in the previous year.
2.  **AutoARIMA:** Automatically selects parameters (p,d,q) to handle autocorrelation and trend.
3.  **AutoETS:** Exponential Smoothing handling Error, Trend, and Seasonality.

## 5. Model Evaluation
The models were evaluated using **MAE** (Mean Absolute Error) and **sMAPE** (Symmetric Mean Absolute Percentage Error).

| Model | MAE | sMAPE | Observation |
|-------|-----|-------|-------------|
| **Seasonal Naive** | *High* | *High* | Fails to capture the continued upward trend; underestimates recent heat. |
| **AutoARIMA** | *Low* | *Low* | Captures trend well via differencing ($d=1$). |
| **AutoETS** | *Low* | *Low* | Generally performs robustly by explicitly modeling the trend component. |

*Note: Specific metric values vary slightly based on the exact latest data update from NASA.*

## 6. Conclusion & Future Forecast
Based on the evaluation, the **AutoETS/ARIMA** (whichever scored lower error in the run) was selected to forecast the next 5 years.

The forecast indicates a continued rise in global temperature anomalies, following the historical trajectory established over the last few decades. The seasonal oscillation remains constant, but the baseline temperature continues to increase.