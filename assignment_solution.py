# %% [markdown]
# # Global Temperature Time Series Forecasting
# ## Assignment Solution
# 
# This script analyzes global temperature anomalies from 1880 to present.
# It performs data cleaning, EDA, decomposition, and forecasting using Naive, ARIMA, and ETS models.

# %%
# --- 1. SETUP & DATA LOADING ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Statsmodels for decomposition and ACF
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

# Sktime for forecasting
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sktime.utils.plotting import plot_series

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams["figure.figsize"] = (12, 6)

# Load Data (Using provided starter function)
url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

def prepare_nasa_data(url):
    df = pd.read_csv(url, skiprows=1)
    df = df.iloc[:, :13] 
    df.columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df_long = df.melt(id_vars=['Year'], var_name='Month', value_name='Temperature_Anomaly')
    df_long = df_long[df_long['Temperature_Anomaly'] != '***']
    df_long['Temperature_Anomaly'] = df_long['Temperature_Anomaly'].astype(float)
    df_long['Date'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + df_long['Month'], format='%Y-%b')
    df_long = df_long.sort_values('Date')
    
    # Set Index and Frequency (Critical for Time Series models)
    ts_data = df_long.set_index('Date')['Temperature_Anomaly']
    ts_data = ts_data.asfreq('MS') # 'MS' = Month Start
    
    return ts_data

ts_data = prepare_nasa_data(url)
print(f"Data Loaded: {ts_data.index.min()} to {ts_data.index.max()}")
print(f"Total Observations: {len(ts_data)}")

# %%
# --- 2. DATA EXPLORATION ---

# Plot 1: Overall Time Series
plt.figure(figsize=(12, 6))
plt.plot(ts_data, label='Temp Anomaly', color='tab:red', linewidth=1)
plt.title('Global Temperature Anomalies (1880 - Present)')
plt.ylabel('Deviation (°C)')
plt.xlabel('Year')
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.legend()
plt.show()

# Plot 2: Seasonal Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x=ts_data.index.month, y=ts_data.values, palette="coolwarm")
plt.title('Seasonal Plot: Monthly Distribution of Anomalies')
plt.xlabel('Month')
plt.ylabel('Anomaly')
plt.show()

# Plot 3: Autocorrelation (ACF)
plt.figure(figsize=(10, 5))
plot_acf(ts_data.dropna(), lags=48, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.show()

# %%
# --- 3. DECOMPOSITION ---
# We use 'additive' because anomalies can be negative. 
# Multiplicative models cannot handle zero or negative values easily.

decomposition = seasonal_decompose(ts_data.dropna(), model='additive', period=12)

fig = decomposition.plot()
fig.set_size_inches(12, 10)
fig.suptitle('Time Series Decomposition (Additive)', fontsize=16)
plt.tight_layout()
plt.show()

# %%
# --- 4. FORECASTING: TRAIN/TEST SPLIT ---
# Split the last 36 months (3 years) for testing
y_train, y_test = temporal_train_test_split(ts_data, test_size=36)

print(f"Training Data: {y_train.index.min()} to {y_train.index.max()}")
print(f"Testing Data:  {y_test.index.min()} to {y_test.index.max()}")

plot_series(y_train, y_test, labels=["Train", "Test"])
plt.title("Train/Test Split")
plt.show()

# %%
# --- 5. MODEL IMPLEMENTATION ---

# Model A: Seasonal Naive
# Predicts that next Jan = last Jan
naive_model = NaiveForecaster(strategy="seasonal_last", sp=12)
naive_model.fit(y_train)
fh = np.arange(1, len(y_test) + 1)
y_pred_naive = naive_model.predict(fh=fh)

# Model B: AutoARIMA
# Automatically finds p,d,q and P,D,Q parameters
print("Fitting AutoARIMA (this may take a moment)...")
arima_model = AutoARIMA(sp=12, suppress_warnings=True, seasonal=True)
arima_model.fit(y_train)
y_pred_arima = arima_model.predict(fh=fh)
print(f"Optimal ARIMA Parameters: {arima_model.get_fitted_params()}")

# Model C: AutoETS (Exponential Smoothing)
# Error, Trend, Seasonality
print("Fitting AutoETS...")
ets_model = AutoETS(auto=True, sp=12, n_jobs=-1)
ets_model.fit(y_train)
y_pred_ets = ets_model.predict(fh=fh)
print(f"Optimal ETS Method: {ets_model.get_fitted_params()['method']}")

# %%
# --- 6. EVALUATION ---

def get_metrics(y_true, y_pred, model_name):
    # MAE: Mean Absolute Error (Interpretability)
    mae = mean_absolute_error(y_true, y_pred)
    # sMAPE: Symmetric Mean Absolute Percentage Error (Good for stability)
    smape = mean_absolute_percentage_error(y_true, y_pred, symmetric=True)
    
    return {"Model": model_name, "MAE": mae, "sMAPE": smape}

results = [
    get_metrics(y_test, y_pred_naive, "Seasonal Naive"),
    get_metrics(y_test, y_pred_arima, "AutoARIMA"),
    get_metrics(y_test, y_pred_ets, "AutoETS")
]

results_df = pd.DataFrame(results)
print("\nModel Performance Metrics:")
print(results_df)

# Visual Comparison (Zooming in on last 10 years + Test)
plt.figure(figsize=(15, 6))
plot_series(y_train[-120:], y_test, y_pred_naive, y_pred_arima, y_pred_ets, 
            labels=["Train (History)", "Actual Test", "Naive", "ARIMA", "ETS"])
plt.title("Forecast Model Comparison (Zoomed)")
plt.show()

# %%
# --- 7. FINAL FUTURE FORECAST ---
# Retrain best model (Usually ETS or ARIMA) on FULL dataset
# Let's pick AutoETS for this example as it often handles trends well

best_model = AutoETS(auto=True, sp=12, n_jobs=-1)
best_model.fit(ts_data)

# Forecast next 5 years (60 months)
future_horizon = np.arange(1, 61)
future_forecast = best_model.predict(fh=future_horizon)

# Visualize Future

#Used Gemini to help with this"
plt.figure(figsize=(12, 6))
plot_series(ts_data[-180:], future_forecast, labels=["Historical Data (Last 15 Years)", "5-Year Forecast"])
plt.title("Global Temperature Forecast: Next 5 Years")
plt.show()
