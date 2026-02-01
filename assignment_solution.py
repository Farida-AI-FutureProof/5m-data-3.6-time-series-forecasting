#!/usr/bin/env python3
"""
Global Temperature Forecasting (NASA GISTEMP)

This script downloads NASA GISTEMP monthly temperature anomalies,
converts them into a proper monthly time series, and prints a sanity check.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# seaborn is optional; your script should still run without it
try:
    import seaborn as sns  # noqa: F401
except ModuleNotFoundError:
    sns = None  # type: ignore

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
)
from sktime.utils.plotting import plot_series


URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"


def prepare_nasa_data(url: str) -> pd.Series:
    """
    Load NASA GISTEMP CSV and return a monthly time series (MS frequency)
    of temperature anomalies as a pandas Series with a DatetimeIndex.
    """
    df = pd.read_csv(url, skiprows=1)

    # Keep Year + 12 months
    df = df.iloc[:, :13]
    df.columns = [
        "Year", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    # Wide -> Long
    df_long = df.melt(id_vars=["Year"], var_name="Month", value_name="Temperature_Anomaly")

    # Drop missing markers
    df_long = df_long[df_long["Temperature_Anomaly"] != "***"].copy()

    # Convert to float
    df_long["Temperature_Anomaly"] = df_long["Temperature_Anomaly"].astype(float)

    # Build Date (Month is like 'Jan' so %b is correct)
    df_long["Date"] = pd.to_datetime(
        df_long["Year"].astype(str) + "-" + df_long["Month"],
        format="%Y-%b",
    )

    df_long = df_long.sort_values("Date")

    ts = df_long.set_index("Date")["Temperature_Anomaly"]

    # Enforce monthly start frequency
    ts = ts.asfreq("MS")

    return ts


def main() -> None:
    ts = prepare_nasa_data(URL)

    print("Range:", ts.index.min(), "to", ts.index.max(), "n=", len(ts))
    print("Missing (monthly) values:", int(ts.isna().sum()))
    print("Head:\n", ts.head(), "\n")
    print("Tail:\n", ts.tail(), "\n")

    # Optional quick plot (comment out if not needed)
    # fig, ax = plt.subplots()
    # ts.plot(ax=ax, title="NASA GISTEMP Monthly Temperature Anomaly")
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Temperature Anomaly (°C)")
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
