"""Utility functions for exploratory data analysis and time-series preprocessing.

This module contains helpers for resampling sensor data, decomposing seasonal
patterns, and calculating a simple variance-based R-squared style summary.
"""

# preprocessing
from utils import DataCleanPipleLine

# viz
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plotting as plot
import numpy as np

# time series eda
import pandas as pd
# ACF / PACF (numeric values and plots)
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose, STL

# Stationarity tests
from statsmodels.tsa.stattools import adfuller, kpss



def take_n_avg(df: pd.DataFrame, feature:str, sample: str | None = None) -> pd.Series:
    """
    Resample a sensor feature to a chosen time interval and return the mean value.

    This is used to aggregate high-frequency readings into daily, hourly, or other
    time-based summaries so time-series analysis and plotting are easier to interpret.

    Args:
        df (pd.DataFrame): Source time-series data containing the selected feature.
        feature (str): Column name to aggregate.
        sample (str | None): Resampling frequency such as 'D' for daily or 'H' for hourly.
            Must be provided.

    Returns:
        pd.Series: The resampled series with the mean of the selected feature across
        each time bin.
    """

    # error handling for my sake!
    if sample is None:
        raise ValueError("Please select a sampling method")

    # create sampled data
    df = df.copy()
    sampled_df = df.resample(sample)[feature].mean()

    return sampled_df


def season_decomp(df:pd.DataFrame, resample_size:str) -> None:
    """
    Decompose each selected time-series feature into trend, seasonal, and residual
    components after resampling the data to a target frequency.

    The function is intended for exploratory analysis of humidity and temperature
    patterns, generating plots that help identify recurring seasonal behavior and
    residual variation after removing the trend and repeating seasonal signal.

    Args:
        df (pd.DataFrame): Input dataframe containing the 'Humidity' and
            'Temperature (oC)' columns indexed by time.
        resample_size (str): Resampling interval to apply before decomposition,
            such as 'H', 'D', or 'W'.

    Returns:
        None: This function creates diagnostic plots and prints R-squared-style
        summary values for each feature rather than returning data.
    """
    
    # resample - mean
    df_copy = df.copy()
    df_copy = df_copy.resample(resample_size).mean()

    # run a season decomposition on humidity and temp
    model= 'additive'
    period = int(pd.Timedelta("1D") / pd.Timedelta(resample_size))

    for feature in df_copy.columns:
        # run seasonal decomp
        result = seasonal_decompose(df_copy[feature], model=model, period=period)
        # return r2
        calculate_r2(df=df_copy[feature],result=result.resid, feature=feature)


        # show results
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)

        # show observed
        result.observed.plot(ax=axes[0], title=feature, lw=0.8)

        # rolling avg (trend)
        result.trend.plot(ax=axes[1], ylabel="Trend")

        # only show the first 3 cycles as these repeat
        result.seasonal.iloc[:period * 3].plot(ax=axes[2], ylabel="Seasonal (3 cycles)")
        axes[3].scatter(result.resid.index, result.resid.values, s=2, alpha=0.3)
        axes[3].axhline(0, color="black", lw=0.8)
        axes[3].set_ylabel("Resid")
        plt.tight_layout()

    return result

def calculate_r2(df:pd.DataFrame, result:object, feature:str) -> None:
    """
    Calculate a basic R-squared-style fit score for the residuals from a seasonal
    decomposition output.

    This helper compares the variance of the original data series to the variance of
    the residual component and prints the resulting score for each feature.

    Args:
        df (pd.DataFrame): Original time-series data used in the decomposition.
        result (object): Residual array or series produced by the decomposition.
        feature (str): Feature name being assessed.

    Returns:
        None: Prints the R2-style value for the selected feature to the console.
    """
    data_var = np.var(df)
    result_var = np.var(result)
    print(f"{feature} R2: {1 - result_var/data_var:.2f}")

def model_vs_reality(df:pd.DataFrame, model:object) -> None:
    """
    
    """

    