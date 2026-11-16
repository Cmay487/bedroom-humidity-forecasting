# preprocessing
from utils import DataCleanPipleLine

# viz
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plotting as plot

# time series eda
import pandas as pd
# ACF / PACF (numeric values and plots)
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose, STL

# Stationarity tests
from statsmodels.tsa.stattools import adfuller, kpss



def take_n_avg(df: pd.DataFrame, feature:str, sample: str | None = None) -> pd.DataFrame:

    # error handling for my sake!
    if sample is None:
        raise ValueError("Please select a sampling method")

    # create sampled data
    df = df.copy()
    sampled_df = df.resample(sample)[feature].mean()

    return sampled_df


def season_decomp(df:pd.DataFrame, resample_size:str):
    
    # take a copy
    df_copy = df[['Humidity','Temperature (oC)']].copy()

    # resample - mean
    df_copy = df_copy.resample(resample_size).mean()

    # run a season decomposition on humidity and temp
    model= 'additive'
    period = int(pd.Timedelta("1D") / pd.Timedelta(resample_size))

    for feature in df_copy.columns:
        # run seasonal decomp
        result = seasonal_decompose(df_copy[feature], model=model, period=period)

        #plot results
        fig = result.plot();
        fig.set_size_inches(14, 10)
        plt.show();
