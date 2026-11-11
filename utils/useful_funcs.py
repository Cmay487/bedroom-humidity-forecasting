import pandas as pd
def take_n_avg(df: pd.DataFrame, feature:str, sample: str | None = None) -> pd.DataFrame:

    # error handling for my sake!
    if sample is None:
        raise ValueError("Please select a sampling method")

    # create sampled data
    df = df.copy()
    sampled_df = df.resample(sample)[feature].mean()

    return sampled_df


