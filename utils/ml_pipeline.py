import pandas as pd
from sklearn.model_selection import train_test_split

def train_val_test_split(df:pd.DataFranme) -> dict:
    """
    
    """
    df_copy = df.copy()

    train_size = 0.8
    test_size = 1 - train_size

    x_train, x_test = train_test_split(
        df_copy,
        test_size=test_size,
        train_size=train_size,
        random_state=42,
        shuffle=False
        )

    x_train, x_val = train_test_split(
        x_train,
        test_size=test_size,
        train_size=train_size,
        random_state=42,
        shuffle=False
        )

    # extract target
    y_train, y_val, y_test = (
        x_train['Humidity'], 
        x_val['Humidity'], 
        x_test['Humidity']
    )

    # drop target
    cols_to_drop = ['Humidity', 'DailyAvgHumidity', 'DailyAvgTemperature']

    # save a df for eda purposes
    train_df = x_train.copy()

    x_train, x_val, x_test = (
        x_train.drop(columns=cols_to_drop),
        x_val.drop(columns=cols_to_drop),
        x_test.drop(columns=cols_to_drop)
        )

    out_dict = {
        "x_train":x_train,
        "x_val":x_val,
        "x_test":x_test,
        "y_train":y_train,
        "y_val":y_val,
        "y_test":y_test,
        "train_df":train_df
    }

    return out_dict




    

