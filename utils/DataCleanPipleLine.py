"""Data cleaning and preprocessing utilities for sensor data.

This module loads raw sensor readings, standardises dates, removes duplicate and
outlier values, fills missing records, and adds time-based feature columns used in
subsequent forecasting and analysis workflows.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# in the dataload, check if a batch is in the dataset but not in the cleaned dataset. extract what is not in the cleaned dataset and 
# pass through the pipeline.

class DataImportAndCleaningPipeline:
    def __init__(self,file_location : str):
        """
        Initialise the pipeline with the location of the source CSV file.

        Args:
            file_location (str): Path to the raw sensor dataset.

        Returns:
            None: Stores the file path on the instance and creates an empty dataframe
            placeholder.
        """
        self.df = None
        self.file_location = file_location
        
    # load dataset & clean
    def DataLoad(self):
        """
        Load the CSV file into memory and perform initial cleaning of column names and
        the leading rows.

        Args:
            None.

        Returns:
            None: Updates the instance's ``df`` attribute with the loaded and trimmed
            dataset.
        """
        # load the data into the pipeline
        self.df = pd.read_csv(self.file_location)

        # remove deadspace in the column names
        self.df.columns = self.df.columns.str.strip()

        # drop the first 10 rows
        self.df = self.df.iloc[9:,:].copy()

        print("*"*100)
        print(f"DF Shape: {self.df.shape}")
        print("*"*100)
        print("*"*100)
        print(f"{self.df.tail(10)}")
        print("*"*100)

    
    def RemoveDuplicates(self):
        """
        Remove duplicate readings based on the DateTime field.

        Args:
            None.

        Returns:
            None: Drops repeated rows from ``self.df`` in place.
        """
        # ====================
        # Duplicates
        # ====================
        duplicates = self.df['DateTime'].duplicated().sum()

        # as we know the duplicates are likely when the sensor comes on, remove them
        if duplicates > 0:
            self.df.drop_duplicates(subset=['DateTime'], keep='first', inplace=True)
            print("*"*30)
            print(f"{duplicates} duplicates were removed")
            print("*"*30)
    
    def RemoveOutliers(self):
        """
        Replace numeric outliers with the previous valid reading using a 3-sigma rule.

        Args:
            None.

        Returns:
            None: Modifies ``self.df`` by imputing outlier values and removing the
            temporary standardised columns used during detection.
        """
        # ====================
            # Outliers
        # ====================

        # 1. standardise the targeteted columns and remove outliers z > 3
        # as we know temp, humidity and pressure do not spike out of nowhere, the imputation will be focussed there
        for col in self.df.select_dtypes(include=['float64','int64']):
            #if col not in ['eCo2','VOC']:
            # take row over row change and fill na with 0 = first row
            self.df[f"{col}_Diff"] = self.df[col].diff().fillna(0)
            std_scalar = StandardScaler()
            self.df[f"{col}_Diff_scaled"] = (
            std_scalar.fit_transform(self.df[col].values.reshape(-1,1)) 
                    if col in ['Pressure','eCo2','VOC']
                    else std_scalar.fit_transform(self.df[f"{col}_Diff"].values.reshape(-1,1))
                    )

            # I shall follow the empiral rule:
            # 68% of data falls within 1 std
            # 95% of data falls within 2 std
            # 99.6% of data falls within 3 std
            # anything above 2 std is an outlier, remove 5% of data that fall within 3 std = 5 data points for every 100 sample

            # locate the data that is outside 3 stds and replace with the previous value reading
            outlier_mask = np.abs(self.df[f"{col}_Diff_scaled"]) > 3

            # filter the data and replace with Nan then ffil
            self.df.loc[outlier_mask,col] = np.nan
            nan_count = self.df[col].isna().sum()
            self.df[col] = self.df[col].ffill() 
            print("*"*100)
            print(f"{nan_count} Outliers were imputed in {col} feature")
            print("*"*100)
                

            # drop the scaled columns
            self.df.drop(
                columns=[f"{col}_Diff",f"{col}_Diff_scaled"],
                inplace=True
            )
            
    
    def CleanDateData(self):
        """
        Standardise the DateTime column to a consistent datetime format and set it as the index.

        Args:
            None.

        Returns:
            None: Converts the raw date strings into timezone-aware pandas timestamps
            and stores them in the dataframe index.
        """
    
        # ========================
        # Date Type Cleaning
        # ========================

        # remove any dead space.
        self.df['DateTime'] = self.df['DateTime'].astype(str).str.strip()

        # we have dates where they are 18:00 not 18:00:00 and pandas chucks an error.
        # in the loop we want to add :00 where the length of the string is less than 19
        self.df['DateTime'] = self.df['DateTime'].apply(
            lambda x: f"{x}:00" if len(str(x)) <= 16 else x
        )

        # we have some dates like 01/04/26 and others like 04-01-26 causing pandas to fail.
        # clean this date data to ensure one style is present -> 2026-04-01

        self.df['DateTime'] = self.df['DateTime'].apply(
            lambda x: f"{x[6:10]}-{x[3:5]}-{x[:2]}{x[10:]}" if "/" in str(x) else x
        ).astype(str).str.strip()

        # convert to datetime and floor the seconds to 00
        self.df['DateTime'] = pd.to_datetime(
            self.df['DateTime'],
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce', 
            utc=True
        ).dt.floor('min')

        # set datetime index
        self.df.set_index('DateTime', inplace=True)

        # keep the datetime feature for future use
        self.df['DateTime'] = self.df.index


    def AddTimeSeriesFeatures(self):
        """
        Create time-based features for forecasting and exploratory analysis.

        Args:
            None.

        Returns:
            None: Adds date, month, week, time, and daily average feature columns to
            ``self.df``.
        """
        # ========================
        # Add Time Series Features
        # ========================
        self.df['Date'] = pd.to_datetime(self.df.index.date, dayfirst=True)

        # add year month
        self.df['YearMonth'] = self.df['Date'].dt.to_period("M")

        # add end of week & convert to uk date format
        self.df['EndOfWeek'] = (
            self.df['Date'] - pd.to_timedelta(self.df['Date'].dt.weekday, unit='D') + pd.to_timedelta(6, unit='D')
            )

        # get the time value
        self.df['Time'] = self.df['Date'].dt.time
        self.df['Time_Numeric'] = pd.to_timedelta(self.df['Date'].dt.strftime('%H:%M:%S')).dt.floor('min')

        daily = self.df["Date"].dt.floor("D")

        # create a daily average column
        self.df["DailyAvgTemperature"] = (
            self.df.groupby(daily)["Temperature (oC)"].transform("mean")
        )
        self.df["DailyAvgHumidity"] = (
            self.df.groupby(daily)["Humidity"].transform("mean")
        )

    def ImputeMissingData(self):
        """
        Reindex the dataset to a one-minute timeline and linearly interpolate missing
        numeric values.

        Args:
            None.

        Returns:
            None: Replaces missing sensor readings in ``self.df`` with interpolated
            values and stores the completed dataframe back on the instance.
        """

        # ========================
        # Impute NaN with Interpolation
        # ========================

        # take min and max dates
        min_dt = self.df.index.min()
        max_dt = self.df.index.max()

        # create full date df
        full_dt_df = pd.date_range(
                start=min_dt,
                end=max_dt,
                freq="1min"
            )

        # join the raw data to full_dt_df
        merged_df = self.df.reindex(full_dt_df)

        # count the NaN filled and print report
        self.count_of_missing_data(df=merged_df)

        # create feature that flags when the sensor is off and on
        merged_df['Sensor_Status'] = np.where(merged_df['Humidity'].isna(), 'Off', 'On')

        # interpolate only numeric columns
        num_cols = merged_df.select_dtypes("number").columns
        merged_df[num_cols] = merged_df[num_cols].interpolate(method="pchip")

        # create self.df as merged_df
        self.df = merged_df

    def count_of_missing_data(self, df:pd.DataFrame, col:str="Humidity") -> None:
        """
        Print a summary of how much missing data was present in a target field.

        Args:
            df (pd.DataFrame): Dataframe to inspect for missing values.
            col (str): Column name to use when counting missing entries. Defaults to
                ``"Humidity"``.

        Returns:
            None: Prints the missing count and percentage to the console.
        """
        
        samples_size = len(df)
        missing_count = df[col].isna().sum()

        print("*"*100)
        print(f"{missing_count} missing data points was imputed\n")
        print(f"{missing_count/samples_size*100:.2f}% was imputed")
        print("*"*100)



                          
    def RunPipeline(self, impute_missing_data:bool=True) -> pd.DataFrame:
        """
        Run the full cleaning pipeline in the correct order.

        Args:
            impute_missing_data (bool): If ``True``, interpolate missing readings after
                removing duplicates and outliers. Defaults to ``True``.

        Returns:
            pd.DataFrame: The cleaned dataframe with repaired timestamps, imputed
            values, and engineered time features.
        """
        self.DataLoad()
        self.CleanDateData()
        self.RemoveDuplicates()
        self.RemoveOutliers()

        if impute_missing_data:
            self.ImputeMissingData()
            
        self.AddTimeSeriesFeatures()

        return self.df



