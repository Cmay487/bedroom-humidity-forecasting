import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# in the dataload, check if a batch is in the dataset but not in the cleaned dataset. extract what is not in the cleaned dataset and 
# pass through the pipeline.

class DataImportAndCleaningPipeline:
    def __init__(self,file_location : str):
        self.df = None
        self.file_location = file_location
        
    # load dataset & clean
    def DataLoad(self):
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
        """Drop duplicate rows based on the 'DateTime' column."""
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
    # ========================
    # Add Time Series Features
    # ========================
        self.df['Date'] = pd.to_datetime(self.df['DateTime'].dt.date, dayfirst=True)
        self.df['UkDate'] = self.df['Date'].dt.strftime('%d/%m/%Y')


        # add year month
        self.df['YearMonth'] = self.df['DateTime'].dt.to_period("M")

        # add end of week & convert to uk date format
        self.df['EndOfWeek'] = self.df['DateTime'] - pd.to_timedelta(self.df['DateTime'].dt.weekday, unit='D') + pd.to_timedelta(6, unit='D')

        # get the time value
        self.df['Time'] = self.df['DateTime'].dt.time
        self.df['Time_Numeric'] = pd.to_timedelta(self.df['DateTime'].dt.strftime('%H:%M:%S')).dt.floor('min')

        daily = self.df["DateTime"].dt.floor("D")
        # create a daily average column
        self.df["DailyAvgTemperature"] = (
            self.df.groupby(daily)["Temperature (oC)"].transform("mean")
        )
        self.df["DailyAvgHumidity"] = (
            self.df.groupby(daily)["Humidity"].transform("mean")
)

    def ImputeMissingData(self):
    # ========================
    # Impte NaN with TS LR
    # ========================
        # Step 1: Loop through the data and identify where the NaN values are.
        # Step 2: Normalise the data
        # Step 3: build and train a TS Linear Regression model on a sample of the last 500  minutes
        # Step 3: write to cleaned data csv file
        print("Starting Missing Data Impute Process...") 

        all_time = pd.DataFrame(
            pd.date_range(
                start=self.df['DateTime'].min(),
                end=self.df['DateTime'].max(),
                freq="1min"),
            columns=['DateTimeFullList']
            )

        #merge the dfs to have a full range of dates
        FullDatesDf = pd.merge(
            left=all_time,
            right=self.df,
            how='left',
            left_on='DateTimeFullList',
            right_on='DateTime'
        )

        FullDatesDf['Time_'] = pd.to_timedelta(FullDatesDf['DateTimeFullList'].dt.strftime('%H:%M:%S')).dt.floor('min')
        FullDatesDf['Temperature_lag1Min'] = FullDatesDf['Temperature (oC)'].shift(1).fillna(18)
        FullDatesDf['Humidity_lag1Min'] = FullDatesDf['Humidity'].shift(1).fillna(0.5)
        FullDatesDf['Time_Float'] = FullDatesDf['Time_'].values.astype(float) / 10**9


        FullDatesDf['SensorStatus'] = 'On'
        for i in range(1, len(FullDatesDf) - 1 ):

            #loop thorugh the dataset until you find a NaN and use the preceding data to train the model
            target_list = ['Temperature (oC)','Humidity']
            for target in target_list:
                target_val = FullDatesDf.loc[i, target]
                next_val = FullDatesDf.loc[i+1, target]

                # check for nans in two steps current and next
                if pd.isna(target_val) and pd.isna(next_val):
                    print("Starting Data Impute Process")
                    train_df = (
                        FullDatesDf
                        .iloc[:i-1]
                        .dropna(subset=['Temperature (oC)', 'Temperature_lag1Min','Time_Float'])
                        .copy()
                    )

                    # update status feature
                    FullDatesDf['SensorStatus', i] = 'Off'


                    x = train_df[['Temperature_lag1Min','Time_Float']].values
                    y = train_df['Temperature (oC)'].values.reshape(-1, 1)


                    # remove small samples
                    if len(x) < 50:
                        continue

                    # apply standardisation
                    x_sc = StandardScaler()
                    y_sc = StandardScaler()

                    x_scaled = x_sc.fit_transform(x)
                    y_scaled = y_sc.fit_transform(y)


                    # build the model
                    inverse_matrix =  np.linalg.inv(x_scaled.T @ x_scaled) 
                    weights = inverse_matrix @ x_scaled.T @ y_scaled

                    #make predictions
                    x_test_sliced =  x_scaled[-1].reshape(1, -1)

                    # make the predictions
                    predictions = x_test_sliced @ weights
                    
                    pred = y_sc.inverse_transform(predictions)

                    # fill in the blanks
                    FullDatesDf.loc[i, 'Temperature (oC)'] = pred[0, 0]

                    # update lag 1 feature for next iteration
                    FullDatesDf.loc[i+1, 'Temperature_lag1Min'] = pred[0, 0]

                elif pd.isna(target_val):
                    # Single NaN or end of a gap: Use Forward Fill
                    FullDatesDf.loc[i, 'Temperature (oC)'] = FullDatesDf.loc[i-1, 'Temperature (oC)']
            
        # write to cleaned data csv file to prevent running the same code over and over 
        # when the data is 100k rows+

        file_name = "Cleaned_air_qual_log.csv" if "eCo2" in self.df.columns else "Cleaned_temp_log.csv" 
        folder = r"C:\Users\Charl\OneDrive\Python\Projects\Temp Data\\"
        location =  os.path.join(folder, file_name)

        # 2. Check if the FOLDER exists. If not, create it.
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
            
        # 3. Check if the FILE exists so we know if we need a header
        file_exists = os.path.isfile(location)
            
        #4. Use 'a' for APPEND so we don't delete previous data
        header = FullDatesDf.columns
        payload = FullDatesDf.loc[:i+1].values
        with open(location, 'a') as file:
            if not file_exists:
                # Add the header only if the file was just created
                file.write(header)
                

            # 5. Append the new data
            file.write(f"{payload}\n")
        
        print(f"Process Complete. File saved to: {location}")


                          
    def RunPipeline(self):
        """Helper method to run everything in the correct order."""
        self.DataLoad()
        self.CleanDateData()
        self.RemoveDuplicates()
        self.RemoveOutliers()
        #self.ImputeMissingData()
        self.AddTimeSeriesFeatures()

        return self.df



