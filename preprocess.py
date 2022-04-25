import pandas as pd
import numpy as np

def get_data(data_path, window_size, future_size):
    HAP_df = pd.read_parquet(f'{data_path}/pricing_data/avgHighPrice.parquet.gzip')
    LAP_df = pd.read_parquet(f'{data_path}/pricing_data/avgLowPrice.parquet.gzip')
    HAV_df = pd.read_parquet(f'{data_path}/pricing_data/highPriceVolume.parquet.gzip')
    LAV_df = pd.read_parquet(f'{data_path}/pricing_data/lowPriceVolume.parquet.gzip')
    
    HAV_df = HAV_df.fillna(0)
    LAV_df = LAV_df.fillna(0)
   
    HAP_df = HAP_df.interpolate(method = "linear")
    LAP_df = LAP_df.interpolate(method = "linear")
    
    HAP_df = HAP_df.iloc[300: , :]
    LAP_df = LAP_df.iloc[300: , :]
    HAV_df = HAV_df.iloc[300: , :]
    LAV_df = LAV_df.iloc[300: , :]
    
    
    cols = []
    for col in HAP_df.columns:
        if HAP_df[col].isnull().any() == False and LAP_df[col].isnull().any() == False:
            cols.append(col)
    
    HAP_df = HAP_df[cols]
    LAP_df = LAP_df[cols]
    HAV_df = HAV_df[cols]
    LAV_df = LAV_df[cols]
    
    dfs = [HAP_df,LAP_df, HAV_df,LAV_df]
    arrays = [df.values for df in dfs]
    data = np.stack(arrays, axis =0)
    #shape = (metric, time, item)
    
    observations =[]
    values = []
    for i in range(window_size,len(HAP_df)-future_size):
        X_indices = list(range(i-window_size,i))
        y_indices = list(range(i, i+future_size))
        observations.append(data[:,X_indices,:])
        values.append(data[:,y_indices,:])
    
    X = np.stack(observations,axis = 0)
    y = np.stack(values, axis = 0)
    #X_shape = observation, metrics, time, item
    return X,y