import json
import pandas as pd
import tensorflow_addons as tfa

HAP_df = pd.read_parquet('data/pricing_data/avgHighPrice.parquet.gzip')
LAP_df = pd.read_parquet('data/pricing_data/avgLowPrice.parquet.gzip')
HAV_df = pd.read_parquet('data/pricing_data/highPriceVolume.parquet.gzip')
LAV_df = pd.read_parquet('data/pricing_data/lowPriceVolume.parquet.gzip')

#take a weighted average of the prices to get overall avrg price
weighted_avg = ((HAP_df.values*HAV_df.values)+(LAV_df.values*HAV_df.values))/(LAV_df.values+HAV_df.values)
df = pd.DataFrame(columns = HAP_df.columns, data = weighted_avg)
totv = pd.DataFrame(columns = HAP_df.columns, data = (LAV_df.values+HAV_df.values)) #a dataframe of volume traded that day

with open("data/name_dicts/ID_to_name.json","r") as f:
    id2name=json.load(f)
with open("data/name_dicts/name_to_ID.json","r") as f:
    name2id=json.load(f)

prices = df[str(name2id["Abyssal whip"])].to_list() #lists of time and value
times = df["timestamps"].to_list()


print("file completed successfully")