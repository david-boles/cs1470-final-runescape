import json
from math import floor
from turtle import st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ta


def get_data(data_path, window_size, future_size, interp_window_ratio=0.9):
    """
    interp_window_ratio defines the maximum region that data is allowed
    to be interpolated within as a proportion of window_size. For example
    at 0.5, regions of interpolated data for all items included will not
    exceed 50% of the window period. Can be null for no limit. Items for
    which this condition cannot be met are removed.
    """
    # Load dataframes for high/low price/volume
    # Columns are timestamp (Unix timestamps/integer seconds) and non-sequential integers
    # that can be corresponded to item names using the jsons in `data/name_dicts`.
    raw_HAP_df = pd.read_parquet(f"{data_path}/pricing_data/avgHighPrice.parquet.gzip")
    raw_LAP_df = pd.read_parquet(f"{data_path}/pricing_data/avgLowPrice.parquet.gzip")
    raw_HAV_df = pd.read_parquet(
        f"{data_path}/pricing_data/highPriceVolume.parquet.gzip"
    )
    raw_LAV_df = pd.read_parquet(
        f"{data_path}/pricing_data/lowPriceVolume.parquet.gzip"
    )

    assert all(raw_HAP_df.columns == raw_LAP_df.columns)
    assert all(raw_HAP_df.columns == raw_HAV_df.columns)
    assert all(raw_HAP_df.columns == raw_LAV_df.columns)
    assert all(raw_HAP_df.timestamps == raw_LAP_df.timestamps)
    assert all(raw_HAP_df.timestamps == raw_HAV_df.timestamps)
    assert all(raw_HAP_df.timestamps == raw_LAV_df.timestamps)

    # Find points at which timestamps skip (probably due to server restarts),
    # and split the dataframes into contiguous periods.
    # This should be done first since other operations assume contiguous data.
    timestamps = raw_HAP_df.timestamps.to_numpy()
    (skips,) = ((timestamps[1:] - timestamps[:-1]) != 3600).nonzero()
    assert len(skips) == 10  # sanity check, reminder to look at this if data is changed
    indices = [0, *(skips + 1), len(timestamps)]
    indices = zip(indices[:-1], indices[1:])
    period_HAP_dfs = []
    period_LAP_dfs = []
    period_HAV_dfs = []
    period_LAV_dfs = []
    for (start, end) in indices:
        # Copies shouldn't be strictly necessary (esp. given regions
        # are non-overlapping) but they make Pandas happier when we
        # operate inplace later on.
        period_HAP_dfs.append(raw_HAP_df.iloc[start:end, :].copy()),
        period_LAP_dfs.append(raw_LAP_df.iloc[start:end, :].copy()),
        period_HAV_dfs.append(raw_HAV_df.iloc[start:end, :].copy()),
        period_LAV_dfs.append(raw_LAV_df.iloc[start:end, :].copy()),

    # If an item isn't traded, both the volume and prices will be NaN.
    # Fill these with the correct and interpolated values respectively.
    # We don't necessarily want to interpolate prices for extremely long
    # periods however, so there's a configurable limit.
    # Any items with periods of no trading so long that they don't
    # have interpolated prices get removed.
    for V_df in [*period_HAV_dfs, *period_LAV_dfs]:
        V_df.fillna(0, inplace=True)

    limit = None
    if interp_window_ratio is not None:
        # Round down: choosing a lower limit is safer than a higher one
        limit = floor(window_size * interp_window_ratio)

    col_has_nan_after_interp_s = pd.Series(False, index=raw_HAP_df.columns)
    for P_df in [*period_HAP_dfs, *period_LAP_dfs]:
        if limit > 0:
            P_df.interpolate(
                method="linear",
                limit_direction="both",
                limit=limit,
                inplace=True,
            )

        col_has_nan_after_interp_s |= P_df.isna().sum() > 0

    # Make sure we're left with at least some valid items, besides timestamps
    assert col_has_nan_after_interp_s.sum() + 1 < len(raw_HAP_df.columns)

    to_drop = [
        col for col, has_nan in col_has_nan_after_interp_s.iteritems() if has_nan
    ]
    for df in [*period_HAP_dfs, *period_LAP_dfs, *period_HAV_dfs, *period_LAV_dfs]:
        df.drop(columns=to_drop, inplace=True)
        pass

    #  Feature engineeering 
    dfs = [raw_HAP_df,raw_LAP_df, raw_HAV_df,raw_LAV_df]
    arrays = [df.values for df in dfs]
    data_matrix = np.stack(arrays, axis =0)
    bin_size = 10
    temp_list_roc = []
    temp_list_roc_bin = []
    temp_list_ma = []
    temp_list_EOM = []
    temp_list_Ulcer = []
    temp_list_MI = []
    for i in range(data_matrix.shape[2]):
        feature_roc = ta.momentum.ROCIndicator(close = pd.Series(data_matrix[0,:,i]), window = 1)
        generate_roc = feature_roc.roc()
        temp_list_roc.append(generate_roc)

        feature_roc_bin = ta.momentum.ROCIndicator(close = pd.Series(data_matrix[0,:,i]), window = bin_size)
        generate_roc_bin = feature_roc_bin.roc()
        temp_list_roc_bin.append(generate_roc_bin)

        feature_MA = ta.trend.SMAIndicator(close = pd.Series(data_matrix[0,:,i]), window = bin_size)
        generate_MA = feature_MA.sma_indicator()
        temp_list_ma.append(generate_MA)

        feature_EOM = ta.volume.EaseOfMovementIndicator(high= pd.Series(data_matrix[0,:,i]), low = pd.Series(data_matrix[1,:,i]), volume=pd.Series(data_matrix[2,:,i]))
        generate_EOM = feature_EOM.ease_of_movement()
        temp_list_EOM.append(generate_EOM)

        # feature_Volatility = ta.volatility.UlcerIndex(close = pd.Series(data_matrix[0,:,i]), window = bin_size)
        # generate_Volatility = feature_Volatility.ulcer_index()
        # temp_list_Ulcer.append(generate_Volatility)

        feature_MI = ta.trend.mass_index(high= pd.Series(data_matrix[0,:,i]), low = pd.Series(data_matrix[1,:,i]), fillna = True)
        temp_list_MI.append(feature_MI)

    ROC_df = pd.DataFrame(np.vstack(temp_list_roc).T)
    ROC_bin_df = pd.DataFrame(np.vstack(temp_list_roc_bin).T)
    MA_df = pd.DataFrame(np.vstack(temp_list_ma).T)
    EOM_df = pd.DataFrame(np.vstack(temp_list_EOM).T)
    # Volatility_df = pd.DataFrame(np.vstack(temp_list_Ulcer).T)
    MI_df = pd.DataFrame(np.vstack(temp_list_MI).T)


    # TODO: Split each region into examples. Will's previous code is below
    # but I think we need to think a little bit more about keeping the sets
    # distinct and representative if also want overlapping windows.

    # observations = []
    # values = []
    # for i in range(window_size, len(raw_HAP_df) - future_size):
    #     X_indices = list(range(i - window_size, i))
    #     y_indices = list(range(i, i + future_size))
    #     observations.append(data[:, X_indices, :])
    #     values.append(data[:, y_indices, :])

    # X = np.stack(observations, axis=0)
    # y = np.stack(values, axis=0)
    # # X_shape = observation, metrics, time, item
    # return X, y


get_data("./data", 10, 1)
