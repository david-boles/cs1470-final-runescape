from itertools import repeat
import json
from math import ceil, floor
from sklearn.metrics import mean_squared_error as MSE
# from turtle import st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ta
import tensorflow_addons as tfa
from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers import LSTM, Dropout,Dense
import tensorflow as tf

def get_data(data_path, window_size=24, interp_limit=1, train_set_ratio=0.8):
    """
    interp_limit defines the maximum region that data is allowed
    to be interpolated within. Can be 0 for no interpolation or null
    for no limit. Items for which this condition cannot be met are
    removed. Should probably be less than the window size.

    Returns a tuple of train_input, train_output, test_input, test_output.
    Inputs are ndarrays of shape num_examples x window_size x num_items x num_input_metrics.
    Where the input metrics are the following:
    High price, low price, high volume, low volume, ...engineered metrics.
    Outputs are ndarrays of shape num_examples x num_items x num_output_metrics.
    Where the output metrics are the four non-engineered metrics.
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

    print("Loaded raw data")

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
        # Also remove the timestamps as they're no longer needed.
        # Copies shouldn't be strictly necessary (esp. given regions
        # are non-overlapping) but they make Pandas happier when we
        # operate inplace later on.
        period_HAP_dfs.append(raw_HAP_df.iloc[start:end, 1:].copy()),
        period_LAP_dfs.append(raw_LAP_df.iloc[start:end, 1:].copy()),
        period_HAV_dfs.append(raw_HAV_df.iloc[start:end, 1:].copy()),
        period_LAV_dfs.append(raw_LAV_df.iloc[start:end, 1:].copy()),

    num_periods = len(period_HAP_dfs)
    period_lengths = [df.shape[0] for df in period_HAP_dfs]
    item_index = period_HAP_dfs[0].columns

    print("Separated into contiguous regions")

    # If an item isn't traded, both the volume and prices will be NaN.
    # Fill these with the correct and interpolated values respectively.
    # We don't necessarily want to interpolate prices for extremely long
    # periods however, so there's a configurable limit.
    # Any items with periods of no trading so long that they don't
    # have interpolated prices get removed.
    for V_df in [*period_HAV_dfs, *period_LAV_dfs]:
        V_df.fillna(0, inplace=True)

    col_has_nan_after_interp_s = pd.Series(False, index=item_index)
    for ind, P_df in enumerate([*period_HAP_dfs, *period_LAP_dfs]):
        if interp_limit > 0:
            P_df.interpolate(
                method="linear",
                limit_direction="both",
                limit=interp_limit,
                inplace=True,
            )
        print(f"Interpolated {ind+1} dataframes")

        col_has_nan_after_interp_s |= P_df.isna().sum() > 0

    # Make sure we're left with at least some valid items
    assert col_has_nan_after_interp_s.sum() < len(item_index)

    to_drop = [
        col for col, has_nan in col_has_nan_after_interp_s.iteritems() if has_nan
    ]
    for df in [*period_HAP_dfs, *period_LAP_dfs, *period_HAV_dfs, *period_LAV_dfs]:
        df.drop(columns=to_drop, inplace=True)

    item_index = period_HAP_dfs[0].columns

    print("Finished 0 volume processing")

    # Feature engineeering
    # Added features - Rate of change, Binned Rate of change, Moving average,
    # Ease of movement, Ulcer Volatility (?), Mass Index
    period_ROC_dfs = []
    period_ROC_bin_dfs = []
    period_MA_dfs = []
    period_EOM_dfs = []
    # period_Volatility_dfs = []
    period_MI_dfs = []

    for per_ind in range(num_periods):
        dfs = [
            period_HAP_dfs[per_ind],
            period_LAP_dfs[per_ind],
            period_HAV_dfs[per_ind],
            period_LAV_dfs[per_ind],
        ]
        arrays = [df.values for df in dfs]
        data_matrix = np.stack(arrays, axis=0)
        # 1<Bin_size<window_size, otherwise weird things happen
        # Could set to something like floor(period_lengths/10) or something
        bin_size = 10
        temp_list_roc = []
        temp_list_roc_bin = []
        temp_list_ma = []
        temp_list_EOM = []
        # temp_list_Ulcer = []
        temp_list_MI = []
        for item_ind in range(data_matrix.shape[2]):
            # Rate of change
            # First value is always Na because no change from first one
            feature_roc = ta.momentum.ROCIndicator(
                close=pd.Series(data_matrix[0, :, item_ind]), window=1, fillna=True
            )
            generate_roc = feature_roc.roc()
            temp_list_roc.append(generate_roc)

            # Binned rate of change
            # Rate of change calculated over average value over bin_size time stamps
            # First bin_size values are Na # TODO problem?
            feature_roc_bin = ta.momentum.ROCIndicator(
                close=pd.Series(data_matrix[0, :, item_ind]), window=bin_size, fillna=True
            )
            generate_roc_bin = feature_roc_bin.roc()
            temp_list_roc_bin.append(generate_roc_bin)

            # Moving Average for bins
            feature_MA = ta.trend.SMAIndicator(
                close=pd.Series(data_matrix[0, :, item_ind]), window=bin_size, fillna=True
            )
            generate_MA = feature_MA.sma_indicator()
            temp_list_ma.append(generate_MA)

            # Ease of movement
            feature_EOM = ta.volume.EaseOfMovementIndicator(
                high=pd.Series(data_matrix[0, :, item_ind]),
                low=pd.Series(data_matrix[1, :, item_ind]),
                volume=pd.Series(data_matrix[2, :, item_ind]),
                fillna=True
            )
            generate_EOM = feature_EOM.ease_of_movement()
            temp_list_EOM.append(generate_EOM)

            # Ulcer index for volatility https://school.stockcharts.com/doku.php?id=technical_indicators:ulcer_index
            # Takes a while to compute, uncomment below to run (need to uncomment temp_list_ulcer, Folatility_df declaration too)
            # feature_Volatility = ta.volatility.UlcerIndex(close = pd.Series(data_matrix[0,:,i]), window = bin_size)
            # generate_Volatility = feature_Volatility.ulcer_index()
            # temp_list_Ulcer.append(generate_Volatility)

            # Mass index, also a volatility indicator tracks change in trend
            # https://www.investopedia.com/terms/m/mass-index.asp#:~:text=Mass%20index%20is%20a%20form,certain%20point%20and%20then%20contracts.
            # Quite a bit faster than Ulcer index, but not a great volatility indicator, only useful for finding inflections
            feature_MI = ta.trend.mass_index(
                high=pd.Series(data_matrix[0, :, item_ind]),
                low=pd.Series(data_matrix[1, :, item_ind]),
                fillna=True,
            )
            temp_list_MI.append(feature_MI)

        period_ROC_dfs.append(
            pd.DataFrame(
                np.vstack(temp_list_roc).T,
                columns=item_index,
            )
        )
        period_ROC_bin_dfs.append(
            pd.DataFrame(
                np.vstack(temp_list_roc_bin).T,
                columns=item_index,
            )
        )
        period_MA_dfs.append(
            pd.DataFrame(
                np.vstack(temp_list_ma).T,
                columns=item_index,
            )
        )
        period_EOM_dfs.append(
            pd.DataFrame(
                np.vstack(temp_list_EOM).T,
                columns=item_index,
            )
        )
        # period_Volatility_dfs.append(
        #     d.DataFrame(
        #         np.vstack(temp_list_Ulcer).T,
        #         columns=item_index,
        #     )
        # )
        period_MI_dfs.append(
            pd.DataFrame(
                np.vstack(temp_list_MI).T,
                columns=item_index,
            )
        )

        print(f"Finished feature engineering for period {per_ind+1}")

    # Split each region into examples. Windows overlap (stride of 1) but test data is only
    # taken from the end of each period so that its output values have never
    # been trained on by the network.
    train_window_inds = []
    test_window_inds = []
    for item_ind, length in enumerate(period_lengths):
        window_inds = list(
            zip(
                repeat(item_ind),
                range(0, (length - window_size)),
                range(window_size, length),
            )
        )
        start_test_windows = floor(len(window_inds) * train_set_ratio)
        train_window_inds += window_inds[:start_test_windows]
        test_window_inds += window_inds[start_test_windows:]

    train_input = np.zeros((len(train_window_inds), window_size, len(item_index), 9))
    train_output = np.zeros((len(train_window_inds), len(item_index), 4))
    test_input = np.zeros((len(test_window_inds), window_size, len(item_index), 9))
    test_output = np.zeros((len(test_window_inds), len(item_index), 4))
    for window_inds, input, output in [
        [train_window_inds, train_input, train_output],
        [test_window_inds, test_input, test_output],
    ]:
        for example_ind, (period_ind, window_start_ind, window_end_ind) in enumerate(
            window_inds
        ):
            for feature_ind, (feature_in_output, period_dfs) in enumerate(
                [
                    [True, period_HAP_dfs],
                    [True, period_LAP_dfs],
                    [True, period_HAV_dfs],
                    [True, period_LAV_dfs],
                    [False, period_ROC_dfs],
                    [False, period_ROC_bin_dfs],
                    [False, period_MA_dfs],
                    [False, period_EOM_dfs],
                    [False, period_MI_dfs],
                ]
            ):
                input[example_ind, :, :, feature_ind] = period_dfs[period_ind].iloc[
                    window_start_ind:window_end_ind, :
                ]
                if feature_in_output:
                    output[example_ind, :, feature_ind] = period_dfs[period_ind].iloc[
                        window_end_ind, :
                    ]

    print("Done preprocessing data")

    return train_input, train_output, test_input, test_output


train_input, train_output, test_input, test_output = get_data("./data", 10, 1)

print(train_output.shape)

units = 1000
con = .3
leaky = .75
sr = .7
dense1 = 200
lr = .002


model = Sequential()
model.add(tfa.layers.ESN(units, connectivity = con, leaky = leaky, spectral_radius = sr, activation = 'tanh'))#, return_sequences = True ))
model.add(tf.keras.layers.Dense(dense1, activation="relu"))
model.add(tf.keras.layers.Dense(1000))
model.add(tf.keras.layers.Dense(500))
#model.add(Dropout(0.2))
model.add(tf.keras.layers.Dense(6919*4*836))
model.add(tf.keras.layers.Reshape((6919,836,4)))
opt = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=opt,loss='mean_squared_error')
history = model.fit(train_input,train_output,epochs=20,batch_size=24)

pass
