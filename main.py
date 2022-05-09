from itertools import repeat
import json
from math import ceil, floor
import os
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ta
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import pickle
import tensorflow as tf


def get_data(window_size=10, interp_limit=1, train_set_ratio=0.8):
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
    Where the output metrics are delta high and low price.
    """
    # Cache folder is gitignored and file name is versioned and labeled
    # in an attempt to prevent stale caches (and because the picke is gibibytes in size :D ).
    # IF GET_DATA OR SOURCE PARQUET FILES ARE MODIFIED, BUMP THE VERSION NUMBER BELOW
    cache_path = f"./data/cache/preprocessed-v2-{window_size}-{interp_limit}-{train_set_ratio}.pickle"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Attempt to load cached pre-processed data first.
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            print(f"Found cache of preprocessed data at {cache_path}")
            return data
    except:
        print("Unable to find cache of preprocessed data")

    # Load dataframes for high/low price/volume
    # Columns are timestamp (Unix timestamps/integer seconds) and non-sequential integers
    # that can be corresponded to item names using the jsons in `data/name_dicts`.
    raw_HAP_df = pd.read_parquet("./data/pricing_data/avgHighPrice.parquet.gzip")
    raw_LAP_df = pd.read_parquet("./data/pricing_data/avgLowPrice.parquet.gzip")
    raw_HAV_df = pd.read_parquet("./data/pricing_data/highPriceVolume.parquet.gzip")
    raw_LAV_df = pd.read_parquet("./data/pricing_data/lowPriceVolume.parquet.gzip")

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

    # Compute delta of prices
    # NaNs are left in first row, but they won't be at the end of a window
    period_delta_HAP_dfs = [df.diff() for df in period_HAP_dfs]
    period_delta_LAP_dfs = [df.diff() for df in period_LAP_dfs]

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
                close=pd.Series(data_matrix[0, :, item_ind]),
                window=bin_size,
                fillna=True,
            )
            generate_roc_bin = feature_roc_bin.roc()
            temp_list_roc_bin.append(generate_roc_bin)

            # Moving Average for bins
            feature_MA = ta.trend.SMAIndicator(
                close=pd.Series(data_matrix[0, :, item_ind]),
                window=bin_size,
                fillna=True,
            )
            generate_MA = feature_MA.sma_indicator()
            temp_list_ma.append(generate_MA)

            # Ease of movement
            feature_EOM = ta.volume.EaseOfMovementIndicator(
                high=pd.Series(data_matrix[0, :, item_ind]),
                low=pd.Series(data_matrix[1, :, item_ind]),
                volume=pd.Series(data_matrix[2, :, item_ind]),
                fillna=True,
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
    # Output data is taken from the timestep after the window so windows do not include the
    # last timestep.
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

    input_features = [
        period_HAP_dfs,
        period_LAP_dfs,
        period_HAV_dfs,
        period_LAV_dfs,
        period_ROC_dfs,
        period_ROC_bin_dfs,
        period_MA_dfs,
        period_EOM_dfs,
        period_MI_dfs,
    ]

    output_features = [
        period_delta_HAP_dfs,
        period_delta_LAP_dfs,
    ]

    input_shape = (window_size, len(item_index), len(input_features))
    output_shape = (len(item_index), len(output_features))

    train_input = np.zeros((len(train_window_inds), *input_shape))
    train_output = np.zeros((len(train_window_inds), *output_shape))
    test_input = np.zeros((len(test_window_inds), *input_shape))
    test_output = np.zeros((len(test_window_inds), *output_shape))

    for window_inds, input, output in [
        [train_window_inds, train_input, train_output],
        [test_window_inds, test_input, test_output],
    ]:
        for example_ind, (period_ind, window_start_ind, window_end_ind) in enumerate(
            window_inds
        ):
            for feature_ind, period_dfs in enumerate(input_features):
                input[example_ind, :, :, feature_ind] = period_dfs[period_ind].iloc[
                    window_start_ind:window_end_ind, :
                ]

            for feature_ind, period_dfs in enumerate(output_features):
                output[example_ind, :, feature_ind] = period_dfs[period_ind].iloc[
                    window_end_ind, :
                ]

    print("Done preprocessing data")

    data = (train_input, train_output, test_input, test_output)

    # Sanity check for remaining NaNs
    for d_arr in data:
        assert not np.any(np.isnan(d_arr))

    with open(cache_path, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    print(f"Cached preprocessed data at {cache_path}")

    return data


def mean_squared_proportional_error(y_true, y_pred):
    # Both volume and prices are non-negative scalars
    # We want a small error relative to whatever true price or
    # volume the item actually has.
    # Since the true values can be 0, we add 1 so it doesn't divide by 0.
    return tf.reduce_mean(((y_true - y_pred) / (y_true + 1)) ** 2)


def mean_abs_proportional_error(y_true, y_pred):
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + 1)))


train_input, train_output, test_input, test_output = get_data()

num_items = train_input.shape[2]
num_output_features = train_output.shape[2]


# Optionally limit # items and, indirectly, network complexity for testing
num_items = 100
train_input = train_input[:, :, :num_items, :]
train_output = train_output[:, :num_items, :]
test_input = test_input[:, :, :num_items, :]
test_output = test_output[:, :num_items, :]

print(num_items)

train_input = train_input.reshape((*train_input.shape[:2], -1))
test_input = test_input.reshape((*test_input.shape[:2], -1))


units = num_items * 10  # arbitrary :shrug:
con = 0.3
leaky = 0.75
sr = 0.7
dense1 = 200
lr = 0.001


model = Sequential()
model.add(
    tfa.layers.ESN(
        units, connectivity=con, leaky=leaky, spectral_radius=sr, activation="tanh"
    )
)  # , return_sequences = True ))
model.add(tf.keras.layers.Dense(dense1, activation="relu"))
model.add(tf.keras.layers.Dense(num_items * 10))
model.add(tf.keras.layers.Dense(num_items * 10))
# model.add(Dropout(0.2))
model.add(tf.keras.layers.Dense(num_items * num_output_features))
model.add(tf.keras.layers.Reshape((num_items, num_output_features)))
opt = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(
    optimizer=opt,
    # loss="mean_squared_error",
    loss=mean_squared_proportional_error,
    metrics=[mean_abs_proportional_error],
)
history = model.fit(
    train_input,
    train_output,
    epochs=20,
    batch_size=24,
    validation_data=(test_input, test_output),
)

pass
