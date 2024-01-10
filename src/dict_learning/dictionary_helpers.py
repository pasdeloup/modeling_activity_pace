import os
import numpy as np

from sklearn.linear_model import LinearRegression

from src.process_answers import ProcessAnswers
from src.settings import (
    n_channels,
    channel_names,
    ANSWERS_ACTIVITY_COLUMNS,
    ANSWERS_PATH,
    DICT_ITER_PATH,
    SERIES_LENGTH
)
from src.helpers import load_data, get_ids_from_signals
from src.modeling_functions import convolve_signals, normalize_signals, log_reg, scoring_auc


def process_data_for_DL():
    """
    Process data for dictionary learning.

    This function reads CSV files into a list of DataFrames, extracts user IDs, defines columns to predict,
    processes the answers data, and prepares the input data by convolving and normalizing.

    Returns:
        tuple: A tuple containing input data (X_list_array_clean) and target labels (y).
    """
    X_list = [load_data(f"data/processed/streams/X_{i}.csv") for i in channel_names]

    ids = get_ids_from_signals(X_list)
    cols_to_predict = ANSWERS_ACTIVITY_COLUMNS

    answers_processor = ProcessAnswers(ANSWERS_PATH)

    answers_processor.process(ids, cols_to_predict)
    df_answers = answers_processor.df

    y = df_answers[cols_to_predict].to_numpy()

    X_list_array = np.array([x.to_numpy()[:, :].T for x in X_list])
    X_list_array_clean = np.zeros(
        (
            X_list_array.shape[0],
            X_list_array.shape[1] - 2,
            X_list_array.shape[2],
        )
    )

    for chan in range(n_channels):
        X_list_array_clean[chan, :, :] = convolve_signals(X_list_array[chan, :, :])
        X_list_array_clean[chan, :, :] = normalize_signals(X_list_array_clean[chan, :, :])

    return X_list_array_clean, y


def make_dict_list():
    """
    This function loads a list of the computed dictionary iterations.

    Returns:
        list: A list of dictionary iterations.
    """
    dict_list = [
        np.load(DICT_ITER_PATH + file)[:, :, :SERIES_LENGTH]
        for file in [f"D_{i}.npy" for i in range(len(os.listdir(DICT_ITER_PATH)))]
    ]
    return dict_list


def reshape_X_and_D(X, D):
    """
    Reshape input data and dictionary for data projection on atoms.

    Args:
        X (numpy.ndarray): Input data.
        D (numpy.ndarray): Dictionary.

    Returns:
        tuple: A tuple containing reshaped input data (X_reshaped) and reshaped dictionary (D_reshaped).
    """
    X_reshaped = X.transpose((2, 0, 1)).reshape(
        (
            X.shape[2],
            X.shape[0] * X.shape[1],
        )
    )
    D_reshaped = D.reshape(D.shape[0], D.shape[1] * D.shape[2])
    return X_reshaped, D_reshaped
