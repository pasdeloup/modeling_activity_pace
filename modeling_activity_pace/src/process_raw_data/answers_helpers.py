import numpy as np

from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit

from src.process_answers import ProcessAnswers
from src.helpers import load_data, get_ids_from_signals
from src.modeling_functions import convolve_signals, normalize_signals, split
from src.settings import (
    TIME_SERIES_PATH,
    channel_names,
    n_channels,
    ANSWERS_PATH,
)


def process_data_for_classifier(
    cols_to_predict,
    records_cols_input=[],
    no_time_series=False,
    add_norm=False,
    omp_param=None,
):
    """
    Process data for classification tasks.

    Parameters:
    - cols_to_predict (list): List of feature columns to predict.
    - records_cols_input (list): List of columns in records that will be considered as input in the model.
    - no_time_series (bool): If True, use only records_cols_input without time series data.
    - add_norm (bool): If True, add norm of non normalized time series data to the input features.
    - omp_param (int or None): Number of non-zero coefficients for Orthogonal Matching Pursuit. None for Linear Regression.

    Returns:
    - X_train: Training input features.
    - X_test: Test input features.
    - y_train_: Training labels.
    - y_test_: Test labels.
    """
    X_list = [load_data(f"{TIME_SERIES_PATH}/X_{i}.csv") for i in channel_names]
    D = np.load("results/dict_learning/D.npy")[:, :, :166]

    # Extract user IDs
    ids = get_ids_from_signals(X_list)

    # Define columns to predict from answers

    # Initialize the answers processor
    answers_processor = ProcessAnswers(ANSWERS_PATH)

    # Process the answers data
    answers_processor.process(ids, cols_to_predict)
    df_answers = answers_processor.df

    # Extract the target labels
    y = df_answers[cols_to_predict].to_numpy()

    answers_procesor_input = ProcessAnswers(ANSWERS_PATH)
    answers_procesor_input.process(ids, records_cols_input)
    df_answers_input = answers_procesor_input.df
    y_input = df_answers_input[records_cols_input].to_numpy()

    if no_time_series:
        X_emb = y_input
    else:
        # Prepare the input data by convolving and normalizing
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

        X_reshaped = X_list_array_clean.transpose((2, 0, 1)).reshape(
            (
                X_list_array_clean.shape[2],
                X_list_array_clean.shape[0] * X_list_array_clean.shape[1],
            )
        )
        D_reshaped = D.reshape(D.shape[0], D.shape[1] * D.shape[2])

        if omp_param == None:
            reg = LinearRegression()
        else:
            reg = OrthogonalMatchingPursuit(n_nonzero_coefs=omp_param)
        reg.fit(D_reshaped.T, X_reshaped.T)
        X_emb = reg.coef_

        if add_norm:
            norms = np.linalg.norm(X_list_array, axis=1).T
            X_emb = np.hstack([X_emb, y_input, norms])

        else:
            X_emb = np.hstack([X_emb, y_input])

    X_train, X_test, y_train_, y_test_ = split(X_emb, y)
    return X_train, X_test, y_train_, y_test_
