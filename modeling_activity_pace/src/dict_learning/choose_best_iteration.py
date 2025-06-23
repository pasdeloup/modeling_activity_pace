import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.dict_learning.dictionary_helpers import make_dict_list, reshape_X_and_D
from src.dict_learning.figures import (
    plot_classification_iterations_perfs,
    plot_reconstruction_iterations_perfs,
)

from src.modeling_functions import (
    train_linear_reg,
    get_logistic_reg_score,
    split,
    normalize_signals,
)
from src.settings import DICT_ITER_PATH, channel_names, n_channels
from src.helpers import save_data


def reconstruction_scores_iterations(X):
    all_scores = []
    dict_list = make_dict_list()
    for chan in range(n_channels):
        X_ = X[chan, :, :]

        dict_scores = []
        for D in tqdm(dict_list):
            D_ = D[:, chan, :]
            reg = train_linear_reg(D_.T, X_)
            reconstructed_signals = reg.coef_ @ D_
            reconstructed_signals = normalize_signals(reconstructed_signals.T).T

            dict_scores.append(
                np.mean(np.sqrt(np.mean((reconstructed_signals - X_.T) ** 2, axis=1)))
            )
        all_scores.append(dict_scores)
    return all_scores


def classification_scores_iterations(X, y, cols_to_predict):
    all_scores = []
    dict_list = make_dict_list()
    for D in tqdm(dict_list):
        X_reshaped, D_reshaped = reshape_X_and_D(X, D)
        reg = train_linear_reg(D_reshaped.T, X_reshaped.T)

        X_emb = reg.coef_
        X_train, X_test, y_train_, y_test_ = split(X_emb, y)
        scores = []
        for label_index, _ in enumerate(cols_to_predict):
            roc_auc = get_logistic_reg_score(
                X_train, X_test, y_train_, y_test_, label_index
            )
            scores.append(roc_auc)
        all_scores.append(scores)

    all_scores = np.array(all_scores)
    y_values = list(np.mean(all_scores, axis=1))
    return y_values


def choose_best_dict(X, y, cols_to_predict):
    """
    Choose the best dictionary iteration based on performance.

    This function iterates through the list of computed dictionaries step by step, trains a logistic regression model,
    evaluates its performance, and ask to select the best iteration.

    Args:
        X (numpy.ndarray): The input data.
        y (numpy.ndarray): The target labels.
        cols_to_predict (list): List of column indices to predict.

    Returns:
        None
    """
    reconstruction_scores = reconstruction_scores_iterations(X)
    class_scores = classification_scores_iterations(X, y, cols_to_predict)
    x_values = list(range(len(class_scores)))

    plot_classification_iterations_perfs(
        x_values,
        class_scores,
        "results/figures/classification_scores_over_iterations.pdf",
    )

    df_class_scores = pd.DataFrame(
        [class_scores, x_values], index=["mean_perf", "iteration"]
    )
    save_data(
        df_class_scores,
        "results/tables/classification_scores_over_iterations.csv",
        index=True,
    )

    df_reconstruction_scores = pd.DataFrame(
        reconstruction_scores, index=channel_names
    ).T
    save_data(
        df_reconstruction_scores,
        "results/tables/reconstruction_scores_over_iterations.csv",
        index=True,
    )
    plot_reconstruction_iterations_perfs(df_reconstruction_scores, "results/figures/reconstruction_scores_over_iterations.pdf")

    ## Uncomment here to manually select the best iteration
    # best_iter = int(input("Choose the best iteration :  "))

    best_iter_offset = 5
    best_iter = best_iter_offset + class_scores[best_iter_offset:].index(max(class_scores[best_iter_offset:]))
    print(f"Best iteration is {best_iter} with score {class_scores[best_iter]}")

    shutil.copy(
        DICT_ITER_PATH
        + [f"D_{i}.npy" for i in range(len(os.listdir(DICT_ITER_PATH)))][best_iter],
        "results/dict_learning/D.npy",
    )
