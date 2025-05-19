import pandas as pd

from tqdm import tqdm

from src.process_answers import ProcessAnswers
from src.settings import channel_names, ANSWERS_PATH
from src.helpers import load_data, get_ids_from_signals
from src.process_raw_data.answers_helpers import process_data_for_classifier
from src.modeling_functions import (
    scoring_auc,
    log_reg,
    grid_search,
    perform_grid_search,
    split,
)


def compute_other_activities_baseline(cols_to_predict):
    """
    Compute the baseline scores for different activities using Logistic Regression classifiers.

    Parameters:
    - cols_to_predict: List of feature columns to predict.

    Returns:
    - List of AUC scores.
    """
    X_list = [load_data(f"data/processed/streams/X_{i}.csv") for i in channel_names]

    # Extract user IDs
    ids = get_ids_from_signals(X_list)

    # Initialize the answers processor
    answers_processor = ProcessAnswers(ANSWERS_PATH)

    # Process the answers data
    answers_processor.process(ids, cols_to_predict)
    df_answers = answers_processor.df

    # Extract the target labels
    y = df_answers[cols_to_predict].to_numpy()

    scores = []
    for label_index in tqdm(
        range(len(cols_to_predict)),
        desc=f"Performing Grid search on {len(cols_to_predict)} labels",
    ):
        _, _, y_train_, y_test_ = split(y, y)
        y_train = y_train_[:, label_index]
        y_test = y_test_[:, label_index]
        X_train_other_activities = y_train_[
            :, [i for i in range(len(cols_to_predict)) if i != label_index]
        ]
        X_test_other_activities = y_test_[
            :, [i for i in range(len(cols_to_predict)) if i != label_index]
        ]
        classifier = log_reg()
        search = grid_search(classifier)
        search.fit(X_train_other_activities, y_train)

        model = log_reg()
        model.set_params(**search.best_params_)
        model.fit(X_train_other_activities, y_train_[:, label_index])

        auc = scoring_auc(model, X_test_other_activities, y_test)

        scores.append(round(auc, 2))
    return scores


def compute_total_volume_baseline(cols_to_predict):
    """
    Computes the total volume baseline.

    Parameters:
    - cols_to_predict (list): List of column names to predict.

    Returns:
    - scores (list): List of rounded AUC scores for each label after performing grid search.
    """

    # Load volume data
    df_volume = load_data("data/processed/streams/X_volume.csv")

    # Sum the volume across time (transpose and sum)
    df = pd.DataFrame(df_volume.T.sum(), columns=["Total volume"])

    # Extract user IDs
    ids = df.index.tolist()

    # Initialize the answers processor
    answers_processor = ProcessAnswers(ANSWERS_PATH)

    # Process the answers data
    answers_processor.process(ids, cols_to_predict)
    df_answers = answers_processor.df

    # Extract the target labels
    y = df_answers[cols_to_predict].to_numpy()

    # Initialize a list to store AUC scores
    scores = []

    # Iterate over each label for grid search
    for label_index in tqdm(
        range(len(cols_to_predict)),
        desc=f"Performing Grid search on {len(cols_to_predict)} labels",
    ):
        # Split the data into training and testing sets
        X_train, X_test, y_train_, y_test_ = split(df.to_numpy(), y)
        y_train = y_train_[:, label_index]
        y_test = y_test_[:, label_index]

        # Initialize logistic regression classifier
        classifier = log_reg()

        # Perform grid search for hyperparameter tuning
        search = grid_search(classifier)
        search.fit(X_train, y_train)

        # Initialize logistic regression model with best parameters
        model = log_reg()
        model.set_params(**search.best_params_)
        model.fit(X_train, y_train_[:, label_index])

        # Calculate AUC score and append to the scores list
        auc = scoring_auc(model, X_test, y_test)
        scores.append(round(auc, 2))

    return scores



def compute_baseline_scores(
    records_cols_input, no_time_series, cols_to_predict, add_norm=False
):
    """
    Compute baseline scores for various input configurations.

    Parameters:
    - records_cols_input (list): List of records answer columns to be considered as input input.
    - no_time_series (bool): Whether to exclude time series embeddings data.
    - add_norm (bool): Whether to include normalization.
    - cols_to_predict (list): Labels to predict.

    Returns:
    - dict: Dictionary of scores.
    """
    # Process data for the classifier
    X_train, X_test, y_train_, y_test_ = process_data_for_classifier(
        cols_to_predict, records_cols_input, no_time_series, add_norm
    )

    # Perform grid search for hyperparameter tuning
    scores = perform_grid_search(X_train, X_test, y_train_, y_test_, cols_to_predict)

    return scores

