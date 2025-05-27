import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, auc

from src.settings import RS, PARAM_GRID, TEST_SIZE


def normalize_signals(Y):
    """
    Normalize the input array.

    Args:
        X (ndarray): The input array to be normalized.

    Returns:
        ndarray: The normalized array.

    """
    X = Y.T
    X = X - np.mean(X, axis=1).reshape(-1, 1)
    max_values = np.max(np.abs(X), axis=1).reshape(-1, 1)
    max_values = np.array([[1 if item[0] == 0 else item[0]] for item in max_values])
    X = X / max_values
    return X.T


def convolve_signals(X):
    """
    Convolve each row of the input array with [1, 1, 1].

    Args:
        X (ndarray): The input array for convolution.

    Returns:
        ndarray: The convolved array.

    """
    return np.array([np.convolve(x, [1, 1, 1], mode="valid") for x in X.T]).T


def scoring_auc(est, x, y):
    """
    Calculate the area under the ROC curve (AUC) score for a given estimator.

    Parameters:
    - est: The estimator (classifier).
    - x: The input features.
    - y: The true labels.

    Returns:
    - AUC score.
    """

    fpr, tpr, _ = roc_curve(y, est.predict_proba(x)[:, 1])
    return auc(fpr, tpr)


def grid_search(model):
    """
    Perform grid search for hyperparameter tuning.

    Parameters:
    - model: Machine learning model to be tuned.

    Returns:
    - GridSearchCV: Grid search object with specified parameters.
    """
    return GridSearchCV(
        model,
        PARAM_GRID,
        cv=5,
        scoring=scoring_auc,
        verbose=0,
        n_jobs=1,
    )


def log_reg():
    """
    Create a logistic regression classifier with balanced class weights.

    Returns:
    - LogisticRegression: Logistic regression classifier with specified parameters.
    """
    return LogisticRegression(class_weight="balanced", random_state=RS, max_iter=10000)


def split(x, y):
    """
    Split the input data into training and testing sets.

    Parameters:
    - x: Features or input data.
    - y: Target labels.

    Returns:
    - Tuple: (X_train, X_test, y_train, y_test) representing the split datasets.
    """
    return train_test_split(x, y, test_size=TEST_SIZE, random_state=RS)

def train_linear_reg(D, X):
    """
    Args:
        D (numpy.ndarray): Dictionary.
        X (numpy.ndarray): Input data.

    Returns:
        LinearRegression: Trained linear regression model.
    """
    reg = LinearRegression()
    reg.fit(D, X)
    return reg


def get_logistic_reg_score(X_train, X_test, y_train_, y_test_, label_index):
    """
    Get the logistic regression score.

    This function calculates the logistic regression roc_auc for a specific label.

    Args:
        X_train (numpy.ndarray): Training input data.
        X_test (numpy.ndarray): Testing input data.
        y_train_ (numpy.ndarray): Training target labels.
        y_test_ (numpy.ndarray): Testing target labels.
        label_index (int): Index of the label to predict.

    Returns:
        float: The area under the ROC curve (AUC) score.
    """
    y_train = y_train_[:, label_index]
    y_test = y_test_[:, label_index]

    model = log_reg().fit(X_train, y_train)
    auc = scoring_auc(model, X_test, y_test)
    return auc

def perform_grid_search(X_train, X_test, y_train_, y_test_, cols_to_predict):
    """
    Perform grid search for logistic regression classifiers and return AUC scores.

    Parameters:
    - X_train: Training input features.
    - X_test: Testing input features.
    - y_train_: Training labels.
    - y_test_: Test labels.
    - cols_to_predict: List of feature columns to predict.

    Returns:
    - List of AUC scores.
    """
    scores = []
    for label_index in tqdm(
        range(len(cols_to_predict)),
        desc=f"Performing Grid search on {len(cols_to_predict)} labels",
    ):
        y_train = y_train_[:, label_index]
        y_test = y_test_[:, label_index]
        classifier = log_reg()
        search = grid_search(classifier)
        search.fit(X_train, y_train)

        model = log_reg()
        model.set_params(**search.best_params_)
        model.fit(X_train, y_train_[:, label_index])
        auc = scoring_auc(model, X_test, y_test)
        scores.append(round(auc, 2))
    return scores