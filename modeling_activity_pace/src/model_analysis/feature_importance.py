import pandas as pd
import numpy as np
from tqdm import tqdm

import statsmodels.api as sm

from src.modeling_functions import log_reg
from src.process_raw_data.answers_helpers import process_data_for_classifier
from src.model_analysis.figures import make_heatmap
from src.model_analysis.figures import write_logreg_report


def feature_importance_logreg(cols_to_predict, cols_to_predict_label):
    """
    Calculate feature importances using Logistic Regression and save them as a heatmap.

    Parameters:
    - cols_to_predict (list): List of feature columns to predict.
    - cols_to_predict_label (list): Label for the predicted columns.

    Returns:
    - None
    """
    X_train, _, y_train_, _ = process_data_for_classifier(
        cols_to_predict, records_cols_input=[]
    )
    importances = []
    for label_index in tqdm(range(len(cols_to_predict))):
        model = log_reg().fit(X_train, y_train_[:, label_index])

        importances.append(np.array(list(model.coef_[0])))

        model = sm.Logit(y_train_[:, label_index], X_train)
        result = str(model.fit().summary())
        write_logreg_report(cols_to_predict[label_index], result)


    n_atoms = X_train.shape[1]
    heatmap_df = pd.DataFrame(
        importances,
        columns=[f"atom{i} " for i in range(n_atoms)] + [],
        index=cols_to_predict_label,
    )
    make_heatmap(
        heatmap_df, np.max(heatmap_df), path="results/figures/logreg_importance.pdf"
    )
