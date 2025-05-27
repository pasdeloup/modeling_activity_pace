import os
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(file_path, index_col=0, low_memory=False)


def save_data(df, path, index=False):
    """
    Save a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
    """
    df.to_csv(path, index=index)


def save_figure(path):
    plt.savefig(path, dpi=300, bbox_inches="tight", format="pdf")


def rename_columns(df, col_names, new_col_names):
    """
    Rename columns in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to rename columns in.
        col_names (list): List of current column names.
        new_col_names (list): List of new column names.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    rename_dict = dict(list(zip(col_names, new_col_names)))
    return df.rename(columns=rename_dict)


def build_result_folder():
    """
    Build the folder structure for storing results, figures, and tables.
    """
    folders = [
        "results",
        "results/dict_learning",
        "results/dict_learning/all_iterations",
        "results/figures",
        "results/figures/atoms",
        "results/tables",
        "results/tables/stats_logreg",
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def get_ids_from_signals(X_list):
    return list(X_list[0].T)[:]
