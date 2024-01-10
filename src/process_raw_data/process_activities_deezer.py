from src.helpers import rename_columns
from src.settings import ANSWERS_ACTIVITY_COLUMNS

COL_NAMES_CONTEXT = [f"B_contexts_deezer_{i}" for i in [1, 4, 5, 2, 12, 10]]


def encode_activities_columns(df, new_col_names):
    """
    Encode context-related columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to encode context columns in.
        new_col_names (list): List of new context-related column names.

    Returns:
        pd.DataFrame: The DataFrame with context columns encoded.
    """
    df[new_col_names] = df[new_col_names].map(lambda x: 1 if isinstance(x, str) else 0)
    return df


def process_activities_deezer_feature(df):
    """
    Process context-related features in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing context-related features.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    df = rename_columns(df, COL_NAMES_CONTEXT, ANSWERS_ACTIVITY_COLUMNS)
    df = encode_activities_columns(df, ANSWERS_ACTIVITY_COLUMNS)
    return df
