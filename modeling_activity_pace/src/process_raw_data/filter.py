import numpy as np

from src.helpers import rename_columns


COL_NAMES = [
    "ResponseId",
    "uid",
]
NEW_COL_NAMES = [
    "id",
    "user_id",
]


def apply_filters(df):
    """
    Apply a set of predefined filters to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to which filters should be applied.

    Returns:
        pd.DataFrame: DataFrame with applied filters.
    """
    filters = {
        "Status": ["IP Address"],
        "Progress": ['100'],
        "Duration (in seconds)": [str(i) for i in list(range(3600))],
        "Q_consent": ["Oui"],
    }
    print(f"Initial number of data: {len(df)}")

    for col, filter_values in filters.items():
        df = filter_dataframe(df, col, filter_values)

    return df


def filter_dataframe(df, col, filter_values):
    """
    Filter a DataFrame based on a specified column and filter values.

    Args:
        df (pd.DataFrame): The DataFrame to be filtered.
        col (str): The name of the column to filter on.
        filter_values (list): List of values to filter on.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    filtered_df = df[df[col].isin(filter_values)]
    print(f"After filter on {col} feature, {len(filtered_df)} data")
    return filtered_df


def remove_duplicate_user_ids(df):
    """
    Remove duplicate user IDs, keeping only the last occurrence.

    Args:
        df (pd.DataFrame): The DataFrame from which duplicate user IDs should be removed.

    Returns:
        pd.DataFrame: The DataFrame with duplicate user IDs removed, keeping the last occurrence.
    """
    df = df.drop_duplicates(subset="user_id", keep="last")

    return df


def convert_user_id_to_int(df):
    """
    Convert the 'user_id' column in a DataFrame to integer values.

    Args:
        df (pd.DataFrame): The DataFrame in which 'user_id' values should be converted.

    Returns:
        pd.DataFrame: The DataFrame with 'user_id' values converted to integers.
    """
    df["user_id"] = df["user_id"].apply(lambda x: int(x) if not np.isnan(float(x)) else -1)
    return df


def process_filter(df):
    """
    Process a DataFrame by applying a series of filtering and data transformation steps.

    Args:
        df (pd.DataFrame): The DataFrame to be processed.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    df = rename_columns(df, COL_NAMES, NEW_COL_NAMES)
    df = apply_filters(df)
    df = remove_duplicate_user_ids(df)
    df = convert_user_id_to_int(df)
    return df
