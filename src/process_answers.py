from src.helpers import load_data
from src.process_raw_data.filter import process_filter
from src.process_raw_data.process_activities_deezer import (
    process_activities_deezer_feature,
)
from src.process_raw_data.process_age_gender import process_age_gender
from src.helpers import load_data


class ProcessAnswers:
    def __init__(self, path) -> None:
        """
        Initialize the ProcessAnswers class.

        Args:
            path (str): The path to the CSV file containing answers data.
        """
        self.path = path

    def import_data(self):
        """
        Import data from the specified CSV file.
        """
        self.df = load_data(self.path).reset_index()

    def filter(self, ids, cols):
        """
        Filter the DataFrame based on user IDs and selected columns.

        Args:
            ids (list): List of user IDs to include in the filtered data.
            cols (list): List of column names to include in the filtered data.
        """
        sorting_dict = dict(list(zip(ids, range(len(ids)))))
        self.df = self.df[self.df["user_id"].isin(ids)]
        self.df = self.df[["user_id"] + cols]
        self.df = self.df.sort_values(by="user_id", key=lambda x: x.map(sorting_dict))

    def process(self, ids, cols):
        """
        Process the answers data by importing and filtering.

        Args:
            ids (list): List of user IDs to filter the data.
            cols (list): List of column names to filter the data.
        """
        self.import_data()
        self.filter(ids, cols)


def process_answer_data_pipeline(df, columns):
    """
    Process answers DataFrame by applying filtering and feature processing.

    Args:
        df (pd.DataFrame): The DataFrame to be processed.
        columns (list): List of columns to keep in the processed DataFrame.

    Returns:
        pd.DataFrame: The processed DataFrame with selected columns.
    """
    df = process_filter(df)
    df = process_activities_deezer_feature(df)
    df = process_age_gender(df)
    return df[columns]
