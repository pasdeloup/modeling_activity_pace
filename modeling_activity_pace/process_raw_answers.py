from src.process_raw_data.process_age_gender import NEW_COL_NAMES_AGE_GENDER
from src.process_answers import process_answer_data_pipeline
from src.helpers import load_data, save_data
from src.settings import ANSWERS_ACTIVITY_COLUMNS


def main():
    """
    Main function to process data and save the processed DataFrame to a CSV file.
    """
    FILE_PATH = "data/raw/answers/records.csv"
    COLS_TO_KEEP = ["user_id"] + ANSWERS_ACTIVITY_COLUMNS + NEW_COL_NAMES_AGE_GENDER
    COLS_TO_KEEP.remove("annee_naissance")

    df = load_data(FILE_PATH)
    df = process_answer_data_pipeline(df, COLS_TO_KEEP)
    save_data(df, "data/processed/answers/processed_records.csv")


if __name__ == "__main__":
    main()
