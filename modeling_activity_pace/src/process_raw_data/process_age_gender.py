import pandas as pd

from src.helpers import rename_columns


ENCODE_AGE = {
    "Entre 45 et 54 ans": 4,
    "Entre 18 et 24 ans": 1,
    "Moins de 18 ans": 0,
    "Entre 25 et 34 ans": 2,
    "Entre 55 et 64 ans": 5,
    "Entre 35 et 44 ans": 3,
    "+ de 65 ans": 6,
}

ENCODE_GENDER = {
    "Un homme": 0,
    "Une femme": 1,
    "Je ne souhaite pas répondre": 2,
    "Je préfère me décrire moi-même": 2,
    "Non-binaire/Transgenre": 2,
}

COL_NAMES_AGE_GENDER = [
    "E_birth_year",
    "E_age_range",
    "E_gender",
]

NEW_COL_NAMES_AGE_GENDER = [
    "annee_naissance",
    "age_group",
    "gender",
]


def convert_age_to_int(df):
    """
    Convert the 'annee_naissance' column in a DataFrame to integer values.

    Args:
        df (pd.DataFrame): The DataFrame in which 'annee_naissance' values should be converted.

    Returns:
        pd.DataFrame: The DataFrame with 'annee_naissance' values converted to integers.
    """
    df["annee_naissance"] = df["annee_naissance"].apply(
        lambda x: int(x) if not pd.isna(x) else -1
    )
    return df


def encode_age_category(df):
    """
    Encode the 'age_group' column in a DataFrame based on predefined age categories.

    Args:
        df (pd.DataFrame): The DataFrame in which 'age_group' values should be encoded.

    Returns:
        pd.DataFrame: The DataFrame with 'age_group' values encoded.
    """
    df["age_group"] = df["age_group"].apply(
        lambda x: ENCODE_AGE[x] if isinstance(x, str) else -1
    )
    return df


def encode_gender(df):
    """
    Encode the 'genre' column in a DataFrame based on predefined gender categories.

    Args:
        df (pd.DataFrame): The DataFrame in which 'genre' values should be encoded.

    Returns:
        pd.DataFrame: The DataFrame with 'genre' values encoded.
    """
    df["gender"] = df["gender"].apply(
        lambda x: ENCODE_GENDER[x] if isinstance(x, str) else -1
    )
    return df


def assign_age_category(df):
    """
    Assign age categories based on the 'annee_naissance' column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame in which age categories should be assigned.

    Returns:
        pd.DataFrame: The DataFrame with 'age_group' values assigned based on age.
    """
    df["age_group"] = df["annee_naissance"].apply(
        lambda x: assign_age_category_helper(2023 - x)
    )
    return df


def assign_age_category_helper(age):
    """
    Helper function to assign age categories based on age.

    Args:
        age (int): The age value.

    Returns:
        int: The assigned age category.
    """
    if age < 18:
        return 0
    elif age <= 24:
        return 1
    elif age <= 34:
        return 2
    elif age <= 44:
        return 3
    elif age <= 54:
        return 4
    elif age <= 64:
        return 5
    else:
        return 6


def process_age_gender(df):
    """
    Process a DataFrame by renaming columns and encoding age and gender categories.

    Args:
        df (pd.DataFrame): The DataFrame to be processed.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    df = rename_columns(df, COL_NAMES_AGE_GENDER, NEW_COL_NAMES_AGE_GENDER)
    df = convert_age_to_int(df)
    df = encode_age_category(df)
    df = encode_gender(df)
    df = assign_age_category(df)
    return df
