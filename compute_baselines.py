import pandas as pd

from src.baselines.baselines_functions import (
    compute_other_activities_baseline,
    compute_total_volume_baseline,
    compute_baseline_scores,
)
from src.baselines.figures import create_image_of_baseline_scores
from src.helpers import save_data, build_result_folder
from src.settings import ANSWERS_ACTIVITY_COLUMNS, ACTIVITY_LABELS


cols_to_predict = ANSWERS_ACTIVITY_COLUMNS
baseline_names = []
all_scores = []


if __name__ == "__main__":
    build_result_folder()

    baseline_names = [
        "Total Volume",
        "Gender & Age",
        "Other Activities",
        "Embeddings",
        "Embeddings and Gender & Age",
    ]
    print(f"{len(baseline_names)} baselines to compute")

    # Total Volume
    all_scores.append(compute_total_volume_baseline(cols_to_predict))

    # Gender & Age
    records_cols_input, no_time_series = ["age_group", "gender"], True
    all_scores.append(
        compute_baseline_scores(records_cols_input, no_time_series, cols_to_predict)
    )

    # Other Activities
    all_scores.append(compute_other_activities_baseline(cols_to_predict))

    # Embeddings
    records_cols_input, no_time_series = [], False
    all_scores.append(
        compute_baseline_scores(records_cols_input, no_time_series, cols_to_predict)
    )

    # Embeddings and Gender & Age
    records_cols_input, no_time_series = ["age_group", "gender"], False
    all_scores.append(
        compute_baseline_scores(records_cols_input, no_time_series, cols_to_predict)
    )

    df_results = pd.DataFrame(all_scores, index=baseline_names, columns=ACTIVITY_LABELS)
    save_data(df_results, "results/tables/baselines_results.csv")
    create_image_of_baseline_scores(df_results)
