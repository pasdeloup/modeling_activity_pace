from src.model_analysis.feature_importance import feature_importance_logreg
from src.settings import ANSWERS_ACTIVITY_COLUMNS, ACTIVITY_LABELS
from src.helpers import build_result_folder

cols_to_predict = ANSWERS_ACTIVITY_COLUMNS
cols_to_predict_label = ACTIVITY_LABELS

def main():
    build_result_folder()
    feature_importance_logreg(cols_to_predict, cols_to_predict_label)


if __name__ == "__main__":
    main()
