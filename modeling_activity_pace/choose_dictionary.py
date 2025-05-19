from src.dict_learning.choose_best_iteration import choose_best_dict
from src.dict_learning.dictionary_helpers import process_data_for_DL
from src.dict_learning.figures import make_graphs
from src.helpers import build_result_folder
from src.modeling_functions import split
from src.settings import ANSWERS_ACTIVITY_COLUMNS


def main():
    build_result_folder()
    cols_to_predict = ANSWERS_ACTIVITY_COLUMNS
    S, y = process_data_for_DL()
    S_train, _, y_train_, _ = split(S.T, y)

    choose_best_dict(S_train.T, y_train_, cols_to_predict)
    make_graphs()


if __name__ == "__main__":
    main()
