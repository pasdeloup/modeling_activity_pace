import numpy as np

from src.dict_learning.dictionary_helpers import process_data_for_DL
from src.dict_learning.multi_dict_learning import DictLearning
from src.helpers import build_result_folder
from src.modeling_functions import split

def main():
    build_result_folder()

    n_atoms = 32
    sqrt_beta = 0
    reg = 1e-16
    n_iter = 100

    S, y = process_data_for_DL()
    S_train, _, y_train_, _ = split(
        S.T,
        y,
    )
    S_train = np.stack(S_train.T, axis=2)
    labels = np.array([str(i) for i in y_train_])

    dl = DictLearning(
        signals=S_train, labels=labels, n_atoms=n_atoms, reg=reg, sqrt_beta=sqrt_beta
    )
    dl.build_signal_label_matrices()
    dl.fit(n_iter=n_iter)


if __name__ == "__main__":
    main()
