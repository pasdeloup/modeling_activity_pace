import numpy as np

from sklearn.preprocessing import OneHotEncoder

from alphacsc import BatchCDL
from src.settings import RS


class DictLearning:
    """
    Dictionary Learning Class for Stream Data

    Args:
        signals (ndarray): The input signal data of shape (t_max, n_users, n_channels).
        labels (ndarray): The labels for the input signals.
        n_atoms (int): The number of atoms in the dictionary.
        reg (float): The regularization parameter.
        sqrt_beta (float): The square root of beta parameter.

    Attributes:
        T (ndarray): The input signals.
        t_max (int): The maximum time range.
        n_users (int): The number of users.
        n_channels (int): The number of channels.
        labels (ndarray): The labels for the input signals.
        n_atoms (int): The number of atoms in the dictionary.
        reg (float): The regularization parameter.
        sqrt_beta (float): Classification task importance parameter.

    Methods:
        build_signal_label_matrices():
            Builds the signal and label matrices.

        fit(n_iter, verbose=1):
            Fits the dictionary learning model.

    """

    def __init__(self, signals, labels, n_atoms, reg, sqrt_beta):
        self.S = signals
        self.t_max = signals.shape[0]
        self.n_users = signals.shape[1]
        self.n_channels = signals.shape[2]
        self.labels = labels
        self.n_atoms = n_atoms
        self.reg = reg
        self.sqrt_beta = sqrt_beta

    def build_signal_label_matrices(self):
        """
        Build signal and label matrices.

        This method constructs the signal and label matrices for dictionary learning.

        """
        one_hot_labels = (
            OneHotEncoder(sparse_output=False)
            .fit_transform(self.labels.reshape(-1, 1))
            .T
        )
        Y = np.repeat(one_hot_labels[:, :, np.newaxis], self.n_channels, axis=2)
        self.S_Y = np.vstack((self.S, self.sqrt_beta * Y)).transpose(1, 2, 0)

    def fit(self, n_iter, verbose=1):
        """
        Fit the dictionary learning model.

        Args:
            n_iter (int): Number of iterations.
            verbose (int): Verbosity level (default=1).

        """
        folder_name = "results/dict_learning/all_iterations/"

        self.cdl = BatchCDL(
            self.n_atoms,
            self.S_Y.shape[2],
            rank1=False,
            n_iter=n_iter,
            n_jobs=-1,
            verbose=verbose,
            random_state=RS,
            sort_atoms=True,
            reg=self.reg,
        )
        self.cdl.fit(self.S_Y, folder=folder_name)

        self.D_W = self.cdl.D_hat_
