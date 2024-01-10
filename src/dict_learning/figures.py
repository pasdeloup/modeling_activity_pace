import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.settings import channel_names, time_labels
from src.helpers import save_figure


def plot_channels_for_atom(D, atom_index, output_dir):
    """
    Plot the channels for a specific atom in the dictionary.

    This function takes a dictionary `D`, an atom index, and an output directory and plots the channels
    for the specified atom.

    Args:
        D (numpy.ndarray): The dictionary of atoms.
        atom_index (int): The index of the atom to plot.
        output_dir (str): The output directory to save the plot.

    Returns:
        None
    """
    plt.figure(figsize=(16, 4))
    for channel_index, channel_name in enumerate(channel_names):
        sns.lineplot(
            x=time_labels,
            y=D[atom_index, channel_index, :],
            label=channel_name,
        )
    max_val = np.max(np.abs(D[atom_index, :, :]))
    bound = max_val * 1.1
    plt.ylim(-bound, bound)
    plt.title(f"Atom {atom_index}", fontsize=14)
    plt.xlabel("Time", fontsize=14)
    plt.xticks([time_labels[i] for i in range(7, len(time_labels), 12)])

    plt.legend(loc="upper left", fontsize=14)
    fig_filename = os.path.join(output_dir, f"atom_{atom_index}_channels.pdf")
    save_figure(fig_filename)
    plt.close()


def plot_classification_iterations_perfs(x, y, path):
    """
    Plots the mean relative roc_auc score over dictionary learning iterations.

    This function takes x and y values and plots the mean relative classification score over
    dictionary learning iterations.

    Args:
        x (list): X values (iterations).
        y (list): Y values (mean relative scores).
        path: path to save figure

    Returns:
        None
    """
    plt.figure(figsize=(25, 6))
    sns.set(style="whitegrid")
    sns.lineplot(x=x, y=y)
    plt.title("Mean relative classification score over dict learning iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Mean relative score")
    plt.xticks(list(range(0, len(x), 5)))
    plt.grid(True)
    save_figure(path)

def plot_reconstruction_iterations_perfs(df, path):
    """
    Plots the L2 distance between signals and reconstructed signals over the different channels over the iterations.

    Args:
        df: data to plot
        path: path to save figure

    Returns:
        None
    """
    plt.figure(figsize=(25, 6))
    sns.set(style="whitegrid")
    sns.lineplot(df)
    plt.title("Reconstruction score over dict learning iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    plt.grid(True)
    save_figure(path)

def make_graphs():
    """ """
    D = np.load("results/dict_learning/D.npy")[:, :, :166]
    for atom_index in range(D.shape[0]):
        plot_channels_for_atom(D, atom_index, "results/figures/atoms")
