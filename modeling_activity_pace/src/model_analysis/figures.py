import matplotlib.pyplot as plt
import seaborn as sns

from src.helpers import save_figure


def make_heatmap(df, v, path):
    """
    Create a heatmap from a DataFrame and save it as an image.

    Parameters:
    - df (DataFrame): The DataFrame containing the data to be visualized in the heatmap.
    - v (float or None): If None, using robust=True for color scale. If a float, it sets the symmetric
      range of values for the color scale from -v to v.
    - path (str): The path where the heatmap image will be saved.

    Returns:
    - None
    """
    _, ax = plt.subplots(figsize=(13, 3))
    if v == None:
        sns.heatmap(df, ax=ax, cmap=sns.diverging_palette(220, 20, n=200), robust=True)
    else:
        sns.heatmap(
            df, ax=ax, cmap=sns.diverging_palette(220, 20, n=200), vmin=-v, vmax=v
        )
    save_figure(path)


def write_logreg_report(label, result):
    with open(f"results/tables/stats_logreg/{label}.txt", "w") as file:
        file.write(result)
