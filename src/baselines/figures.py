import matplotlib.pyplot as plt
from pandas.plotting import table

from src.helpers import save_figure


def create_image_of_baseline_scores(df):
    """
    Create an image of the table of baseline scores from a DataFrame and save it.

    Parameters:
    - df (DataFrame): The DataFrame containing the baseline scores to be displayed in the image.

    Returns:
    - None
    """
    _, ax = plt.subplots(figsize=(20, 4))
    ax.axis("off")
    tab = table(ax, df, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(12)
    tab.scale(1, 1.5)
    save_figure("results/figures/baselines_results.pdf")
