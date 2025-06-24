import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np

from tqdm import tqdm

from src.settings import (
    N_SUBDIVISION_1HOUR,
    INSTANT_ZERO,
    TIME_SERIES_PATH,
    time_labels_full,
    time_labels,
)
from src.helpers import load_data, save_figure, build_result_folder

tqdm.pandas()

build_result_folder()


def add_time_range(df, n_subdivisions, instant_zero):
    time_difference = (df["ts_listen"] - instant_zero).dt.total_seconds()
    df["time_range"] = (time_difference % (n_subdivisions * 3600)) // (
        n_subdivisions * 3600 / n_subdivisions
    )
    return df


# Import data

df_processed = load_data(f"{TIME_SERIES_PATH}/X_volume.csv")

SELECTED_USER_ID = df_processed.index[0]

streams_df_list = []
for i in tqdm(os.listdir("data/streams/")):
    df_streams = load_data("data/streams/" + i).reset_index()
    streams_df_list.append(df_streams[df_streams["user_id"] == SELECTED_USER_ID])

df = pd.concat(streams_df_list)

# Process raw data

df["ts_listen"] = df["ts_listen"].progress_apply(dt.datetime.fromtimestamp)

y = df_processed.T[SELECTED_USER_ID].tolist()

y_norm = y.copy()
y_norm = np.convolve(y_norm, [1, 1, 1], mode="valid")
y_norm = y_norm - np.mean(y_norm)
y_norm = y_norm / np.max(abs(y_norm))

df["date"] = df["ts_listen"].dt.strftime("%Y/%m/%d")

df = add_time_range(df, N_SUBDIVISION_1HOUR, INSTANT_ZERO)

# Plots

plt.figure(figsize=(24, 15))
sns.set_style(rc={"axes.facecolor": "whitesmoke"})

tick_positions = range(0, len(time_labels_full), 12)
tick_labels = [time_labels_full[i] for i in tick_positions]

######################################################################
plt.subplot(3, 1, 1)

sns.histplot(data=df, x="ts_listen", bins=30 * 9, kde=False, color="tab:blue")
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("counts", fontsize=25)
plt.xlabel("")

######################################################################
plt.subplot(3, 1, 2)

sns.histplot(
    data=df, x="time_range", bins=N_SUBDIVISION_1HOUR, kde=False, color="tab:blue"
)
plt.xticks(tick_positions, tick_labels, fontsize=19)
plt.yticks(fontsize=25)
plt.ylabel("aggregated week counts", fontsize=22)
plt.xlabel("")


######################################################################
plt.subplot(3, 1, 3)

sns.lineplot(y=y_norm, x=time_labels, color="tab:blue")
plt.xticks(tick_positions, tick_labels, fontsize=19)
plt.yticks(fontsize=25)
plt.ylabel("normalized and smoothed", fontsize=22)
plt.xlabel("")

######################################################################

save_figure("results/figures/fig1.pdf")
