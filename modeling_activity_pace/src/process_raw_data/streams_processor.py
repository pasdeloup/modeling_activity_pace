import os
import pandas as pd
import numpy as np
import datetime as dt

from tqdm import tqdm
from collections import Counter

from src.helpers import load_data

tqdm.pandas()


class ProcessStreams:
    def __init__(self, path, usr_drop_rate=0) -> None:
        """
        Initialize the ProcessStreams class.

        Args:
            path (str): The path to the directory containing stream files.
            usr_drop_rate (float): The user drop rate for filtering users (0-1).
        """
        self.path = path
        self.usr_drop_rate = usr_drop_rate
        self.liked_df = load_data("data/raw/other/user_favorites.csv").reset_index()

    def import_data(self):
        """
        Import data from CSV files in the specified directory.
        """
        df_list = [
            load_data(os.path.join(self.path, file)).reset_index()
            for file in tqdm(
                os.listdir(self.path), desc="Importing stream files"
            )
        ]
        self.df = pd.concat(df_list)
        # self.df = self.df[self.df["user_id"].isin(self.df.user_id.tolist()[:500] + [3356219324])]

    def convert_timestamps(self):
        """
        Convert timestamp columns to datetime objects.
        """
        self.df["ts_listen"] = self.df["ts_listen"].progress_apply(
            dt.datetime.fromtimestamp
        )

    def filter(self, min_date, max_date):
        """
        Filter the DataFrame based on date and listening time.

        Args:
            min_date (datetime.date): Minimum date.
            max_date (datetime.date): Maximum date.
        """
        self.df = self.df[
            (self.df["ts_listen"].dt.date >= min_date)
            & (self.df["ts_listen"].dt.date <= max_date)
            & (self.df["listening_time"] >= 30)
        ]

    def build_ids_list(self):
        """
        Build a list of user IDs sorted by the number of occurrences.
        """
        ids = [
            i[0] for i in sorted(Counter(self.df.user_id).items(), key=lambda x: -x[1])
        ]
        self.ids = ids

    def add_time_range(self, n_subdivisions, instant_zero):
        """
        Add a time range column to the DataFrame.

        Args:
            n_subdivisions (int): Number of time subdivisions.
            instant_zero (datetime): Reference datetime for time range calculation.
        """

        self.n_subdivisions = n_subdivisions
        self.instant_zero = instant_zero
        time_difference = (self.df["ts_listen"] - instant_zero).dt.total_seconds()
        self.df["time_range"] = (time_difference % (n_subdivisions * 3600)) // (
            n_subdivisions * 3600 / n_subdivisions
        )

    def add_date(self):
        """
        Add a date column to the DataFrame.
        """
        self.df["date"] = self.df["ts_listen"].dt.strftime("%Y/%m/%d")

    def filter_users(self):
        """
        Filter users based on the user drop rate.
        """
        n = len(self.ids)
        keep_n = n - int(n * self.usr_drop_rate)
        self.ids = self.ids[:keep_n]
        self.df = self.df[self.df["user_id"].isin(self.ids)]

    def compute_is_organic(self):
        """
        Compute an 'is_organic' column based on the 'context_4' column.
        """
        self.df["is_organic"] = (self.df["context_4"] == "organic").astype(int)

    def identify_context(self, x):
        """
        Identify context types based on the 'context_type' column.

        Args:
            x (str): The context string.

        Returns:
            str: The identified context type.
        """
        context_keywords = ["album", "albums", "playlist", "playlists"]
        if any(keyword in x for keyword in context_keywords):
            return "album" if "album" in x or "albums" in x else "playlist"
        return "other"

    def convert_context(self):
        """
        Convert the 'context_type' column to 'context_identified'.
        """
        self.df["context_identified"] = self.df["context_type"].progress_apply(
            self.identify_context
        )

    def is_fav(self, row, dict_liked):
        """
        Check if a row is marked as a favorite.

        Args:
            row (pd.Series): The row to check.
            dict_liked (dict): Dictionary of favorite items.

        Returns:
            int: 1 if it's a favorite, 0 otherwise.
        """
        song_id = row.media_id
        if "song" in dict_liked and song_id in dict_liked["song"]:
            return 1
        context_identified = row.context_identified
        context_id = row.context_id
        if (
            context_identified in ["playlist", "album"]
            and context_identified in dict_liked
            and context_id in dict_liked[context_identified]
        ):
            return 1
        return 0

    def process(self, min_date, max_date, n_subdivisions, instant_zero):
        """
        Process the data by importing, filtering, and adding columns.

        Args:
            min_date (str): Minimum date in YYYY-MM-DD format.
            max_date (str): Maximum date in YYYY-MM-DD format.
            n_subdivisions (int): Number of time subdivisions.
            instant_zero (datetime): Reference datetime for time range calculation.
        """
        print("import data")
        self.import_data()
        print("convert timestamps")
        self.convert_timestamps()
        # print("filter")
        # self.filter(min_date, max_date)  # Done in the data
        print("build ids list")
        self.build_ids_list()
        print("filter users")
        self.filter_users()
        print("add time range")
        self.add_time_range(n_subdivisions, instant_zero)
        print("add date")
        self.add_date()
        # print("compute is organic")
        # self.compute_is_organic()
        # print("convert context")
        # self.convert_context()
        self.all_time_date_couples = set(
            [tuple(i) for i in self.df[["time_range", "date"]].to_numpy()]
        )

    def compute_full_df_user(self, df_user):
        """
        Compute the full DataFrame for a user with zero values when he/she didn't listen to any tracks.

        Args:
            df_user (pd.DataFrame): The user's DataFrame.

        Returns:
            pd.DataFrame: The full DataFrame with zero values.
        """
        df_user = (
            df_user[["time_range", "ts_listen", "date"]]
            .groupby(["time_range", "date"])
            .count()
            .reset_index()
        )

        df_user = df_user.to_numpy()
        user_time_date_couples = set([(i[0], i[1]) for i in df_user])
        df_user_list = list([list(i) for i in df_user])

        for tr, date in self.all_time_date_couples - user_time_date_couples:
            df_user_list.append([tr, date, 0])

        full_df_user = pd.DataFrame(
            df_user_list, columns=["time_range", "date", "ts_listen"]
        ).sort_values(by=["time_range", "date"])
        return full_df_user

    def compute_time_serie_from_full_df_user(self, full_df_user):
        """
        Compute the time series from the full user DataFrame.

        Args:
            full_df_user (pd.DataFrame): The full user DataFrame.

        Returns:
            list: The computed time series.
        """
        y_dict = dict(
            full_df_user[["time_range", "ts_listen"]]
            .groupby("time_range")
            .mean()
            .reset_index()
            .to_numpy()
        )
        for i in range(self.n_subdivisions):
            if not (i in y_dict.keys()):
                y_dict[i] = 0

        y = [i[1] for i in sorted(y_dict.items(), key=lambda x: x[0])]
        return y

    def add_volume(self, df_user, k, channel_index):
        """
        Add counts time serie to the X array.

        Args:
            df_user (pd.DataFrame): The user's DataFrame.
            k (int): Index of the user.
            channel_index (int): Index of the channel.
        """
        self.full_df_user = self.compute_full_df_user(df_user)
        y = self.compute_time_serie_from_full_df_user(self.full_df_user)
        self.X[k, channel_index, :] = y

    def zero_div(self, a, b):
        """
        Handle division by zero gracefully.

        Args:
            a: Numerator.
            b: Denominator.

        Returns:
            str or float: 'nan' or the result of division.
        """
        if b == 0:
            return "nan"
        else:
            return a / b

    def compute_ratio_df(self, df_user):
        """
        Compute the ratio column (e.g ratio of favorited streams over all streams for each time range).

        Args:
            df_user (pd.DataFrame): The user's DataFrame.

        Returns:
            pd.DataFrame: The computed ratio DataFrame.
        """
        full_df_ratio = self.compute_full_df_user(df_user)
        full_df_ratio = full_df_ratio.rename(columns={"ts_listen": "specific_ts"})
        full_df_ratio["ts_listen"] = self.full_df_user["ts_listen"].tolist()
        full_df_ratio["ratio"] = full_df_ratio.apply(
            lambda row: self.zero_div(row.specific_ts, row.ts_listen), axis=1
        )
        full_df_ratio = full_df_ratio[full_df_ratio["ratio"] != "nan"]
        full_df_ratio["ts_listen"] = full_df_ratio["ratio"]
        return full_df_ratio

    def add_repetition(self, df_user, k, channel_index, n=3):
        """
        Add repetitions time serie to the X array.

        Args:
            df_user (pd.DataFrame): The user's DataFrame.
            k (int): Index of the user.
            channel_index (int): Index of the channel.
            n (int): Threshold for considering repetitions.
        """
        non_discovery = [
            i[0] for i in Counter(df_user.media_id.tolist()).items() if i[1] > n
        ]
        df_user["is_repet"] = df_user["media_id"].apply(
            lambda x: 1 if x in non_discovery else 0
        )
        df_user = df_user[df_user["is_repet"] == 1]
        full_df_rep = self.compute_ratio_df(df_user)
        y = self.compute_time_serie_from_full_df_user(full_df_rep)

        self.X[k, channel_index, :] = y

    def add_organicity(self, df_user, k, channel_index):
        """
        Add organic time serie to the X array.

        Args:
            df_user (pd.DataFrame): The user's DataFrame.
            k (int): Index of the user.
            channel_index (int): Index of the channel.
        """
        df_user = df_user[df_user["is_organic"] == 1]
        full_df_orga = self.compute_ratio_df(df_user)
        y = self.compute_time_serie_from_full_df_user(full_df_orga)
        self.X[k, channel_index, :] = y

    def add_liked(self, df_user, k, id, channel_index):
        """
        Add fav time serie to the X array.

        Args:
            df_user (pd.DataFrame): The user's DataFrame.
            k (int): Index of the user.
            id (int): User ID.
            channel_index (int): Index of the channel.
        """
        liked_df_user = self.liked_df[self.liked_df["user_id"] == id]
        dict_liked = dict(
            liked_df_user[
                liked_df_user["item_type"].isin(["album", "playlist", "song"])
            ][["item_id", "item_type"]]
            .groupby("item_type")
            .agg(list)
            .reset_index()
            .values
        )
        df_user["is_fav"] = df_user.apply(
            lambda row: self.is_fav(row, dict_liked), axis=1
        )

        df_user = df_user[df_user["is_fav"] == 1]
        full_df_liked = self.compute_ratio_df(df_user)
        y = self.compute_time_serie_from_full_df_user(full_df_liked)
        self.X[k, channel_index, :] = y

    def compute_time_series(self):
        """
        Compute time series for all users and channels.
        """
        self.X = np.zeros((len(self.ids), 4, self.n_subdivisions))
        for k, id in tqdm(list(enumerate(self.ids)), desc="Computing time series"):
            df_user = self.df[self.df["user_id"] == id].copy()
            self.add_volume(df_user, k, 0)
            self.add_repetition(df_user, k, 1)
            self.add_organicity(df_user, k, 2)
            self.add_liked(df_user, k, id, 3)
