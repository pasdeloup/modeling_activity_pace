import pandas as pd
import os

from src.process_raw_data.streams_processor import ProcessStreams
from src.helpers import save_data
from src.settings import (
    TIME_SERIES_PATH,
    channel_names,
    time_labels_full,
    MIN_DATE,
    MAX_DATE,
    N_SUBDIVISION_1HOUR,
    INSTANT_ZERO,
)

if __name__ == "__main__":
    # Create destination directory if it doesn't exist
    os.makedirs(TIME_SERIES_PATH, exist_ok=True)

    # Initialize a ProcessStreams instance
    stream_processor = ProcessStreams("data/streams/", usr_drop_rate=0)

    # Process the streams data
    stream_processor.process(MIN_DATE, MAX_DATE, N_SUBDIVISION_1HOUR, INSTANT_ZERO)

    # Compute time series
    stream_processor.compute_time_series()

    # Save processed data to CSV files for each channel
    for channel_index, channel_name in enumerate(channel_names):
        # Create a DataFrame from the processed data
        channel_data = pd.DataFrame(
            stream_processor.X[:, channel_index, :],
            index=stream_processor.ids,
            columns=time_labels_full,
        )

        # Save the DataFrame to a CSV file
        save_data(channel_data, f"{TIME_SERIES_PATH}/X_{channel_name}.csv", index=True)
